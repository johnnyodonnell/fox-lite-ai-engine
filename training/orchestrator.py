"""Strict-synchronous self-play / train orchestrator (Phase 5).

Loop (no overlap -> zero GPU contention):
  1. run selfplay_rs (fresh process, current weights) -> a cohort file
  2. REINFORCE update over the cohort
  3. publish new weights (serving_weights.safetensors) for the next self-play
  4. save latest checkpoint (resume)
  5. every --snapshot-every: snapshot (safetensors + onnx) + run the Rust evaluator

Resumes from <out-dir>/latest.pt. Runs indefinitely until killed.
"""

import argparse
import datetime
import glob
import json
import os
import select
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from cohort import read_cohort
from export import export_onnx, save_weights_st
from net import FoxNet, N_BLOCKS, WIDTH, n_params
from train import train_on_cohort

HERE = Path(__file__).resolve().parent
SELFPLAY_BIN = HERE / "selfplay_rs" / "target" / "release" / "selfplay_rs"
EVAL_BIN = HERE / "selfplay_rs" / "target" / "release" / "evaluate_rs"


def parse_duration(spec: str) -> float:
    s = str(spec).strip()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


def worker_env() -> dict:
    """libtorch + CUDA libs on LD_LIBRARY_PATH for the Rust subprocess."""
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    sp = os.path.dirname(os.path.dirname(torch.__file__))  # site-packages
    libs = [torch_lib] + sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(libs) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


# A cohort takes seconds; an ack this late means the worker is wedged, not slow.
# The protocol is one line per request, so the stdout buffer is empty whenever we
# select() — no buffered-line/select mismatch.
ACK_TIMEOUT_SEC = 300


def read_ack(worker, timeout, what):
    """One protocol line from the worker, or raise after `timeout` seconds so a
    wedged worker (e.g. an internal deadlock) degrades to a respawn instead of
    stalling the daemon forever."""
    ready, _, _ = select.select([worker.stdout], [], [], timeout)
    if not ready:
        raise RuntimeError(f"selfplay worker {what} timeout ({timeout:.0f}s)")
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError("selfplay worker died")
    return line


def start_selfplay_worker(env):
    """Spawn the persistent `selfplay-serve` worker (CUDA/libtorch init once) and
    block until it reports ready. Returns the Popen handle."""
    p = subprocess.Popen(
        [str(SELFPLAY_BIN), "selfplay-serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        cwd=str(HERE), env=env, text=True, bufsize=1,
    )
    line = read_ack(p, ACK_TIMEOUT_SEC, "ready")
    if not json.loads(line).get("ready"):
        raise RuntimeError(f"selfplay worker failed to start (got {line!r})")
    return p


def run_cohort(worker, **cmd):
    """Send one cohort command to the worker and block until it writes the cohort
    file and acks `done`. Raises if the worker died or wedged (the selfplay()
    wrapper respawns it and retries once)."""
    worker.stdin.write(json.dumps(cmd) + "\n")
    worker.stdin.flush()
    line = read_ack(worker, ACK_TIMEOUT_SEC, "ack")
    if not json.loads(line).get("done"):
        raise RuntimeError(f"selfplay worker bad ack: {line!r}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/run1")
    ap.add_argument("--snapshot-every", default="30m")
    ap.add_argument("--matches", type=int, default=2048, help="matches per cohort")
    ap.add_argument("--selfplay-batch", type=int, default=1024, help="GPU forward width")
    # Parallel self-play pipeline: N game-worker threads feed one batched-inference
    # thread (CPU game logic overlaps the GPU forward). 0 concurrency => 2x batch.
    ap.add_argument("--selfplay-threads", type=int, default=16, help="self-play worker threads")
    ap.add_argument("--selfplay-concurrency", type=int, default=0,
                    help="games kept in flight (0 = 2x batch); > batch overlaps CPU with GPU")
    # Forwards in flight: the inference thread launches the next batch while a
    # scatter thread reads back the previous forward (keeps the GPU fed). 2 = double-buffer.
    ap.add_argument("--selfplay-slots", type=int, default=2,
                    help="GPU forwards kept in flight (inference runs ahead of scatter)")
    ap.add_argument("--temperature", type=float, default=1.0)
    # Large minibatches => only a handful of SGD steps per cohort, so the policy
    # stays close to the behavior policy that generated the on-policy cohort.
    ap.add_argument("--sgd-batch", type=int, default=65536)
    ap.add_argument("--epochs", type=int, default=1, help="passes over each cohort")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--c-value", type=float, default=1.0)
    ap.add_argument("--c-entropy", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-cohorts", type=int, default=0, help="0 = run forever")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--eval-games", type=int, default=200, help="matches per opponent")
    ap.add_argument("--n-top", type=int, default=2, help="top-rated models kept as opponents")
    ap.add_argument("--n-anchors", type=int, default=3,
                    help="frozen snapshots spread across the Elo range as opponents")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    snapshot_interval = parse_duration(args.snapshot_every)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    net = FoxNet().to(device)
    # Train through a compiled wrapper (shares parameters with `net`); keep the
    # raw module for state_dict/safetensors/ONNX so the key names stay the plain
    # FQNs the Rust forward loads (no `_orig_mod.` prefix).
    net_train = torch.compile(net)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    base_elapsed, total_cohorts, total_games, total_steps = 0.0, 0, 0, 0
    next_snapshot_at = snapshot_interval
    latest = out_dir / "latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["weights"])
        opt.load_state_dict(ckpt["opt"])
        base_elapsed = ckpt.get("elapsed_sec", 0.0)
        total_cohorts = ckpt.get("cohorts", 0)
        total_games = ckpt.get("games", 0)
        total_steps = ckpt.get("train_steps", 0)
        next_snapshot_at = ckpt.get("next_snapshot_at", snapshot_interval)
        if "np_rng" in ckpt:
            rng.bit_generator.state = ckpt["np_rng"]
        print(f"resumed from {latest} (elapsed={base_elapsed/3600:.2f}h "
              f"cohorts={total_cohorts} games={total_games})", flush=True)
    else:
        torch.manual_seed(args.seed)
        print(f"cold start (seed={args.seed})", flush=True)

    print(f"net: width={WIDTH} blocks={N_BLOCKS} params={n_params(net):,}", flush=True)

    serving_st = out_dir / "serving_weights.safetensors"
    cohort_path = out_dir / "cohort.bin"
    save_weights_st(net, str(serving_st))  # initial weights for the first self-play

    start = time.time()

    def elapsed():
        return base_elapsed + (time.time() - start)

    def save_latest():
        tmp = latest.with_suffix(".tmp")
        torch.save({
            "weights": net.state_dict(),
            "opt": opt.state_dict(),
            "elapsed_sec": elapsed(),
            "cohorts": total_cohorts,
            "games": total_games,
            "train_steps": total_steps,
            "next_snapshot_at": next_snapshot_at,
            "np_rng": rng.bit_generator.state,
        }, tmp)
        tmp.replace(latest)

    def save_snapshot():
        hours = int(elapsed() / 3600)
        utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"snap_h{hours:05d}_{utc}"
        st_path = out_dir / "snapshots" / f"{stem}.safetensors"
        onnx_path = out_dir / "snapshots" / f"{stem}.onnx"
        save_weights_st(net, str(st_path))
        export_onnx(net, str(onnx_path))
        print(f"[snapshot] {st_path.name}", flush=True)
        return st_path

    def run_eval(st_path):
        # The Rust evaluator owns the pool/Elo bookkeeping: it picks the active
        # opponent set (top-N + random + spread anchors) from pool.json, plays
        # the matches, refits a global Bradley-Terry Elo, and writes pool.json.
        # It streams its own `[eval]` logs to our stdout.
        if args.no_eval or not EVAL_BIN.exists():
            return
        try:
            subprocess.run(
                [str(EVAL_BIN), "--run-dir", str(out_dir), "--candidate", str(st_path),
                 "--games", str(args.eval_games), "--n-top", str(args.n_top),
                 "--n-anchors", str(args.n_anchors), "--seed", str(total_cohorts)],
                cwd=str(HERE), env=worker_env(), check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"[eval] failed (exit {e.returncode})", flush=True)

    env = worker_env()
    concurrency = args.selfplay_concurrency or 2 * args.selfplay_batch
    worker = start_selfplay_worker(env)

    def selfplay(seed):
        """One cohort via the persistent worker. On worker death, respawn and
        retry once so a crash degrades to fork-per-cohort rather than killing
        the daemon."""
        nonlocal worker
        cmd = dict(
            weights=str(serving_st), out=str(cohort_path),
            matches=args.matches, batch=args.selfplay_batch,
            concurrency=concurrency, threads=args.selfplay_threads,
            slots=args.selfplay_slots, temperature=args.temperature, seed=seed,
        )
        try:
            run_cohort(worker, **cmd)
        except (RuntimeError, BrokenPipeError, ValueError) as e:
            print(f"[selfplay] worker error ({e}); respawning", flush=True)
            try:
                worker.kill()
            except ProcessLookupError:
                pass
            worker = start_selfplay_worker(env)
            run_cohort(worker, **cmd)

    while True:
        t0 = time.time()
        seed = args.seed + total_cohorts * 7919 + 1
        selfplay(seed)
        sp_sec = time.time() - t0

        cohort = read_cohort(str(cohort_path))
        t1 = time.time()
        stats = train_on_cohort(
            net_train, opt, cohort, device,
            sgd_batch=args.sgd_batch, epochs=args.epochs,
            c_value=args.c_value, c_entropy=args.c_entropy, rng=rng,
        )
        tr_sec = time.time() - t1

        save_weights_st(net, str(serving_st))
        total_cohorts += 1
        total_games += args.matches
        total_steps += stats["steps"]
        save_latest()

        if elapsed() >= next_snapshot_at:
            st_path = save_snapshot()
            next_snapshot_at += snapshot_interval
            run_eval(st_path)

        print(json.dumps({
            "t": round(elapsed(), 1),
            "cohorts": total_cohorts,
            "games": total_games,
            "rows": cohort["n"],
            "tr_steps": total_steps,
            "z_mean": round(float(cohort["z"].mean()), 3),
            "sp_sec": round(sp_sec, 2),
            "tr_sec": round(tr_sec, 2),
            "loss": round(stats["loss"], 4),
            "policy": round(stats["policy"], 4),
            "value": round(stats["value"], 4),
            "entropy": round(stats["entropy"], 4),
            "next_snap_in": round(next_snapshot_at - elapsed(), 1),
        }), flush=True)

        if args.max_cohorts and total_cohorts >= args.max_cohorts:
            print("reached --max-cohorts; stopping", flush=True)
            break


if __name__ == "__main__":
    main()

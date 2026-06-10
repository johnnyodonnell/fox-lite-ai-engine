"""Continuous AlphaZero/ISMCTS self-play orchestrator.

Runs the trainer (PyTorch, paced SGD) in this process and spawns ONE standalone
Rust self-play worker (`selfplay_rs serve`) that holds the net on the GPU, runs
ISMCTS self-play 100% in Rust, and streams finished games back over stdout as
length-prefixed frames. The trainer drains those into a ring ReplayBuffer, runs
SGD paced to the data rate (KataGo-style sample reuse), and republishes weights
to `serving_weights.safetensors`, which the worker hot-reloads on mtime change.

Every `--snapshot-every` it snapshots (safetensors + ONNX) and launches a
detached `evaluate_rs` (ISMCTS evaluation + Elo into pool.json). Resumes from
`<out-dir>/latest.pt`; a fresh run can warm-start from `--init-from` (a `.pt`
checkpoint or a raw `.safetensors` snapshot — e.g. the best run1 snapshot).
"""

import argparse
import datetime
import glob
import json
import os
import queue as pyqueue
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

from encode import INPUT_SIZE, NUM_CARDS
from export import export_onnx, save_weights_st
from net import FoxNet, N_BLOCKS, WIDTH, n_params
from train import ReplayBuffer, train_step

HERE = Path(__file__).resolve().parent
SELFPLAY_BIN = HERE / "selfplay_rs" / "target" / "release" / "selfplay_rs"
EVAL_BIN = HERE / "selfplay_rs" / "target" / "release" / "evaluate_rs"
ROW_FLOATS = INPUT_SIZE + NUM_CARDS + 1  # state[230] + pi[33] + z = 264


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
    """libtorch + CUDA libs on LD_LIBRARY_PATH for the Rust subprocesses."""
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    sp = os.path.dirname(os.path.dirname(torch.__file__))  # site-packages
    libs = [torch_lib] + sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(libs) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def _pid_alive(pid: int) -> bool:
    """True if `pid` is a live (non-zombie) process. A finished-but-unreaped
    detached eval lingers as a zombie whose PID still answers kill(0); treat
    those as dead so a stale lock never blocks every future eval."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
        if data[data.rindex(")") + 1:].split()[0] == "Z":
            return False
    except (OSError, ValueError, IndexError):
        pass
    return True


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/run3")
    ap.add_argument("--init-from", default=None,
                    help="Warm-start a fresh run from a .pt checkpoint or a raw "
                         ".safetensors snapshot (weights only; fresh clock). Only "
                         "used on cold start (ignored once latest.pt exists).")
    ap.add_argument("--snapshot-every", default="30m")
    ap.add_argument("--save-latest-every", default="300s")
    ap.add_argument("--publish-every", default="15s",
                    help="How often to republish serving_weights.safetensors "
                         "(the worker hot-reloads it on mtime change).")
    # self-play worker (ISMCTS)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--threads", type=int, default=16, help="MCTS worker threads")
    ap.add_argument("--slots", type=int, default=2, help="GPU forwards in flight")
    ap.add_argument("--selfplay-batch", type=int, default=512, help="GPU forward width")
    ap.add_argument("--queue-max", type=int, default=64)
    # training (AlphaZero supervised: CE(pi) + MSE(z))
    ap.add_argument("--buffer-capacity", type=int, default=2_000_000,
                    help="Replay window in rows (~1KB each). Must span hours of "
                         "self-play / many model versions for stable AlphaZero "
                         "training; 200k was ~80s of data at live throughput.")
    ap.add_argument("--min-buffer-for-train", type=int, default=2_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--target-reuse", type=float, default=4.0,
                    help="Avg times each generated sample is trained on; paces SGD "
                         "to the self-play data rate (KataGo's tuned cap).")
    ap.add_argument("--max-steps-per-cycle", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=float, default=30.0)
    # evaluation (ISMCTS)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-aoti", action="store_true",
                    help="disable the fused AOTInductor self-play forward (use eager bf16)")
    ap.add_argument("--eval-games", type=int, default=200, help="matches per opponent")
    ap.add_argument("--eval-sims", type=int, default=200,
                    help="ISMCTS eval is synchronous (batch-1); at a full opponent "
                         "pool an eval can outlast the snapshot interval, in which "
                         "case the eval.lock skips the overlapping snapshot's eval")
    ap.add_argument("--n-top", type=int, default=2)
    ap.add_argument("--n-anchors", type=int, default=3)
    return ap.parse_args()


def export_serving_pt2(net, path, batch, device):
    """Export a COPY of `net` to an AOTInductor .pt2 (fused, bf16, STATIC batch) for
    the self-play worker's forward. Exported once; the worker hot-swaps fresh weights
    from the safetensors sidecar, so only the architecture/batch need to be right.
    Never mutates the training net (works on a fresh FoxNet copy)."""
    import torch._inductor  # lazily, so a torch without inductor still trains (eager)
    m = FoxNet()
    m.load_state_dict(net.state_dict())  # copy current weights (cast to the copy)
    m.eval().to(device).bfloat16()
    x = torch.zeros(batch, INPUT_SIZE, device=device, dtype=torch.bfloat16)
    # Keep constants runtime-updatable so the worker can swap weights with no recompile.
    torch._inductor.config.aot_inductor.use_runtime_constant_folding = True
    with torch.no_grad():
        ep = torch.export.export(m, (x,))
        torch._inductor.aoti_compile_and_package(ep, package_path=str(path))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    snapshot_interval = parse_duration(args.snapshot_every)
    save_latest_interval = parse_duration(args.save_latest_every)
    publish_interval = parse_duration(args.publish_every)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    net = FoxNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buf = ReplayBuffer(args.buffer_capacity)
    rng = np.random.default_rng(args.seed)

    # ----- Resume from latest.pt, else warm-start from --init-from, else cold -----
    base_elapsed = 0.0
    total_games = 0
    total_train_steps = 0
    next_snapshot_at = snapshot_interval
    latest = out_dir / "latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["weights"])
        opt.load_state_dict(ckpt["opt"])
        base_elapsed = ckpt.get("elapsed_sec", 0.0)
        total_games = ckpt.get("games", 0)
        total_train_steps = ckpt.get("train_steps", 0)
        next_snapshot_at = ckpt.get("next_snapshot_at", snapshot_interval)
        if "np_rng" in ckpt:
            rng.bit_generator.state = ckpt["np_rng"]
        print(f"resumed from {latest} (elapsed={base_elapsed/3600:.2f}h "
              f"games={total_games} steps={total_train_steps})", flush=True)
    elif args.init_from:
        _load_init(net, opt, args.init_from, device)
        torch.manual_seed(args.seed)
        print(f"warm start from {args.init_from} (fresh clock; seed={args.seed})", flush=True)
    else:
        torch.manual_seed(args.seed)
        print(f"cold start (seed={args.seed})", flush=True)

    print(f"net: width={WIDTH} blocks={N_BLOCKS} params={n_params(net):,}", flush=True)

    serving_st = out_dir / "serving_weights.safetensors"
    save_weights_st(net, str(serving_st))  # initial weights for the first self-play

    # Export the fused AOTInductor forward once; the worker loads it and hot-swaps
    # fresh weights on each publish. On any failure, fall back to the eager bf16 path.
    serving_model = None
    if not args.no_aoti and device.type == "cuda":
        serving_model = out_dir / "serving_model.pt2"
        try:
            t0 = time.time()
            export_serving_pt2(net, serving_model, args.selfplay_batch, device)
            print(f"exported AOTI {serving_model.name} (batch={args.selfplay_batch}) "
                  f"in {time.time() - t0:.1f}s", flush=True)
        except Exception as e:
            print(f"WARNING: AOTI export failed ({e}); using eager bf16 forward", flush=True)
            serving_model = None

    start = time.time()

    def elapsed():
        return base_elapsed + (time.time() - start)

    def save_latest():
        tmp = latest.with_suffix(".tmp")
        torch.save({
            "weights": net.state_dict(),
            "opt": opt.state_dict(),
            "elapsed_sec": elapsed(),
            "games": total_games,
            "train_steps": total_train_steps,
            "next_snapshot_at": next_snapshot_at,
            "np_rng": rng.bit_generator.state,
        }, tmp)
        tmp.replace(latest)

    def save_snapshot():
        hours = int(elapsed() / 3600)
        utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"snap_h{hours:05d}_{utc}"
        st_path = out_dir / "snapshots" / f"{stem}.safetensors"
        save_weights_st(net, str(st_path))
        export_onnx(net, str(out_dir / "snapshots" / f"{stem}.onnx"))
        print(f"[snapshot] {st_path.name}", flush=True)
        return st_path

    def launch_eval(st_path):
        if args.no_eval or not EVAL_BIN.exists():
            return
        # Skip if a prior eval is still running (ISMCTS eval can outlast a short
        # snapshot interval; piling up evals would thrash the GPU vs self-play).
        lock = out_dir / "eval.lock"
        if lock.exists():
            try:
                if _pid_alive(int(lock.read_text().split()[0])):
                    print(f"[eval] prior eval still running; skipping {st_path.name}", flush=True)
                    return
            except (ValueError, IndexError, OSError):
                pass
        logf = open(out_dir / "eval.log", "a")
        try:
            proc = subprocess.Popen(
                [str(EVAL_BIN), "--run-dir", str(out_dir), "--candidate", str(st_path),
                 "--games", str(args.eval_games), "--sims", str(args.eval_sims),
                 "--n-top", str(args.n_top), "--n-anchors", str(args.n_anchors),
                 "--seed", str(total_train_steps)],
                cwd=str(HERE), stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True, env=worker_env(),
            )
        finally:
            logf.close()
        (out_dir / "eval.lock").write_text(f"{proc.pid} {time.time()}")
        print(f"[eval] launched evaluate_rs pid={proc.pid} for {st_path.name}", flush=True)

    # ----- Spawn the Rust self-play worker + a reader thread -----
    out_queue = pyqueue.Queue(maxsize=args.queue_max)
    stop = threading.Event()

    def read_exact(f, n):
        chunks = bytearray()
        while len(chunks) < n:
            b = f.read(n - len(chunks))
            if not b:
                return None
            chunks += b
        return bytes(chunks)

    def reader_loop(stdout):
        while True:
            hdr = read_exact(stdout, 4)
            if hdr is None:
                return
            (n_rows,) = struct.unpack("<I", hdr)
            payload = read_exact(stdout, n_rows * ROW_FLOATS * 4)
            if payload is None:
                return
            flat = np.frombuffer(payload, dtype="<f4").reshape(n_rows, ROW_FLOATS)
            rows = [(r[:INPUT_SIZE].copy(),
                     r[INPUT_SIZE:INPUT_SIZE + NUM_CARDS].copy(),
                     float(r[-1])) for r in flat]
            while not stop.is_set():
                try:
                    out_queue.put((rows, 1), timeout=0.5)
                    break
                except pyqueue.Full:
                    pass

    def start_selfplay():
        if not SELFPLAY_BIN.exists():
            raise SystemExit(f"self-play binary not found: {SELFPLAY_BIN} "
                             f"(build it: cd selfplay_rs && cargo build --release)")
        cmd = [str(SELFPLAY_BIN), "serve",
               "--weights", str(serving_st),
               "--threads", str(args.threads),
               "--sims", str(args.sims),
               "--slots", str(args.slots),
               "--batch", str(args.selfplay_batch),
               "--seed", str(args.seed + 1000)]
        if serving_model is not None:
            cmd += ["--model", str(serving_model)]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=worker_env(),
        )
        t = threading.Thread(target=reader_loop, args=(proc.stdout,), daemon=True)
        t.start()
        fwd = "AOTI" if serving_model is not None else "eager-bf16"
        print(f"spawned selfplay_rs worker pid={proc.pid} ({fwd}; "
              f"threads={args.threads} sims={args.sims} batch={args.selfplay_batch})", flush=True)
        return proc

    selfplay_proc = start_selfplay()

    last_log = last_latest_save = last_publish = time.time()
    while next_snapshot_at <= elapsed():
        next_snapshot_at += snapshot_interval
    print(f"snapshot every {args.snapshot_every} (next at "
          f"elapsed={next_snapshot_at/3600:.2f}h)", flush=True)

    last_loss = None
    gen_samples = 0   # tuples generated this session (reuse pacing)
    session_steps = 0  # SGD steps this session (reuse pacing)
    net.train()
    try:
        while True:
            if selfplay_proc.poll() is not None:
                print("warn: self-play worker died; restarting", flush=True)
                selfplay_proc = start_selfplay()

            if elapsed() >= next_snapshot_at:
                st_path = save_snapshot()
                save_latest()
                last_latest_save = time.time()
                next_snapshot_at += snapshot_interval
                launch_eval(st_path)
                continue

            # Drain finished games into the ring buffer.
            got_any = False
            for _ in range(args.queue_max):
                try:
                    rows, n_games = out_queue.get_nowait()
                except pyqueue.Empty:
                    break
                buf.add_many(rows)
                total_games += n_games
                gen_samples += len(rows)
                got_any = True
            if len(buf) < args.min_buffer_for_train and not got_any:
                try:
                    rows, n_games = out_queue.get(timeout=1.0)
                    buf.add_many(rows)
                    total_games += n_games
                    gen_samples += len(rows)
                except pyqueue.Empty:
                    pass

            # SGD paced to the data rate (cumulative samples-trained near
            # target_reuse * samples-generated).
            cycle_tr_time = 0.0
            allowed = int(gen_samples * args.target_reuse / args.batch_size) - session_steps
            n_steps = max(0, min(allowed, args.max_steps_per_cycle))
            if len(buf) >= args.min_buffer_for_train and n_steps > 0:
                t0 = time.time()
                losses = [train_step(net, opt, buf.sample(args.batch_size, rng=rng), device)
                          for _ in range(n_steps)]
                cycle_tr_time = time.time() - t0
                total_train_steps += n_steps
                session_steps += n_steps
                last_loss = {
                    "loss": round(float(np.mean([l["loss"] for l in losses])), 4),
                    "policy": round(float(np.mean([l["policy_loss"] for l in losses])), 4),
                    "value": round(float(np.mean([l["value_loss"] for l in losses])), 4),
                    "entropy": round(float(np.mean([l["target_entropy"] for l in losses])), 4),
                }
            elif not got_any:
                time.sleep(0.05)

            now = time.time()
            if now - last_publish >= publish_interval:
                save_weights_st(net, str(serving_st))
                last_publish = now
            if now - last_latest_save >= save_latest_interval:
                save_latest()
                last_latest_save = now
            if now - last_log >= args.log_every:
                print(json.dumps({
                    "t": round(elapsed(), 1),
                    "games": total_games,
                    "buf": len(buf),
                    "tr_steps": total_train_steps,
                    "reuse": round(session_steps * args.batch_size / max(1, gen_samples), 2),
                    "qsize": out_queue.qsize(),
                    "tr_sec": round(cycle_tr_time, 2),
                    "loss": last_loss,
                    "next_snap_in": round(next_snapshot_at - elapsed(), 1),
                }), flush=True)
                last_log = now
    finally:
        print("shutting down: signaling self-play", flush=True)
        stop.set()
        try:
            selfplay_proc.stdin.close()  # EOF -> worker shuts down
        except Exception:
            pass
        selfplay_proc.terminate()
        try:
            selfplay_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            selfplay_proc.kill()


def _load_init(net, opt, path, device):
    """Warm-start weights from a .pt checkpoint or a raw .safetensors snapshot."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        net.load_state_dict(load_file(str(path), device=str(device)))
        return
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["weights"])
    if "opt" in ckpt:
        try:
            opt.load_state_dict(ckpt["opt"])
        except (ValueError, KeyError) as e:
            print(f"warn: skipped optimizer state from --init-from ({e})", flush=True)


if __name__ == "__main__":
    main()

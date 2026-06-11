#!/usr/bin/env python
"""One REINFORCE training step in isolation (the orchestrator's middle stage).

Builds a FoxNet + AdamW, optionally loads weights, then runs `train_on_cohort`
`--iters` times over a cohort, timing each pass. The cohort is either a real
self-play file (`--cohort`) or a synthetic one of the correct shape
(`--synthetic N`) so the training step can be profiled with no dependency on
self-play.

  python train_step.py --cohort runs/iso/cohort.bin --iters 5
  python train_step.py --synthetic 200000 --sgd-batch 65536 --iters 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAIN_DIR))

import torch  # noqa: E402

from cohort import read_cohort  # noqa: E402
from encode import INPUT_SIZE, NUM_CARDS  # noqa: E402
from export import load_weights_st, save_weights_st  # noqa: E402
from net import FoxNet, N_BLOCKS, WIDTH, n_params  # noqa: E402
from train import train_on_cohort  # noqa: E402


def synthetic_cohort(n: int, seed: int) -> dict:
    """A random cohort with the on-disk layout's invariants: a legal mask with
    >=1 legal move per row, an action that is legal, and z in {-1, +1}."""
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((n, INPUT_SIZE)).astype(np.float32)
    masks = (rng.random((n, NUM_CARDS)) < 0.4).astype(np.float32)
    empty = masks.sum(axis=1) == 0
    masks[empty, rng.integers(0, NUM_CARDS, size=int(empty.sum()))] = 1.0
    # one legal index per row: argmax of noise over legal cells (illegal -> -1, so
    # a legal index always wins even when its noise draw is 0.0).
    noise = np.where(masks > 0, rng.random((n, NUM_CARDS)), -1.0)
    actions = noise.argmax(axis=1).astype(np.int64)
    z = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    return {"states": states, "masks": masks, "actions": actions, "z": z, "n": n}


def parse_args():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--cohort", help="self-play cohort .bin to train on")
    src.add_argument("--synthetic", type=int, metavar="N",
                     help="generate an N-row synthetic cohort instead")
    ap.add_argument("--weights", help="starting weights .safetensors (else cold init)")
    ap.add_argument("--out", help="write updated weights here after the last iter")
    ap.add_argument("--iters", type=int, default=1, help="training passes over the cohort")
    ap.add_argument("--sgd-batch", type=int, default=65536)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--c-value", type=float, default=1.0)
    ap.add_argument("--c-entropy", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    if args.synthetic:
        cohort = synthetic_cohort(args.synthetic, args.seed)
        print(f"[train] synthetic cohort rows={cohort['n']}", flush=True)
    elif args.cohort:
        cohort = read_cohort(args.cohort)
        print(f"[train] cohort={args.cohort} rows={cohort['n']}", flush=True)
    else:
        raise SystemExit("pass --cohort PATH or --synthetic N")

    net = FoxNet().to(device)
    if args.weights:
        load_weights_st(net, args.weights, device=device)
        print(f"[train] loaded weights={args.weights}", flush=True)
    else:
        print("[train] cold-init weights", flush=True)
    print(f"[train] device={device} net: width={WIDTH} blocks={N_BLOCKS} "
          f"params={n_params(net):,}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    rows = cohort["n"]
    for i in range(args.iters):
        t0 = time.time()
        stats = train_on_cohort(
            net, opt, cohort, device,
            sgd_batch=args.sgd_batch, epochs=args.epochs,
            c_value=args.c_value, c_entropy=args.c_entropy, rng=rng,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"[train] iter {i + 1}/{args.iters} "
              f"steps={stats['steps']} {dt:.2f}s "
              f"({rows / dt:,.0f} rows/s)  "
              f"loss={stats['loss']:.4f} policy={stats['policy']:.4f} "
              f"value={stats['value']:.4f} entropy={stats['entropy']:.4f}",
              flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_weights_st(net, args.out)
        print(f"[train] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

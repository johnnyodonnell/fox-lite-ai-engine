"""End-to-end AlphaZero training loop for Fox in the Forest Lite.

Each iteration:
  1. Self-play `GAMES_PER_ITER` matches with the current checkpoint,
     appending all decision tuples to the replay buffer.
  2. Train `TRAIN_STEPS_PER_ITER` minibatches from the buffer.
  3. Run an arena gate vs `best.pt`. Promote if win rate >= ARENA_WIN_RATE.

The arena gate is critical: AlphaZero loops occasionally regress silently;
a 55% gate filters those out. The challenger always becomes the next
self-play opponent, so we still explore — only `best.pt` is conservative.

Run:
    python training/scripts/run_training.py
    python training/scripts/run_training.py --iterations 2 --games-per-iter 4   # smoke test
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, TRAINING_DIR)

import config  # noqa: E402
from alphazero.arena import play_series  # noqa: E402
from alphazero.network import PolicyValueNet  # noqa: E402
from alphazero.replay_buffer import ReplayBuffer  # noqa: E402
from alphazero.selfplay import play_one_match  # noqa: E402
from alphazero.train import train_iteration  # noqa: E402

CHECKPOINT_DIR = os.path.join(TRAINING_DIR, "checkpoints")
BEST_PATH = os.path.join(CHECKPOINT_DIR, "best.pt")
LATEST_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=config.NUM_ITERATIONS)
    p.add_argument("--games-per-iter", type=int, default=config.GAMES_PER_ITER)
    p.add_argument("--arena-games", type=int, default=config.ARENA_GAMES)
    p.add_argument("--device", default=None,
                   help="cuda / cpu (default: auto)")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  iterations={args.iterations}  "
          f"games/iter={args.games_per_iter}  arena games={args.arena_games}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    challenger = PolicyValueNet().to(device)
    if os.path.exists(BEST_PATH):
        print(f"loading existing best.pt from {BEST_PATH}")
        challenger.load_state_dict(torch.load(BEST_PATH, map_location=device))
    else:
        print("no best.pt — initializing fresh net")
        torch.save(challenger.state_dict(), BEST_PATH)

    optimizer = torch.optim.AdamW(
        challenger.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    buffer = ReplayBuffer(config.BUFFER_SIZE)

    for it in range(1, args.iterations + 1):
        t0 = time.time()
        # --- self-play ----------------------------------------------------
        challenger.eval()
        new_records = 0
        for g in range(args.games_per_iter):
            records = play_one_match(challenger, rng)
            buffer.add_many(records)
            new_records += len(records)
        sp_time = time.time() - t0

        # --- train --------------------------------------------------------
        t1 = time.time()
        stats = train_iteration(challenger, buffer, optimizer, rng, device)
        tr_time = time.time() - t1

        torch.save(challenger.state_dict(), LATEST_PATH)

        # --- arena --------------------------------------------------------
        t2 = time.time()
        champion = PolicyValueNet().to(device)
        champion.load_state_dict(torch.load(BEST_PATH, map_location=device))
        champion.eval()
        result = play_series(challenger, champion, args.arena_games, rng)
        ar_time = time.time() - t2

        promoted = result["challenger_win_rate"] >= config.ARENA_WIN_RATE
        if promoted:
            torch.save(challenger.state_dict(), BEST_PATH)

        print(
            f"iter {it:3d}  buf={len(buffer)}  "
            f"sp={sp_time:.0f}s/{new_records}rec  "
            f"tr={tr_time:.0f}s  p_loss={stats.policy_loss:.3f}  v_loss={stats.value_loss:.3f}  "
            f"ar={ar_time:.0f}s  wr={result['challenger_win_rate']*100:.0f}%  "
            f"margin={result['mean_point_margin']:+.1f}  "
            f"{'PROMOTED' if promoted else ''}"
        )


if __name__ == "__main__":
    main()

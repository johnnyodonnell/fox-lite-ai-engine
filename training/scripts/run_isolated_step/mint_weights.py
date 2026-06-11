#!/usr/bin/env python
"""Write a fresh cold-init FoxNet to a safetensors file.

Lets the isolated self-play / eval steps run with zero prior artifacts: when no
input weights exist, the wrappers mint a cold-start net here.

  python mint_weights.py runs/iso/weights.safetensors [--seed 42]
"""

import argparse
import sys
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAIN_DIR))

import torch  # noqa: E402

from export import save_weights_st  # noqa: E402
from net import FoxNet, N_BLOCKS, WIDTH, n_params  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    net = FoxNet()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_weights_st(net, args.out)
    print(f"[mint] wrote {args.out} "
          f"(width={WIDTH} blocks={N_BLOCKS} params={n_params(net):,})")


if __name__ == "__main__":
    main()

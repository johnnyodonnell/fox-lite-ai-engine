"""Promote a snapshot to a self-contained browser ONNX.

  python training/promote.py --snapshot runs/run1/snapshots/snap_XXXX.safetensors \
                             --out /tmp/current.onnx

Then copy --out to the web app's public/models/current.onnx and commit.
"""

import argparse

from safetensors.torch import load_file

from export import export_onnx
from net import foxnet_for_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="snapshot .safetensors")
    ap.add_argument("--out", required=True, help="output .onnx (self-contained)")
    args = ap.parse_args()

    net = foxnet_for_state(load_file(args.snapshot))
    export_onnx(net, args.out)
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()

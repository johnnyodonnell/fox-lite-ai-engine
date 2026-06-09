"""Benchmark batched FoxNet forward passes to find the best inference batch size.

Sweeps a range of batch sizes, runs warmup + timed iterations for each, and
reports per-batch latency and throughput (forwards/sec). Throughput is the
metric that matters for self-play (we want to maximize forwards/sec); latency
matters if a single decision must come back quickly.

Usage (on the GPU host, inside the training venv):
    .venv/bin/python bench_forward_batch.py
    .venv/bin/python bench_forward_batch.py --dtype bf16 --max-batch 65536
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from encode import INPUT_SIZE
from net import FoxNet


def time_batch(net, x, iters, warmup):
    """Return median per-iteration wall time (seconds) for forwarding x."""
    # Warmup (also triggers cuDNN/cuBLAS autotuning and allocator growth).
    for _ in range(warmup):
        net(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        net(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "tf32", "fp16", "bf16"],
                    default="fp32",
                    help="fp32 matches the Rust tch self-play forward; tf32 "
                         "keeps fp32 storage but allows TF32 matmuls.")
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--max-batch", type=int, default=32768)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    autocast_dtype = None
    if args.dtype == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.dtype == "fp16":
        autocast_dtype = torch.float16
    elif args.dtype == "bf16":
        autocast_dtype = torch.bfloat16

    net = FoxNet().to(device).eval()
    n_params = sum(p.numel() for p in net.parameters())

    print(f"# device   : {torch.cuda.get_device_name(0)}")
    print(f"# torch     : {torch.__version__}")
    print(f"# dtype     : {args.dtype}")
    print(f"# input_size: {INPUT_SIZE}")
    print(f"# params    : {n_params:,}")
    print(f"# iters/warmup: {args.iters}/{args.warmup}")
    print()

    batches = []
    b = args.min_batch
    while b <= args.max_batch:
        batches.append(b)
        b *= 2

    header = f"{'batch':>8} {'lat_ms':>10} {'us/sample':>11} {'fwd/sec':>14} {'speedup':>9}"
    print(header)
    print("-" * len(header))

    base_per_sample = None
    results = []
    for bs in batches:
        x = torch.randn(bs, INPUT_SIZE, device=device)
        # Scale iters down for big batches so each size takes similar wall time.
        iters = max(5, min(args.iters, args.iters * 1024 // bs)) if bs > 1024 else args.iters
        try:
            with torch.no_grad():
                if autocast_dtype is not None:
                    with torch.autocast("cuda", dtype=autocast_dtype):
                        lat = time_batch(net, x, iters, args.warmup)
                else:
                    lat = time_batch(net, x, iters, args.warmup)
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>8}  OOM")
            torch.cuda.empty_cache()
            break

        per_sample = lat / bs
        if base_per_sample is None:
            base_per_sample = per_sample
        thr = bs / lat
        results.append((bs, lat, per_sample, thr))
        print(f"{bs:>8} {lat * 1e3:>10.3f} {per_sample * 1e6:>11.3f} "
              f"{thr:>14,.0f} {base_per_sample / per_sample:>8.1f}x")
        del x
        torch.cuda.empty_cache()

    if results:
        best = max(results, key=lambda r: r[3])
        print()
        print(f"# peak throughput: batch={best[0]} -> {best[3]:,.0f} forwards/sec "
              f"({best[2] * 1e6:.3f} us/sample, {best[1] * 1e3:.3f} ms/batch)")
        # "Knee": smallest batch reaching >=95% of peak per-sample efficiency.
        peak_eff = best[3]
        knee = next((r for r in results if r[3] >= 0.95 * peak_eff), best)
        print(f"# 95%-of-peak knee: batch={knee[0]} -> {knee[3]:,.0f} forwards/sec "
              f"({knee[1] * 1e3:.3f} ms/batch latency)")


if __name__ == "__main__":
    main()

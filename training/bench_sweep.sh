#!/usr/bin/env bash
# Self-play-only throughput sweep over GPU forward batch sizes.
# Stops nothing; assumes fox-train is already stopped so the GPU is free.
# Usage: bench_sweep.sh "<space-separated batch sizes>" [run_secs] [interval_secs]
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
source .venv/bin/activate

# Same LD_LIBRARY_PATH the orchestrator builds for the Rust subprocesses.
export LD_LIBRARY_PATH="$(python -c 'import os,glob,torch; tl=os.path.join(os.path.dirname(torch.__file__),"lib"); sp=os.path.dirname(os.path.dirname(torch.__file__)); print(":".join([tl]+sorted(glob.glob(os.path.join(sp,"nvidia","*","lib")))))'):${LD_LIBRARY_PATH:-}"

BIN="$DIR/selfplay_rs/target/release/selfplay_rs"
WEIGHTS="${WEIGHTS:-/home/johnny/Workspace/fox-lite/runs/run2/serving_weights.safetensors}"
THREADS="${THREADS:-16}"
SLOTS="${SLOTS:-2}"
SIMS="${SIMS:-200}"
WARMUP="${WARMUP:-90}"
MEASURE="${MEASURE:-150}"
RUN_SECS=$(( WARMUP + MEASURE ))
INTERVAL="${INTERVAL:-15}"

for B in $1; do
  echo "########## batch=$B ##########"
  "$BIN" bench --weights "$WEIGHTS" --threads "$THREADS" --slots "$SLOTS" \
    --sims "$SIMS" --batch "$B" --seed 1042 \
    --run-secs "$RUN_SECS" --interval-secs "$INTERVAL" --warmup-secs "$WARMUP"
  echo
done

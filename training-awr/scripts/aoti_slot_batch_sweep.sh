#!/usr/bin/env bash
# AOTI (fused, production-path) self-play-only throughput sweep over GPU forward
# batch sizes x inference slots. Unlike bench_sweep.sh (which benches the EAGER
# forward), this exports a static-batch .pt2 per batch size and runs the bench
# through the fused AOTInductor forward the live worker actually uses.
#
# Assumes fox-train is stopped so the GPU is free.
# Usage: aoti_slot_batch_sweep.sh "<batches>" "<slots>" [warmup] [measure]
set -euo pipefail

DIR=/home/johnny/Workspace/fox-lite/fox-lite-ai-engine/training
cd "$DIR"
source .venv/bin/activate

# Same LD_LIBRARY_PATH the orchestrator builds for the Rust subprocesses.
export LD_LIBRARY_PATH="$(python -c 'import os,glob,torch; tl=os.path.join(os.path.dirname(torch.__file__),"lib"); sp=os.path.dirname(os.path.dirname(torch.__file__)); print(":".join([tl]+sorted(glob.glob(os.path.join(sp,"nvidia","*","lib")))))'):${LD_LIBRARY_PATH:-}"

BIN="$DIR/selfplay_rs/target/release/selfplay_rs"
WEIGHTS="${WEIGHTS:-/home/johnny/Workspace/fox-lite/runs/run3/serving_weights.safetensors}"
THREADS="${THREADS:-16}"
SIMS="${SIMS:-200}"
WARMUP="${3:-60}"
MEASURE="${4:-90}"
INTERVAL="${INTERVAL:-30}"
RUN_SECS=$(( WARMUP + MEASURE ))

BATCHES="${1:-1024 2048 4096}"
SLOTS_LIST="${2:-4 8 16}"

echo "=== AOTI slot/batch sweep  weights=$WEIGHTS threads=$THREADS sims=$SIMS warmup=${WARMUP}s measure=${MEASURE}s ==="
for B in $BATCHES; do
  PT2="/tmp/sm_${B}.pt2"
  if [[ ! -f "$PT2" ]]; then
    echo "### export pt2 batch=$B -> $PT2"
    ( cd selfplay_rs && python export_pt2.py "$WEIGHTS" "$PT2" "$B" )
  fi
  for S in $SLOTS_LIST; do
    echo "########## batch=$B slots=$S ##########"
    "$BIN" bench --weights "$WEIGHTS" --model "$PT2" \
      --threads "$THREADS" --slots "$S" --sims "$SIMS" --batch "$B" --seed 1042 \
      --run-secs "$RUN_SECS" --interval-secs "$INTERVAL" --warmup-secs "$WARMUP"
    echo
  done
done
echo "=== sweep done ==="

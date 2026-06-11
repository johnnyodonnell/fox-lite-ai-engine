#!/usr/bin/env bash
# Run ONE evaluation step in isolation: a candidate snapshot vs the active pool.
#
#   scripts/run_isolated_step/evaluate.sh
#   CANDIDATE=runs/run1/snapshots/snap_x.safetensors GAMES=200 \
#       scripts/run_isolated_step/evaluate.sh
#
# Defaults to a SCRATCH run-dir (runs/iso/eval) so it never mutates a real
# pool.json; with a fresh pool the only opponent is `random`. Point RUN_DIR at a
# real run to evaluate against its accumulated pool — note this WILL append to and
# refit that run's pool.json. Mints a cold-start candidate if none exists.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_bin "$EVAL_BIN"

RUN_DIR="${RUN_DIR:-$ISO_SCRATCH/eval}"
CANDIDATE="${CANDIDATE:-$ISO_SCRATCH/weights.safetensors}"
GAMES="${GAMES:-100}"
N_TOP="${N_TOP:-2}"
N_ANCHORS="${N_ANCHORS:-3}"
SEED="${SEED:-0}"
mkdir -p "$RUN_DIR"

if [[ ! -f "$CANDIDATE" ]]; then
  echo "[eval] $CANDIDATE missing — minting cold-start candidate"
  python "$ISO_DIR/mint_weights.py" "$CANDIDATE" --seed 42
fi

echo "[eval] candidate=$CANDIDATE run-dir=$RUN_DIR games=$GAMES n-top=$N_TOP n-anchors=$N_ANCHORS"
t0="$(now)"
"$EVAL_BIN" --run-dir "$RUN_DIR" --candidate "$CANDIDATE" \
  --games "$GAMES" --n-top "$N_TOP" --n-anchors "$N_ANCHORS" --seed "$SEED" "$@"
echo "[eval] done in $(elapsed "$t0")s"

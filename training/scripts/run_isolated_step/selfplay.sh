#!/usr/bin/env bash
# Run ONE self-play step in isolation: weights -> cohort.bin (+ throughput).
#
#   scripts/run_isolated_step/selfplay.sh
#   WEIGHTS=runs/run1/serving_weights.safetensors MATCHES=2048 BATCH=2048 \
#       scripts/run_isolated_step/selfplay.sh
#
# Mints cold-start weights when WEIGHTS is unset and the default doesn't exist,
# so this runs with zero prior artifacts. Extra args pass through to selfplay_rs
# (e.g. --cpu for a no-GPU smoke test).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_bin "$SELFPLAY_BIN"

WEIGHTS="${WEIGHTS:-$ISO_SCRATCH/weights.safetensors}"
OUT="${OUT:-$ISO_SCRATCH/cohort.bin}"
MATCHES="${MATCHES:-512}"
BATCH="${BATCH:-1024}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-1}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "[selfplay] $WEIGHTS missing — minting cold-start weights"
  python "$ISO_DIR/mint_weights.py" "$WEIGHTS" --seed "$SEED"
fi

echo "[selfplay] weights=$WEIGHTS matches=$MATCHES batch=$BATCH temp=$TEMPERATURE seed=$SEED"
t0="$(now)"
"$SELFPLAY_BIN" selfplay \
  --weights "$WEIGHTS" --out "$OUT" \
  --matches "$MATCHES" --batch "$BATCH" \
  --temperature "$TEMPERATURE" --seed "$SEED" "$@"
dt="$(elapsed "$t0")"
awk -v m="$MATCHES" -v dt="$dt" 'BEGIN{printf "[selfplay] done in %ss (%.1f matches/s) -> '"$OUT"'\n", dt, m/dt}'
python cohort.py "$OUT" || true

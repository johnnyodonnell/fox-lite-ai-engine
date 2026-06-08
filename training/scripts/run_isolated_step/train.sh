#!/usr/bin/env bash
# Run ONE training step (REINFORCE update) in isolation over a cohort.
#
#   scripts/run_isolated_step/train.sh                 # real cohort if present, else synthetic
#   COHORT=runs/run1/cohort.bin ITERS=5 scripts/run_isolated_step/train.sh
#   SYNTHETIC=200000 ITERS=10 scripts/run_isolated_step/train.sh   # perf test, no self-play needed
#
# Defaults to the shared scratch cohort (runs/iso/cohort.bin, written by
# selfplay.sh); if it's absent and SYNTHETIC is unset, falls back to a synthetic
# cohort so the step still runs standalone. Extra args pass through to
# train_step.py. Set OUT to persist the updated weights.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

COHORT="${COHORT:-$ISO_SCRATCH/cohort.bin}"

ARGS=()
if [[ -n "${SYNTHETIC:-}" ]]; then
  ARGS+=(--synthetic "$SYNTHETIC")
elif [[ -f "$COHORT" ]]; then
  ARGS+=(--cohort "$COHORT")
else
  echo "[train] no cohort at $COHORT — using a synthetic one (run selfplay.sh first, or set COHORT/SYNTHETIC)"
  ARGS+=(--synthetic "200000")
fi

exec python "$ISO_DIR/train_step.py" \
  "${ARGS[@]}" \
  ${WEIGHTS:+--weights "$WEIGHTS"} \
  ${OUT:+--out "$OUT"} \
  --iters "${ITERS:-1}" \
  --sgd-batch "${SGD_BATCH:-65536}" \
  --epochs "${EPOCHS:-1}" \
  --lr "${LR:-1e-3}" \
  --c-value "${C_VALUE:-1.0}" \
  --c-entropy "${C_ENTROPY:-0.05}" \
  --seed "${SEED:-42}" \
  "$@"

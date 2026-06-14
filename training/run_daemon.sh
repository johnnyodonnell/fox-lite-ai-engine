#!/usr/bin/env bash
# Indefinite self-play training daemon (strict-synchronous loop, 30-min snapshots).
#
#   cd training && ./run_daemon.sh            # foreground
#   OUT_DIR=runs/run1 nohup ./run_daemon.sh > runs/run1.log 2>&1 &   # detached
#
# Resumes from <OUT_DIR>/latest.pt if present. The orchestrator sets
# LD_LIBRARY_PATH for the Rust self-play / eval subprocesses itself.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
export SELFPLAY_PYTHON="$DIR/.venv/bin/python"

exec python orchestrator.py \
  --out-dir "${OUT_DIR:-runs/run1}" \
  --snapshot-every "${SNAPSHOT_EVERY:-30m}" \
  --matches "${MATCHES:-2048}" \
  --selfplay-batch "${SELFPLAY_BATCH:-1024}" \
  --selfplay-threads "${SELFPLAY_THREADS:-16}" \
  --sgd-batch "${SGD_BATCH:-65536}" \
  --lr "${LR:-1e-3}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --temp-end "${TEMP_END:-0.5}" \
  --alpha-init "${ALPHA_INIT:-0.05}" \
  --alpha-lr "${ALPHA_LR:-0.02}" \
  --ent-target-frac "${ENT_TARGET_FRAC:-0.5}" \
  --eval-games "${EVAL_GAMES:-200}" \
  --n-top "${N_TOP:-2}" \
  --n-anchors "${N_ANCHORS:-3}" \
  --seed "${SEED:-42}"

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
  --selfplay-batch "${SELFPLAY_BATCH:-2048}" \
  --sgd-batch "${SGD_BATCH:-65536}" \
  --lr "${LR:-1e-3}" \
  --c-entropy "${C_ENTROPY:-0.05}" \
  --eval-games "${EVAL_GAMES:-200}" \
  --eval-pool "${EVAL_POOL:-8}" \
  --seed "${SEED:-42}"

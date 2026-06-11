#!/usr/bin/env bash
# pm2 entrypoint for indefinite AWR self-play training (continuous search-free
# loop: streaming self-play -> replay buffer -> paced AWR SGD, 30m snapshots +
# raw-policy eval).
#
#   OUT_DIR=$HOME/Workspace/fox-lite/runs/run4 \
#     pm2 start training-awr/run_daemon.sh --name fox-train-awr --kill-timeout 20000
#
# Resumes from <OUT_DIR>/latest.pt if present. INIT_FROM warm-starts a fresh run
# (weights only) from a .pt checkpoint or a raw .safetensors snapshot; it is
# ignored once latest.pt exists. The orchestrator sets LD_LIBRARY_PATH for the
# Rust self-play / eval subprocesses itself.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
export SELFPLAY_PYTHON="$DIR/.venv/bin/python"

INIT_ARGS=()
if [[ -n "${INIT_FROM:-}" ]]; then
  INIT_ARGS=(--init-from "$INIT_FROM")
fi

exec python orchestrator.py \
  --out-dir "${OUT_DIR:-runs/run4}" \
  --snapshot-every "${SNAPSHOT_EVERY:-30m}" \
  --threads "${THREADS:-19}" \
  --slots "${SLOTS:-4}" \
  --selfplay-batch "${SELFPLAY_BATCH:-4096}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --lr "${LR:-2e-3}" \
  --beta "${BETA:-1.0}" \
  --w-max "${W_MAX:-20.0}" \
  --entropy-coef "${ENTROPY_COEF:-0.0}" \
  --buffer-capacity "${BUFFER_CAPACITY:-10000000}" \
  --eval-games "${EVAL_GAMES:-200}" \
  --seed "${SEED:-42}" \
  "${INIT_ARGS[@]}"

#!/usr/bin/env bash
# pm2 entrypoint for indefinite ISMCTS self-play training (continuous AlphaZero
# loop: streaming self-play -> replay buffer -> paced SGD, 4h snapshots + eval).
#
#   OUT_DIR=.../runs/run3 INIT_FROM=.../runs/run1/snapshots/snap_h00019_*.safetensors \
#     pm2 start training/run_daemon.sh --name fox-train --kill-timeout 20000
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
  --out-dir "${OUT_DIR:-runs/run3}" \
  --snapshot-every "${SNAPSHOT_EVERY:-4h}" \
  --sims "${SIMS:-200}" \
  --threads "${THREADS:-19}" \
  --slots "${SLOTS:-4}" \
  --selfplay-batch "${SELFPLAY_BATCH:-4096}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --lr "${LR:-2e-3}" \
  --eval-games "${EVAL_GAMES:-200}" \
  --eval-sims "${EVAL_SIMS:-200}" \
  --seed "${SEED:-42}" \
  "${INIT_ARGS[@]}"

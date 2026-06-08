# _common.sh — shared setup for the isolated-step scripts. Source me, don't exec.
#
# Activates training/.venv and puts libtorch + the CUDA runtime libs on
# LD_LIBRARY_PATH so the Rust binaries (selfplay_rs / evaluate_rs) link at run
# time — this mirrors orchestrator.worker_env(). Also defines the path vars and
# a scratch dir that the wrappers share, so the three steps can chain by default
# yet each still runs standalone.

set -euo pipefail

ISO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "$ISO_DIR/../.." && pwd)"
cd "$TRAIN_DIR"

if [[ ! -f "$TRAIN_DIR/.venv/bin/activate" ]]; then
  echo "error: $TRAIN_DIR/.venv missing — see training/README.md 'Setup'." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$TRAIN_DIR/.venv/bin/activate"
export SELFPLAY_PYTHON="$TRAIN_DIR/.venv/bin/python"

# libtorch + CUDA runtime libs for the Rust subprocesses (== worker_env()).
LD_LIBRARY_PATH="$(python - <<'PY'
import glob, os, torch
lib = os.path.join(os.path.dirname(torch.__file__), "lib")
sp = os.path.dirname(os.path.dirname(torch.__file__))
nvidia = sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
print(":".join([lib, *nvidia]))
PY
)":"${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH

SELFPLAY_BIN="$TRAIN_DIR/selfplay_rs/target/release/selfplay_rs"
EVAL_BIN="$TRAIN_DIR/selfplay_rs/target/release/evaluate_rs"
ISO_SCRATCH="${ISO_SCRATCH:-$TRAIN_DIR/runs/iso}"
mkdir -p "$ISO_SCRATCH"

require_bin() {
  if [[ ! -x "$1" ]]; then
    echo "error: $(basename "$1") not built — run: (cd selfplay_rs && cargo build --release)" >&2
    exit 1
  fi
}

# elapsed <start-epoch-ns>  ->  echoes seconds with one decimal
now() { date +%s.%N; }
elapsed() { awk -v s="$1" -v e="$(now)" 'BEGIN{printf "%.1f", e - s}'; }

# Isolated training steps

Run each of the three core pieces of the self-play loop **on its own** so you can
profile and optimize them independently. Each script reproduces exactly what the
orchestrator does for that stage (`orchestrator.py`), but as a single shot with
timing — and each is self-sufficient (it mints/synthesizes its own inputs if the
upstream artifact isn't there).

All three share `_common.sh`, which activates `training/.venv` and sets
`LD_LIBRARY_PATH` for the Rust binaries (== `orchestrator.worker_env()`). Run them
from anywhere; outputs land in the scratch dir `runs/iso/` (override with
`ISO_SCRATCH`). Prereqs: the venv and the Rust binaries are built — see
`training/README.md` "Setup".

## The three steps

```sh
cd training

# 1. Self-play: weights -> cohort.bin (prints matches/s)
scripts/run_isolated_step/selfplay.sh

# 2. Training: cohort -> REINFORCE update (prints rows/s + loss breakdown)
scripts/run_isolated_step/train.sh

# 3. Evaluation: candidate snapshot vs the active pool (prints Elo + win%)
scripts/run_isolated_step/evaluate.sh
```

With defaults they also chain: `selfplay.sh` writes `runs/iso/weights.safetensors`
+ `runs/iso/cohort.bin`, which `train.sh` then consumes.

## Knobs (env vars)

Each script reads env vars and passes any extra CLI args straight through to the
underlying binary/driver.

| Step | Common knobs | Notes |
|------|--------------|-------|
| `selfplay.sh` | `WEIGHTS` `OUT` `MATCHES` `BATCH` `TEMPERATURE` `SEED` | mints cold weights if `WEIGHTS` absent; `--cpu` passes through |
| `train.sh` | `COHORT` `SYNTHETIC=N` `WEIGHTS` `OUT` `ITERS` `SGD_BATCH` `EPOCHS` `LR` `C_VALUE` `C_ENTROPY` | `SYNTHETIC=N` profiles training with no self-play dependency |
| `evaluate.sh` | `CANDIDATE` `RUN_DIR` `GAMES` `N_TOP` `N_ANCHORS` `SEED` | scratch `RUN_DIR` by default so a real `pool.json` is never touched |

## Examples

```sh
# Self-play throughput at the production batch size:
MATCHES=2048 BATCH=2048 scripts/run_isolated_step/selfplay.sh

# Training-step speed only, decoupled from self-play, 10 timed passes:
SYNTHETIC=300000 ITERS=10 scripts/run_isolated_step/train.sh

# Evaluate a real snapshot against a real run's pool (this updates that pool.json):
CANDIDATE=runs/run1/snapshots/snap_h00019_*.safetensors RUN_DIR=runs/run1 \
    GAMES=200 scripts/run_isolated_step/evaluate.sh
```

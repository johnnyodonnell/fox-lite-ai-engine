# Fox-Lite AWR self-play training

**Search-free AWR** (Advantage-Weighted Regression, [Peng et al. 2019](https://arxiv.org/abs/1910.00177))
self-play for the Fox-in-the-Forest-Lite engine. 100% Rust self-play +
evaluator (`tch`/libtorch on GPU), Python trainer, continuous streaming loop
(self-play → replay buffer → paced SGD), 30-minute snapshots.

Forked from `training/` (the run3 AlphaZero/ISMCTS setup) after the run3
postmortem: search wasn't amplifying the net (44/100 vs its own raw policy),
and the deployed browser engine is a single forward pass anyway — so run4
trains what we deploy. Self-play picks moves with **no search**: one forward
pass per decision, sampling from the legal-masked policy at the match-annealed
temperature (1.0 → 0.25). Each decision's `(state, legal_mask, action)` plus
the match outcome `z = ±1` becomes a training row; the AWR update is

    adv         = z - value(state).detach()
    w           = exp(adv / beta).clamp(max=w_max)
    policy_loss = -(w * log pi(action|state)).mean()
    value_loss  = MSE(value(state), z)

i.e. the original run1 REINFORCE step with a clipped exponential weight in
place of the linear advantage — and, unlike REINFORCE, explicitly off-policy:
the big multi-model-version replay buffer is part of the algorithm.
Evaluation (`evaluate_rs --sims 1`) and the deployed browser engine
(`src/engine/neural.js`) both play raw-policy argmax.

## Layout

```
foxlite_core/     Rust crate: Lite rules + canonical encoder
  src/mcts.rs         ISMCTS primitives (used by evaluate_rs only; self-play is search-free)
selfplay_rs/      Rust: self-play worker (selfplay_rs serve/bench) + evaluator (evaluate_rs)
  src/pipeline.rs     continuous decision-batched search-free pipeline (frame stream)
net.py            PyTorch residual MLP (policy + value) — unchanged arch
encode.py         canonical encoder (parity reference)
train.py          AWR update: ReplayBuffer + exp-weighted policy regression + MSE(v, z)
orchestrator.py   continuous loop: stream games -> replay buffer -> paced SGD -> (snapshot + eval)
export.py         self-contained ONNX export + safetensors I/O
promote.py        snapshot .safetensors -> browser ONNX
elo.py            legacy one-shot Elo fit (superseded by evaluate_rs's global refit)
runs/             (gitignored) weights, snapshots/, serving_weights.safetensors, pool.json
```

## Setup (asus-nvidia / GB10)

```sh
cd training-awr
ln -s ../training/.venv .venv   # or create one as in training/README.md

# Build the Rust binaries (tch links the venv's libtorch).
source .venv/bin/activate
export SELFPLAY_PYTHON=$(which python)
(cd selfplay_rs && cargo build --release)   # builds selfplay_rs + evaluate_rs
```

`.cargo/config.toml` sets `LIBTORCH_USE_PYTORCH=1` and
`LIBTORCH_BYPASS_VERSION_CHECK=1`; activate the venv before building so `python3`
is the project interpreter.

## Train

Managed by pm2 (`fox-train-awr`). Env vars drive `run_daemon.sh`; run4 is a
cold start (no `INIT_FROM`):

```sh
OUT_DIR=$HOME/Workspace/fox-lite/runs/run4 \
  pm2 start training-awr/run_daemon.sh --name fox-train-awr --kill-timeout 20000
```

Resumes from `<OUT_DIR>/latest.pt`. The trainer streams finished games from the
Rust worker into a ring replay buffer and runs SGD paced to the data rate
(`--target-reuse`); search-free generation is ~100x ISMCTS rates, so expect to
be trainer-bound (logged `reuse` below target = data fresher than required).
Watch progress:

```sh
pm2 logs fox-train-awr         # JSON lines: games/buf/tr_steps/loss{policy,value,entropy,w_mean}
cat  $HOME/Workspace/fox-lite/runs/run4/pool.json   # models + ratings + match results
cat  $HOME/Workspace/fox-lite/runs/run4/eval.log    # detached evaluate_rs [eval] lines
```

Health signals: `entropy` collapsing toward 0 → set `ENTROPY_COEF=0.01`;
`w_mean` should sit near 1 early (drifting to 0 or pinned at `W_MAX` means
`BETA` is mis-scaled).

Quick self-play throughput probe (no trainer):

```sh
./selfplay_rs/target/release/selfplay_rs bench \
  --weights $HOME/Workspace/fox-lite/runs/run4/serving_weights.safetensors \
  --threads 16 --batch 512
```

Each snapshot is evaluated by `evaluate_rs` at `--sims 1` (raw-policy argmax,
matching deployment): it picks an active opponent set (top-`--n-top` rated +
`random` + `--n-anchors` frozen snapshots spread across the Elo range), plays
`--eval-games` per opponent, then refits a global Bradley-Terry Elo over all
accumulated results (random pinned at 0) into `pool.json`. Promotion to the
browser model stays manual (see below).

## Promote a model to the browser (manual)

```sh
python promote.py --snapshot $HOME/Workspace/fox-lite/runs/run4/snapshots/snap_XXXX.safetensors --out /tmp/current.onnx
# then, on the web-app host:
scp asus-nvidia:/tmp/current.onnx public/models/current.onnx   # and commit
```

## Tests / parity gates

```sh
# Generate the (gitignored) parity fixtures first:
node scripts/dump_rules_traces.mjs 300 > foxlite_core/tests/rules_traces.json
node scripts/dump_encode_fixtures.mjs 60 > fixtures/encode_fixtures.json

(cd foxlite_core && cargo test)                 # rules + encoder parity vs game.js
(cd selfplay_rs && cargo test)                  # incl. search-free row legality
python test_encode_parity.py                    # Python encoder vs JS reference
python make_forward_fixture.py && python test_onnx_parity.py   # ONNX vs PyTorch
./selfplay_rs/target/release/selfplay_rs forward-check ./fixtures  # Rust tch vs PyTorch
node ../training/scripts/test_browser_engine_e2e.mjs   # real ONNX + JS engine, legal moves
```

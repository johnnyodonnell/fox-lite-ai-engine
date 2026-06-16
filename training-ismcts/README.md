# Fox-Lite self-play training

**AlphaZero-style ISMCTS** self-play for the Fox-in-the-Forest-Lite engine. 100%
Rust self-play + evaluator (`tch`/libtorch on GPU), Python trainer, continuous
streaming loop (self-play → replay buffer → paced SGD), 4-hour snapshots.

Self-play picks moves by **Information-Set MCTS**: PUCT with the policy head as
prior and the value head at the leaves, re-determinizing the hidden cards
(opponent hand + unused pile) every simulation (true ISMCTS). The search-improved
root visit distribution becomes the policy target. The value target `z` is awarded
**per round**: a round that does not decide the match is scored on its normalized
point differential (`(self_pts − opp_pts) / 6` ∈ [-1, 1]), while the match-deciding
round uses the match outcome `z=±1` (its own points don't matter).
The deployed browser engine (`src/engine/neural.js`) still runs the
exported ONNX with a **single forward pass — no search**; evaluation, by contrast,
uses ISMCTS.

## Layout

```
foxlite_core/     Rust crate: Lite rules + canonical encoder
  src/determinize.rs  ISMCTS determinization (sample a hidden world from an info set)
  src/mcts.rs         ISMCTS primitives: availability-PUCT, seat-aware backprop, run_search
selfplay_rs/      Rust: self-play worker (selfplay_rs `serve`/`bench`) + evaluator (evaluate_rs)
  src/pipeline.rs     continuous leaf-batched ISMCTS pipeline (tch fp32 forward, frame stream)
net.py            PyTorch residual MLP (policy + value) — unchanged arch
encode.py         canonical encoder (parity reference)
train.py          AlphaZero update: ReplayBuffer + CE(pi, visits) + MSE(v, z)
orchestrator.py   continuous loop: stream games -> replay buffer -> paced SGD -> (snapshot + eval)
export.py         self-contained ONNX export + safetensors I/O
promote.py        snapshot .safetensors -> browser ONNX
elo.py            legacy one-shot Elo fit (superseded by evaluate_rs's global refit)
runs/             (gitignored) weights, snapshots/, serving_weights.safetensors, pool.json
```

## Setup (asus-nvidia / GB10)

```sh
cd training
~/.local/bin/uv venv .venv --python 3.12
~/.local/bin/uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/nightly/cu128 torch
~/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt

# Build the Rust binaries (tch links the venv's libtorch).
source .venv/bin/activate
export SELFPLAY_PYTHON=$(which python)
(cd selfplay_rs && cargo build --release)   # builds selfplay_rs + evaluate_rs
```

`.cargo/config.toml` sets `LIBTORCH_USE_PYTORCH=1` and
`LIBTORCH_BYPASS_VERSION_CHECK=1`; activate the venv before building so `python3`
is the project interpreter.

## Train

Managed by pm2 (`fox-train`). Env vars drive `run_daemon.sh`; warm-start a fresh
run from the best prior snapshot with `INIT_FROM` (a `.pt` or raw `.safetensors`,
ignored once `latest.pt` exists):

```sh
OUT_DIR=$PWD/runs/run3 \
INIT_FROM=$PWD/runs/run1/snapshots/snap_h00019_20260608T160917Z.safetensors \
SIMS=400 \
  pm2 start training/run_daemon.sh --name fox-train --kill-timeout 20000
```

Resumes from `runs/run3/latest.pt`. The trainer streams finished games from the
Rust worker into a ring replay buffer and runs SGD paced to the data rate
(`--target-reuse`). Watch progress:

```sh
pm2 logs fox-train             # JSON lines: games/buf/tr_steps/loss{policy,value}
cat  runs/run3/pool.json       # models + ratings + match results + top list
cat  runs/run3/eval.log        # detached evaluate_rs [eval] lines
```

Quick self-play throughput probe (no trainer):

```sh
./selfplay_rs/target/release/selfplay_rs bench \
  --weights runs/run3/serving_weights.safetensors --sims 400 --threads 16 --batch 512
```

Each snapshot is evaluated by `evaluate_rs` (now via ISMCTS, `--sims`): it picks
an active opponent set (top-`--n-top` rated + `random` + `--n-anchors` frozen
snapshots spread across the Elo range), plays `--eval-games` per opponent, then
refits a global Bradley-Terry Elo over all accumulated results (random pinned at
0) into `pool.json`. Promotion to the browser model stays manual (see below).

## Promote a model to the browser (manual)

```sh
python promote.py --snapshot runs/run3/snapshots/snap_XXXX.safetensors --out /tmp/current.onnx
# then, on the web-app host:
scp asus-nvidia:/tmp/current.onnx public/models/current.onnx   # and commit
```

## Tests / parity gates

```sh
# Generate the (gitignored) parity fixtures first:
node scripts/dump_rules_traces.mjs 300 > foxlite_core/tests/rules_traces.json
node scripts/dump_encode_fixtures.mjs 60 > fixtures/encode_fixtures.json

(cd foxlite_core && cargo test)                 # rules + encoder parity vs game.js
python test_encode_parity.py                    # Python encoder vs JS reference
python make_forward_fixture.py && python test_onnx_parity.py   # ONNX vs PyTorch
./selfplay_rs/target/release/selfplay_rs forward-check ./fixtures  # Rust tch vs PyTorch
node ../training/scripts/test_browser_engine_e2e.mjs   # real ONNX + JS engine, legal moves
```

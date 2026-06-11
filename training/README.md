# Fox-Lite self-play training

Search-free **REINFORCE** self-play for the Fox-in-the-Forest-Lite engine. 100%
Rust self-play + evaluator (`tch`/libtorch on GPU), Python trainer, strict-
synchronous loop, 30-minute snapshots. The deployed browser engine
(`src/engine/neural.js`) runs the exported ONNX with a single forward pass — no
search.

## Layout

```
foxlite_core/     Rust crate: Lite rules (port of src/engine/game.js) + canonical encoder
selfplay_rs/      Rust: self-play worker (selfplay_rs) + evaluator (evaluate_rs), tch forward
net.py            PyTorch residual MLP (policy + value)
encode.py         canonical encoder (parity reference)
train.py          REINFORCE update (advantage = z - V; + value MSE + entropy bonus)
orchestrator.py   strict-sync loop: selfplay -> train -> publish -> (snapshot + eval)
export.py         self-contained ONNX export + safetensors I/O
promote.py        snapshot .safetensors -> browser ONNX
cohort.py         read/verify a self-play cohort file
elo.py            legacy one-shot Elo fit (superseded by evaluate_rs's global refit)
runs/             (gitignored) weights, snapshots/, cohort, pool.json
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

```sh
OUT_DIR=runs/run1 nohup ./run_daemon.sh > runs/run1.log 2>&1 &
```

Resumes from `runs/run1/latest.pt`. Watch progress + the Elo curve:

```sh
tail -f runs/run1.log          # per-cohort loss/entropy/value + [eval] elo lines
cat  runs/run1/pool.json       # models + ratings + match results + top list
```

Each snapshot is evaluated by `evaluate_rs`: it picks an active opponent set
(top-`--n-top` rated + `random` + `--n-anchors` frozen snapshots spread across
the Elo range), plays `--eval-games` per opponent, then refits a global
Bradley-Terry Elo over all accumulated results (random pinned at 0) into
`pool.json`. Promotion to the browser model stays manual (see below).

## Promote a model to the browser (manual)

```sh
python promote.py --snapshot runs/run1/snapshots/snap_XXXX.safetensors --out /tmp/current.onnx
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

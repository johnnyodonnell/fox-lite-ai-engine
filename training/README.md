# Training — neural Fox in the Forest Lite bot

This folder trains the project's neural bot via **self-play reinforcement
learning** — no human games, no hand-coded heuristic teacher. The bot is an
AlphaZero-style policy/value network guided by **PIMC** (Perfect Information
Monte Carlo) search: at every decision, K determinizations of the unseen cards
are sampled, a PUCT tree is run in each, and the search results are averaged.

The web app (`../src/`) ships only the trained weights and a hand-written JS
forward pass + JS PIMC — no Python and no ML runtime in the browser.

## Setup (run on `asus-nvidia`)

```sh
ssh asus-nvidia
cd ~/Code/fox-lite-ai-engine/training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls PyTorch from the nightly **cu130** wheel index because
GB10 (Blackwell, compute capability `sm_121`) needs CUDA 13. Stable PyTorch
wheels for aarch64 + CUDA 13 don't exist yet.

Verify the GPU is visible:

```sh
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
# -> NVIDIA GB10
```

## Workflow

*Filled in as each phase lands.* See the project's plan at
`~/.claude/plans/swirling-whistling-river.md` for the phase roadmap.

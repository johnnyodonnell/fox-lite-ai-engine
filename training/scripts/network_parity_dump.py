"""Dump Python-side forward-pass outputs for the network parity check.

For a fixed batch of encoded inputs (re-used from the encode parity corpus),
runs `best.pt` in float64 and writes (input, expected_logits, expected_value)
tuples. `network_parity_check.mjs` reads this plus src/engine/weights.json,
runs the JS forward pass on the same inputs, and asserts agreement.

Both sides do float64 arithmetic, so any disagreement above ~1e-12 indicates
a real implementation bug rather than precision drift.

Run (after exporting weights):
    python training/scripts/export_weights.py
    python training/scripts/network_parity_dump.py
    node   training/scripts/network_parity_check.mjs
"""

from __future__ import annotations

import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, TRAINING_DIR)

import torch  # noqa: E402

from alphazero.network import PolicyValueNet, infer  # noqa: E402

CORPUS_PATH = os.path.join(TRAINING_DIR, "parity_expected.json")
OUT_PATH = os.path.join(TRAINING_DIR, "network_parity_expected.json")
CHECKPOINT = os.path.join(TRAINING_DIR, "checkpoints", "best.pt")


def main() -> int:
    if not os.path.exists(CORPUS_PATH):
        print(f"Missing corpus: {CORPUS_PATH}. Run parity_corpus.mjs first.")
        return 2
    if not os.path.exists(CHECKPOINT):
        print(f"Missing checkpoint: {CHECKPOINT}. Train first.")
        return 2

    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    inputs = [c["expected"] for c in corpus["encode"]]

    net = PolicyValueNet()
    net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    net.double()
    net.eval()

    cases = []
    for x in inputs:
        logits, value = infer(net, x)
        cases.append({"input": x, "policyLogits": logits, "value": value})

    payload = {
        "meta": {"checkpoint": CHECKPOINT, "inputSize": len(inputs[0]) if inputs else 0},
        "cases": cases,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f)
    print(f"wrote {len(cases)} network parity cases -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

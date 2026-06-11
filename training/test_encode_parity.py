"""Parity test: the Python encoder must reproduce the reference JS encoder
(src/engine/encode.js) bit-for-bit at every recorded decision point.

Regenerate the fixture:
  node training/scripts/dump_encode_fixtures.mjs 60 > training/fixtures/encode_fixtures.json

Run:
  python training/test_encode_parity.py
"""

import json
import sys
from pathlib import Path

import numpy as np

from encode import INPUT_SIZE, encode, legal_mask

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "encode_fixtures.json"


def set_indices(v: np.ndarray) -> list:
    return [int(i) for i in np.nonzero(v)[0]]


def sparse(v: np.ndarray) -> list:
    return [[int(i), float(v[i])] for i in np.nonzero(v)[0]]


def main() -> int:
    if not FIXTURE.exists():
        print(f"missing fixture {FIXTURE}\n"
              "generate: node training/scripts/dump_encode_fixtures.mjs 60 "
              "> training/fixtures/encode_fixtures.json")
        return 1
    data = json.loads(FIXTURE.read_text())
    assert data["inputSize"] == INPUT_SIZE, (data["inputSize"], INPUT_SIZE)
    cases = data["cases"]
    assert cases, "no cases"

    for i, case in enumerate(cases):
        state, mover = case["state"], case["mover"]
        enc = sparse(encode(state, mover))
        assert enc == case["enc"], f"case {i}: encoding mismatch"
        mask = set_indices(legal_mask(state, mover))
        assert mask == case["mask"], f"case {i}: legal mask mismatch"

    print(f"encode parity OK over {len(cases)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())

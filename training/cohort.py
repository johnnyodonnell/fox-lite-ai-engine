"""Read a self-play cohort file written by selfplay_rs.

Row layout (ROW_FLOATS = INPUT_SIZE + NUM_CARDS + 2):
  [ state(INPUT_SIZE) | legal_mask(NUM_CARDS) | action_index | z ]
File: u32 num_rows (LE), u32 row_floats (LE), then num_rows*row_floats f32 LE.

Run as a script to sanity-check a cohort:
  python training/cohort.py /path/to/cohort.bin
"""

import struct
import sys

import numpy as np

from encode import HIST, HIST_TOKENS, INPUT_SIZE, NUM_CARDS, TOKEN_FEATS

ROW_FLOATS = INPUT_SIZE + NUM_CARDS + 2


def read_cohort(path: str) -> dict:
    with open(path, "rb") as f:
        n_rows = struct.unpack("<I", f.read(4))[0]
        row_floats = struct.unpack("<I", f.read(4))[0]
        if row_floats != ROW_FLOATS:
            raise ValueError(f"row_floats {row_floats} != {ROW_FLOATS}")
        data = np.frombuffer(f.read(), dtype="<f4")
    if data.size != n_rows * row_floats:
        raise ValueError(f"truncated cohort: {data.size} != {n_rows * row_floats}")
    data = data.reshape(n_rows, row_floats)
    return {
        "states": np.ascontiguousarray(data[:, :INPUT_SIZE]),
        "masks": np.ascontiguousarray(data[:, INPUT_SIZE:INPUT_SIZE + NUM_CARDS]),
        "actions": np.ascontiguousarray(data[:, INPUT_SIZE + NUM_CARDS]).astype(np.int64),
        "z": np.ascontiguousarray(data[:, -1]),
        "n": n_rows,
    }


def main() -> int:
    c = read_cohort(sys.argv[1])
    a, m, z = c["actions"], c["masks"], c["z"]
    chosen_mask = m[np.arange(c["n"]), a]
    legal_ok = bool(np.all(chosen_mask == 1.0))
    z_ok = bool(np.all(np.isin(z, [-1.0, 1.0])))
    # history tokens: [card 0..32, self 0/1, valid 0/1] per slot, valid bits a
    # prefix (events fill slots in play order, padding only at the tail)
    tok = c["states"][:, :HIST].reshape(-1, HIST_TOKENS, TOKEN_FEATS)
    card, self_bit, valid = tok[:, :, 0], tok[:, :, 1], tok[:, :, 2]
    tok_ok = bool(
        np.all((card >= 0) & (card < NUM_CARDS) & (card == np.floor(card)))
        and np.all(np.isin(self_bit, [0.0, 1.0]))
        and np.all(np.isin(valid, [0.0, 1.0]))
        and np.all(np.diff(valid, axis=1) <= 0)  # prefix-monotone
        and np.all(card * (1.0 - valid) == 0.0)  # padded slots all-zero
        and np.all(self_bit * (1.0 - valid) == 0.0)
    )
    # each input row should have a fixed number of one-hot blocks set; spot-check
    # that own-hand counts are in 1..13 (mover always holds >=1 card on its turn)
    own_hand_counts = c["states"][:, HIST:HIST + NUM_CARDS].sum(axis=1)
    print(f"rows={c['n']}")
    print(f"  chosen-action-legal: {legal_ok}")
    print(f"  z in {{-1,1}}: {z_ok}   z.mean={float(z.mean()):+.3f}")
    print(f"  history tokens well-formed: {tok_ok}")
    print(f"  valid-token count range: [{int(valid.sum(axis=1).min())},{int(valid.sum(axis=1).max())}]")
    print(f"  action idx range: [{int(a.min())},{int(a.max())}]")
    print(f"  own-hand card count range: [{int(own_hand_counts.min())},{int(own_hand_counts.max())}]")
    ok = legal_ok and z_ok and tok_ok
    print("COHORT OK" if ok else "COHORT BAD")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

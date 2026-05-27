"""Verify that encode() is truly mover-frame canonical.

If the encoding is canonical, then for any state S:
    encode(S, BOT) == encode(swap_seats(S), HUMAN)
i.e. the network input is identical whether you encode the original state
from the bot's view or the seat-swapped state from the human's view. Both
describe "what the mover sees."

If this test FAILS, there's a remaining encoding asymmetry. If it PASSES
but the trained net still gives asymmetric outputs, the issue is the
network learning, not the encoding.
"""

from __future__ import annotations

import copy
import os
import random
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, TRAINING_DIR)

from games.foxlite import (  # noqa: E402
    BOT,
    HUMAN,
    advance_after_trick,
    deal_round,
    encode,
    legal_moves,
    play_card,
)


def swap_seats(state: dict) -> dict:
    """Flip every seat-labeled field: hands, tricks, scores, leader,
    awaiting, lastTrick winner/leader, and the per-event player in
    trickHistory. The state's *positions* are unchanged — only which seat
    label they wear."""
    def other(p):
        if p == HUMAN: return BOT
        if p == BOT: return HUMAN
        return p

    new = dict(state)
    new["humanHand"] = state["botHand"]
    new["botHand"] = state["humanHand"]
    new["leader"] = other(state["leader"])
    new["awaiting"] = other(state["awaiting"]) if state["awaiting"] is not None else None
    new["tricksWon"] = {"human": state["tricksWon"]["bot"], "bot": state["tricksWon"]["human"]}
    new["score"] = {"human": state["score"]["bot"], "bot": state["score"]["human"]}
    if state.get("lastTrick"):
        new["lastTrick"] = {
            **state["lastTrick"],
            "leader": other(state["lastTrick"]["leader"]),
            "winner": other(state["lastTrick"]["winner"]),
        }
    new["trickHistory"] = [
        {"trick": e["trick"], "player": other(e["player"]), "card": e["card"]}
        for e in state["trickHistory"]
    ]
    return new


def main() -> None:
    rng = random.Random(123)
    failures = 0
    max_diff_seen = 0.0

    for trial in range(50):
        # Build a random state by playing some moves forward.
        state = deal_round(1, {"human": 0, "bot": 0}, rng=rng)
        for _ in range(rng.randint(0, 14)):
            if state["phase"] != "playing":
                break
            mover = state["awaiting"]
            hand_key = "humanHand" if mover == HUMAN else "botHand"
            legal = legal_moves(state[hand_key], state["ledCard"])
            state = play_card(state, rng.choice(legal))
            if state["phase"] == "trick-complete":
                state = advance_after_trick(state)
        if state["phase"] != "playing":
            continue

        swapped = swap_seats(state)
        # Original mover encoded normally; the swapped state encoded from
        # the corresponding "other" mover's perspective.
        for mover in [BOT, HUMAN]:
            other_mover = HUMAN if mover == BOT else BOT
            enc_a = encode(state, mover)
            enc_b = encode(swapped, other_mover)
            diffs = [a - b for a, b in zip(enc_a, enc_b)]
            max_diff = max(abs(d) for d in diffs)
            if max_diff > 1e-12:
                failures += 1
                max_diff_seen = max(max_diff_seen, max_diff)
                if failures <= 3:
                    diff_indices = [i for i, d in enumerate(diffs) if abs(d) > 1e-12]
                    print(f"FAIL  trial {trial}  mover={mover}  trickNum={state['trickNum']}")
                    print(f"      max diff = {max_diff}")
                    print(f"      differing indices ({len(diff_indices)}): {diff_indices[:10]}...")
                    print(f"      enc_a at these indices: {[enc_a[i] for i in diff_indices[:10]]}")
                    print(f"      enc_b at these indices: {[enc_b[i] for i in diff_indices[:10]]}")

    print()
    if failures == 0:
        print("PASS — encode() is mover-frame canonical (seat-swap symmetric)")
    else:
        print(f"FAIL — {failures} cases mismatched, worst max-diff = {max_diff_seen}")


if __name__ == "__main__":
    main()

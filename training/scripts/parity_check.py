"""Parity check (Python side): replay the corpus dumped by parity_corpus.mjs
through training/games/foxlite.py and assert byte-identical state transitions.

JS is the source of truth — the rules already ship in production. This script
catches any drift in the Python port.

Run:
    node training/scripts/parity_corpus.mjs
    python training/scripts/parity_check.py
"""

from __future__ import annotations

import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, TRAINING_DIR)

from games.foxlite import (  # noqa: E402
    INPUT_SIZE,
    advance_after_trick,
    encode,
    end_round,
    legal_moves,
    play_card,
    score_for_tricks,
    trick_winner,
)

CORPUS_PATH = os.path.join(TRAINING_DIR, "parity_expected.json")


def _diff_summary(label: str, expected, actual) -> str:
    """Tiny diff for human-readable mismatch reports."""
    e = json.dumps(expected, sort_keys=True)
    a = json.dumps(actual, sort_keys=True)
    if len(e) > 200:
        e = e[:200] + "..."
    if len(a) > 200:
        a = a[:200] + "..."
    return f"  {label} expected: {e}\n  {label} actual  : {a}"


def check_score_for_tricks(cases) -> int:
    failures = 0
    for c in cases:
        got = score_for_tricks(c["n"])
        if got != c["expected"]:
            print(f"FAIL scoreForTricks(n={c['n']})  expected={c['expected']}  got={got}")
            failures += 1
    return failures


def check_trick_winner(cases) -> int:
    failures = 0
    for c in cases:
        got = trick_winner(c["led"], c["follow"], c["trumpSuit"])
        if got != c["expected"]:
            print(
                f"FAIL trickWinner(led={c['led']['id']}, follow={c['follow']['id']}, "
                f"trumpSuit={c['trumpSuit']})  expected={c['expected']}  got={got}"
            )
            failures += 1
    return failures


def check_legal_moves(cases) -> int:
    failures = 0
    for c in cases:
        got = legal_moves(c["hand"], c["ledCard"])
        if got != c["expected"]:
            print(
                f"FAIL legalMoves(hand size {len(c['hand'])}, "
                f"led={c['ledCard'] and c['ledCard']['id']})"
            )
            print(_diff_summary("legal", c["expected"], got))
            failures += 1
    return failures


def _initial_state(deal) -> dict:
    """Reconstruct a fresh round-1 state from a JS-side initial deal."""
    return {
        "humanHand": deal["humanHand"],
        "botHand": deal["botHand"],
        "trump": deal["trump"],
        "leader": deal["leader"],
        "ledCard": None,
        "awaiting": deal["leader"],
        "tricksWon": {"human": 0, "bot": 0},
        "score": {"human": 0, "bot": 0},
        "roundNum": 1,
        "trickNum": 1,
        "phase": "playing",
        "lastTrick": None,
        "trickHistory": [],
    }


def _state_fields(state: dict) -> dict:
    """Project the state to the fields the JS snapshot captured."""
    return {
        "humanHand": state["humanHand"],
        "botHand": state["botHand"],
        "trump": state["trump"],
        "leader": state["leader"],
        "ledCard": state["ledCard"],
        "awaiting": state["awaiting"],
        "tricksWon": state["tricksWon"],
        "score": state["score"],
        "roundNum": state["roundNum"],
        "trickNum": state["trickNum"],
        "phase": state["phase"],
        "lastTrick": state["lastTrick"],
        "trickHistory": state["trickHistory"],
    }


def check_play_card(games) -> int:
    """Replay each game and compare state after every event to the JS snapshot.

    Round transitions (endRound) reshuffle the deck — Python's RNG can't match
    the JS shuffle, so we *adopt* the next-round hands/trump from the JS
    snapshot when we hit one. Everything else is verified deterministically.
    """
    failures = 0
    for g_idx, game in enumerate(games):
        state = _initial_state(game["initialDeal"])
        for ev_idx, ev in enumerate(game["trace"]):
            kind = ev["kind"]
            if kind == "play":
                state = play_card(state, ev["card"])
            elif kind == "advance":
                state = advance_after_trick(state)
            elif kind == "endRound":
                state = end_round(state)
                # Replace the freshly-shuffled round with JS's actual deal —
                # we are not parity-testing the shuffle.
                if state["phase"] == "playing":
                    after = ev["after"]
                    state["humanHand"] = after["humanHand"]
                    state["botHand"] = after["botHand"]
                    state["trump"] = after["trump"]
                    state["leader"] = after["leader"]
                    state["awaiting"] = after["awaiting"]
            else:
                print(f"FAIL game[{g_idx}] event[{ev_idx}] unknown kind={kind}")
                failures += 1
                break

            got = _state_fields(state)
            if got != ev["after"]:
                print(f"FAIL game[{g_idx}] event[{ev_idx}] kind={kind}")
                # Spot the first mismatching field for a friendlier message.
                for k in got:
                    if got[k] != ev["after"][k]:
                        print(_diff_summary(k, ev["after"][k], got[k]))
                        break
                failures += 1
                break
    return failures


def check_encode(cases) -> int:
    """Verify Python's encoder produces byte-identical vectors to JS's."""
    failures = 0
    if cases and len(cases[0]["expected"]) != INPUT_SIZE:
        print(
            f"FAIL encode: corpus reports inputSize={len(cases[0]['expected'])} "
            f"but Python's INPUT_SIZE={INPUT_SIZE}"
        )
        failures += 1
    for i, c in enumerate(cases):
        got = encode(c["state"], c["mover"])
        if got != c["expected"]:
            max_diff = max(abs(a - b) for a, b in zip(got, c["expected"]))
            print(
                f"FAIL encode[{i}]  mover={c['mover']}  trickNum={c['state']['trickNum']}  "
                f"max abs diff={max_diff}"
            )
            # Show first few mismatched indices.
            for j, (a, b) in enumerate(zip(got, c["expected"])):
                if a != b:
                    print(f"    idx {j}: got {a}, expected {b}")
                    break
            failures += 1
    return failures


def main() -> int:
    if not os.path.exists(CORPUS_PATH):
        print(
            f"Corpus not found at {CORPUS_PATH}\n"
            f"Run:  node training/scripts/parity_corpus.mjs"
        )
        return 2
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    sections = [
        ("scoreForTricks", check_score_for_tricks, corpus["scoreForTricks"]),
        ("trickWinner   ", check_trick_winner, corpus["trickWinner"]),
        ("legalMoves    ", check_legal_moves, corpus["legalMoves"]),
        ("playCard      ", check_play_card, corpus["playCard"]),
        ("encode        ", check_encode, corpus.get("encode", [])),
    ]
    total_failures = 0
    for label, fn, cases in sections:
        failures = fn(cases)
        n = len(cases)
        status = "OK  " if failures == 0 else "FAIL"
        print(f"{status}  {label}  {n - failures}/{n} cases pass")
        total_failures += failures

    if total_failures == 0:
        print("\nPARITY OK — Python rules match JS reference.")
        return 0
    print(f"\nPARITY FAILED — {total_failures} case(s) mismatched.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

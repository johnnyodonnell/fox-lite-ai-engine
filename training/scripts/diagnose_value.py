"""Sanity-check the value head on hand-constructed states.

We want to answer: is the trained value head producing sensible outputs,
or are they random / saturated / sign-flipped? Cheap test before deciding
whether the problem is a bug or training-data noise.

Run on asus-nvidia after training:
    python training/scripts/diagnose_value.py
"""

from __future__ import annotations

import copy
import os
import random
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, TRAINING_DIR)

import torch  # noqa: E402

from alphazero.network import PolicyValueNet, infer  # noqa: E402
from games.foxlite import (  # noqa: E402
    BOT,
    CARD_INDEX,
    HUMAN,
    NUM_CARDS,
    SUITS,
    create_deck,
    deal_round,
    encode,
    legal_moves,
)

CHECKPOINT = os.path.join(TRAINING_DIR, "checkpoints", "best.pt")


def load_net() -> PolicyValueNet:
    net = PolicyValueNet()
    net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    net.double()
    net.eval()
    return net


def predict(net: PolicyValueNet, state: dict, mover: str) -> tuple[list[float], float]:
    return infer(net, encode(state, mover))


# ---------------------------------------------------------------------------
# Test 1: distribution of values across many real states from a self-played
# game. We expect them to span a range, with std > 0 (not collapsed to ~0).
# ---------------------------------------------------------------------------
def test_distribution(net: PolicyValueNet) -> None:
    print("=" * 70)
    print("Test 1 — Value head output distribution across 200 random states")
    print("=" * 70)
    rng = random.Random(0)
    values: list[float] = []
    for _ in range(200):
        state = deal_round(1, {"human": 0, "bot": 0}, rng=rng)
        # Step a random number of plies forward.
        for _ in range(rng.randint(0, 12) * 2):
            if state["phase"] != "playing":
                break
            mover = state["awaiting"]
            hand_key = "humanHand" if mover == HUMAN else "botHand"
            legal = legal_moves(state[hand_key], state["ledCard"])
            from games.foxlite import play_card, advance_after_trick
            state = play_card(state, rng.choice(legal))
            if state["phase"] == "trick-complete":
                state = advance_after_trick(state)
        if state["phase"] != "playing":
            continue
        _, value = predict(net, state, state["awaiting"])
        values.append(value)

    print(f"  n            = {len(values)}")
    print(f"  min, max     = {min(values):+.3f},  {max(values):+.3f}")
    print(f"  mean         = {sum(values) / len(values):+.3f}")
    var = sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)
    print(f"  std          = {var ** 0.5:.3f}")
    # Histogram
    bins = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]
    counts = [0] * (len(bins) - 1)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
    print("  histogram:")
    for i in range(len(bins) - 1):
        bar = "#" * counts[i]
        print(f"    [{bins[i]:+.2f}, {bins[i+1]:+.2f})  {counts[i]:3d}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Test 2: monotonicity — same hand, same trump, different tricks-won totals.
# Higher mover_tricks should not hurt the value (lite scoring is non-monotonic,
# but ascending tricks within 4-9 should give ascending value).
# ---------------------------------------------------------------------------
def test_tricks_won_monotonicity(net: PolicyValueNet) -> None:
    print("=" * 70)
    print("Test 2 — Value vs mover's tricks-won (mid-round, identical hand)")
    print("=" * 70)
    print("Lite scoring: 0-3 -> 6,  4 -> 1,  5 -> 2,  6 -> 3,  7-9 -> 6, 10-13 -> 0")
    print("So value should ROUGHLY: dip at 4-6, peak around 0-3 and 7-9, drop at 10+")
    print()
    rng = random.Random(1)
    state = deal_round(1, {"human": 0, "bot": 0}, rng=rng)

    print("  bot_tricks  opp_tricks  trickNum   net value (bot frame)")
    print("  ----------  ----------  --------   ---------------------")
    for bt in range(0, 13):
        for ht in range(0, 13):
            if bt + ht > 12:
                continue
            s = copy.deepcopy(state)
            s["tricksWon"]["bot"] = bt
            s["tricksWon"]["human"] = ht
            s["trickNum"] = bt + ht + 1
            if s["trickNum"] > 13:
                continue
            _, value = predict(net, s, BOT)
            if ht == 6 - bt or ht == bt:  # show a tidy slice
                print(f"  {bt:^10d}  {ht:^10d}  {s['trickNum']:^8d}   {value:+.3f}")
    print()


# ---------------------------------------------------------------------------
# Test 3: sign symmetry — value(state, mover=BOT) should equal
#                       -value(state_with_seats_swapped, mover=BOT)
# i.e. the net should give opposite values when "self" and "opponent" swap.
# ---------------------------------------------------------------------------
def test_sign_symmetry(net: PolicyValueNet) -> None:
    print("=" * 70)
    print("Test 3 — Mover-frame symmetry: predict(state, BOT) vs predict(state, HUMAN)")
    print("=" * 70)
    print("These should be approximately negatives of each other.")
    print()
    rng = random.Random(2)
    diffs: list[float] = []
    for _ in range(30):
        state = deal_round(1, {"human": 0, "bot": 0}, rng=rng)
        # Step a few plies.
        from games.foxlite import play_card, advance_after_trick
        for _ in range(rng.randint(0, 6) * 2):
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
        _, v_bot = predict(net, state, BOT)
        _, v_hum = predict(net, state, HUMAN)
        diffs.append(v_bot + v_hum)  # should be ~0 if symmetric
    print(f"  n cases             = {len(diffs)}")
    if diffs:
        print(f"  mean (v_BOT + v_HUMAN) = {sum(diffs)/len(diffs):+.4f}  (~0 if symmetric)")
        print(f"  abs max             = {max(abs(d) for d in diffs):.4f}")
    print("  Note: not necessarily zero — different inputs through different forward")
    print("  passes — but the mean should be near zero if the net learned a balanced")
    print("  mover-frame representation.")
    print()


# ---------------------------------------------------------------------------
# Test 4: policy reasonableness — concentrate on a known-good move?
# ---------------------------------------------------------------------------
def test_policy_on_obvious_position(net: PolicyValueNet) -> None:
    print("=" * 70)
    print("Test 4 — Policy distribution on a few states")
    print("=" * 70)
    rng = random.Random(3)
    for trial in range(3):
        state = deal_round(1, {"human": 0, "bot": 0}, rng=rng)
        legal = legal_moves(state["botHand"], state["ledCard"])
        logits, value = predict(net, state, BOT)
        max_l = max(logits[CARD_INDEX[c["id"]]] for c in legal)
        exps = {c["id"]: __import__("math").exp(logits[CARD_INDEX[c["id"]]] - max_l) for c in legal}
        z = sum(exps.values())
        probs = {k: v / z for k, v in exps.items()}
        print(f"  Trial {trial+1}: trump={state['trump']['id']}, mover=BOT leads first trick")
        print(f"    Bot hand: {[c['id'] for c in state['botHand']]}")
        print(f"    Predicted value = {value:+.3f}")
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        for cid, p in ranked[:5]:
            print(f"      {cid:>10s}  p={p:.3f}")
        print()


# ---------------------------------------------------------------------------
# Test 5: weight-norm — are weights still finite and in a sane range?
# ---------------------------------------------------------------------------
def test_weight_norms(net: PolicyValueNet) -> None:
    print("=" * 70)
    print("Test 5 — Weight statistics")
    print("=" * 70)
    for name, p in net.named_parameters():
        print(f"  {name:>30s}  L2={p.norm().item():.3f}  "
              f"mean={p.mean().item():+.4f}  std={p.std().item():.4f}  "
              f"min={p.min().item():+.3f}  max={p.max().item():+.3f}")
    print()


def main() -> None:
    if not os.path.exists(CHECKPOINT):
        print(f"Missing checkpoint: {CHECKPOINT}")
        sys.exit(2)
    net = load_net()
    test_weight_norms(net)
    test_distribution(net)
    test_tricks_won_monotonicity(net)
    test_sign_symmetry(net)
    test_policy_on_obvious_position(net)


if __name__ == "__main__":
    main()

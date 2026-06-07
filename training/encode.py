"""Canonical NN input encoding (suit-canonicalized, mover-frame).

Byte-for-byte parity with src/engine/encode.js and foxlite_core/src/encode.rs.
This Python copy is the reference used to build forward-parity fixtures and to
sanity-check the Rust/JS encoders. It consumes a state dict shaped exactly like
the JS game state (humanHand/botHand lists of {suit,rank,id}, trump, ledCard,
tricksWon, score, trickNum, trickHistory, awaiting).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

SUITS = ("bells", "keys", "moons")
RANKS = tuple(range(1, 12))
HUMAN = "human"
BOT = "bot"
NUM_SUITS = len(SUITS)
NUM_RANKS = len(RANKS)
NUM_CARDS = NUM_SUITS * NUM_RANKS  # 33
TRICKS_PER_ROUND = 13
TARGET_SCORE = 21

# Block sizes (canonical layout).
_OWN_HAND = NUM_CARDS
_PLAYED_SELF = NUM_CARDS
_PLAYED_OPP = NUM_CARDS
_TRUMP_RANK = NUM_RANKS
_OPP_VOIDS = NUM_SUITS
_LED = NUM_CARDS + 1
_SELF_TRICKS = TRICKS_PER_ROUND + 1
_OPP_TRICKS = TRICKS_PER_ROUND + 1
_TRICK_NUM = TRICKS_PER_ROUND
_SCORE_SLOTS = TARGET_SCORE

INPUT_SIZE = (
    _OWN_HAND + _PLAYED_SELF + _PLAYED_OPP + _TRUMP_RANK + _OPP_VOIDS + _LED
    + _SELF_TRICKS + _OPP_TRICKS + _TRICK_NUM + _SCORE_SLOTS + _SCORE_SLOTS
)  # 230


def _suit_index(suit: str) -> int:
    return SUITS.index(suit)


def canon_suit(real_suit_idx: int, trump_idx: int) -> int:
    if real_suit_idx == trump_idx:
        return 0
    slot = 1
    for s in range(NUM_SUITS):
        if s != trump_idx and s < real_suit_idx:
            slot += 1
    return slot


def real_suit_from_canon(canon_slot: int, trump_idx: int) -> int:
    if canon_slot == 0:
        return trump_idx
    non_trump = [s for s in range(NUM_SUITS) if s != trump_idx]
    return non_trump[canon_slot - 1]


def canon_card_index(card: dict, trump_idx: int) -> int:
    return canon_suit(_suit_index(card["suit"]), trump_idx) * NUM_RANKS + (card["rank"] - 1)


def real_card_from_canon_index(ci: int, trump_idx: int) -> dict:
    canon_slot = ci // NUM_RANKS
    rank = ci % NUM_RANKS + 1
    suit = SUITS[real_suit_from_canon(canon_slot, trump_idx)]
    return {"suit": suit, "rank": rank, "id": f"{suit}-{rank}"}


def _opponent_voids(trick_history: list, opponent: str) -> set:
    voids: set = set()
    by_trick: dict = {}
    for ev in trick_history:
        by_trick.setdefault(ev["trick"], []).append(ev)
    for events in by_trick.values():
        if len(events) < 2:
            continue
        lead, follow = events[0], events[1]
        if follow["player"] == opponent and follow["card"]["suit"] != lead["card"]["suit"]:
            voids.add(_suit_index(lead["card"]["suit"]))
    return voids


def encode(state: dict, mover: Optional[str] = None) -> np.ndarray:
    if mover is None:
        mover = state["awaiting"]
    out = np.zeros(INPUT_SIZE, dtype=np.float32)
    trump_idx = _suit_index(state["trump"]["suit"])
    mover_is_human = mover == HUMAN
    opp = BOT if mover_is_human else HUMAN

    own_hand = state["humanHand"] if mover_is_human else state["botHand"]
    self_tricks = state["tricksWon"]["human" if mover_is_human else "bot"]
    opp_tricks = state["tricksWon"]["bot" if mover_is_human else "human"]
    self_score = state["score"]["human" if mover_is_human else "bot"]
    opp_score = state["score"]["bot" if mover_is_human else "human"]

    cur = 0
    for c in own_hand:
        out[cur + canon_card_index(c, trump_idx)] = 1.0
    cur += _OWN_HAND
    played_self_base = cur
    played_opp_base = cur + _PLAYED_SELF
    for ev in state["trickHistory"]:
        base = played_self_base if ev["player"] == mover else played_opp_base
        out[base + canon_card_index(ev["card"], trump_idx)] = 1.0
    cur += _PLAYED_SELF + _PLAYED_OPP
    out[cur + (state["trump"]["rank"] - 1)] = 1.0
    cur += _TRUMP_RANK
    for real_suit in _opponent_voids(state["trickHistory"], opp):
        out[cur + canon_suit(real_suit, trump_idx)] = 1.0
    cur += _OPP_VOIDS
    if state["ledCard"] is not None:
        out[cur + canon_card_index(state["ledCard"], trump_idx)] = 1.0
    else:
        out[cur + NUM_CARDS] = 1.0
    cur += _LED
    out[cur + min(self_tricks, TRICKS_PER_ROUND)] = 1.0
    cur += _SELF_TRICKS
    out[cur + min(opp_tricks, TRICKS_PER_ROUND)] = 1.0
    cur += _OPP_TRICKS
    out[cur + (state["trickNum"] - 1)] = 1.0
    cur += _TRICK_NUM
    out[cur + min(self_score, _SCORE_SLOTS - 1)] = 1.0
    cur += _SCORE_SLOTS
    out[cur + min(opp_score, _SCORE_SLOTS - 1)] = 1.0
    cur += _SCORE_SLOTS

    if cur != INPUT_SIZE:
        raise RuntimeError(f"encode cursor {cur} != {INPUT_SIZE}")
    return out


def legal_mask(state: dict, mover: Optional[str] = None) -> np.ndarray:
    if mover is None:
        mover = state["awaiting"]
    out = np.zeros(NUM_CARDS, dtype=np.float32)
    trump_idx = _suit_index(state["trump"]["suit"])
    hand = state["humanHand"] if mover == HUMAN else state["botHand"]
    led = state["ledCard"]
    legal = hand
    if led is not None:
        same = [c for c in hand if c["suit"] == led["suit"]]
        if same:
            legal = same
    for c in legal:
        out[canon_card_index(c, trump_idx)] = 1.0
    return out

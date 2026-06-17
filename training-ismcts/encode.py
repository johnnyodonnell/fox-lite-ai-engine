"""Canonical NN input encoding (suit-canonicalized, mover-frame).

Byte-for-byte parity with src/engine/encode.js and foxlite_core/src/encode.rs.
This Python copy is the reference used to build forward-parity fixtures and to
sanity-check the Rust/JS encoders. It consumes a state dict shaped exactly like
the JS game state (humanHand/botHand lists of {suit,rank,id}, trump, ledCard,
tricksWon, score, trickNum, trickHistory, awaiting).

v3 layout: [ history tokens | static one-hot blocks ]. One token per COMPLETED
trick of the current round, in DESCENDING order (slot 0 = most recent completed
trick); the in-progress trick's lead is NOT a token — it lives in the static
led-card block. Each token is [lead card index, follow card index, led-by-self,
valid]. Padded slots are all-zero; the valid bit disambiguates padding from a
real card index 0 and is the net's attention/readout mask.
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
HIST_TOKENS = TRICKS_PER_ROUND - 1  # 12 (max completed tricks at a decision)
TOKEN_FEATS = 4  # [lead card 0..32, follow card 0..32, led-by-self 0/1, valid 0/1]
HIST = HIST_TOKENS * TOKEN_FEATS  # 48
_OWN_HAND = NUM_CARDS
_TRUMP_RANK = NUM_RANKS
_LED = NUM_CARDS + 1  # led card one-hot + "no led / I'm leading" slot
_SELF_TRICKS = TRICKS_PER_ROUND + 1
_OPP_TRICKS = TRICKS_PER_ROUND + 1
_TRICK_NUM = TRICKS_PER_ROUND
_SCORE_SLOTS = TARGET_SCORE

INPUT_SIZE = (
    HIST + _OWN_HAND + _TRUMP_RANK + _LED
    + _SELF_TRICKS + _OPP_TRICKS + _TRICK_NUM + _SCORE_SLOTS + _SCORE_SLOTS
)  # 209


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


def encode(state: dict, mover: Optional[str] = None) -> np.ndarray:
    if mover is None:
        mover = state["awaiting"]
    out = np.zeros(INPUT_SIZE, dtype=np.float32)
    trump_idx = _suit_index(state["trump"]["suit"])
    mover_is_human = mover == HUMAN

    own_hand = state["humanHand"] if mover_is_human else state["botHand"]
    self_tricks = state["tricksWon"]["human" if mover_is_human else "bot"]
    opp_tricks = state["tricksWon"]["bot" if mover_is_human else "human"]
    self_score = state["score"]["human" if mover_is_human else "bot"]
    opp_score = state["score"]["bot" if mover_is_human else "human"]

    cur = 0
    # history tokens: completed tricks, most recent first. Events arrive in play
    # order as (lead, follow) pairs; a trailing single event is the in-progress
    # trick's lead and is skipped here (it equals state["ledCard"]).
    events = state["trickHistory"]
    n_complete = len(events) // 2
    if n_complete > HIST_TOKENS:
        raise RuntimeError(f"completed tricks {n_complete} > {HIST_TOKENS}")
    for t in range(n_complete):
        lead = events[2 * (n_complete - 1 - t)]
        follow = events[2 * (n_complete - 1 - t) + 1]
        base = cur + t * TOKEN_FEATS
        out[base] = canon_card_index(lead["card"], trump_idx)
        out[base + 1] = canon_card_index(follow["card"], trump_idx)
        out[base + 2] = 1.0 if lead["player"] == mover else 0.0
        out[base + 3] = 1.0
    cur += HIST
    for c in own_hand:
        out[cur + canon_card_index(c, trump_idx)] = 1.0
    cur += _OWN_HAND
    out[cur + (state["trump"]["rank"] - 1)] = 1.0
    cur += _TRUMP_RANK
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

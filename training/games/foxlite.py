"""Fox in the Forest **Lite** rules — line-for-line port of src/engine/game.js.

"Lite" = the standard rules with every odd-rank special ability removed.
Cards are plain trick-takers; trump still applies. Two players, 33-card deck
(3 suits x 11 ranks), 13 cards per hand, 1 trump revealed, 6 unused.

Data shapes match the JS engine exactly so traces can be replayed across
languages for parity testing:
    card  : {"suit": "bells"|"keys"|"moons", "rank": 1..11, "id": "bells-7"}
    hand  : list[card]
    state : dict — same keys as the JS state object

This module is pure: no I/O, no module-level RNG. `create_game` and `_shuffle`
take an explicit `rng` so games are reproducible for tests.
"""

from __future__ import annotations

import random as _random_module
from typing import Optional

SUITS: tuple[str, ...] = ("bells", "keys", "moons")
RANKS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
HUMAN = "human"
BOT = "bot"
TARGET_SCORE = 21
TRICKS_PER_ROUND = 13

_SUIT_ORDER = {s: i for i, s in enumerate(SUITS)}


def card_id(suit: str, rank: int) -> str:
    return f"{suit}-{rank}"


def create_deck() -> list[dict]:
    return [
        {"suit": s, "rank": r, "id": card_id(s, r)}
        for s in SUITS
        for r in RANKS
    ]


def _shuffle(deck: list[dict], rng: _random_module.Random) -> list[dict]:
    """Fisher-Yates shuffle, mirroring src/engine/game.js's shuffle.

    The RNG sequence is NOT expected to match Math.random across languages —
    parity tests replay deals captured from the JS side rather than relying
    on shared seeds.
    """
    a = list(deck)
    for i in range(len(a) - 1, 0, -1):
        j = rng.randrange(0, i + 1)
        a[i], a[j] = a[j], a[i]
    return a


def sort_hand(hand: list[dict]) -> list[dict]:
    return sorted(hand, key=lambda c: (_SUIT_ORDER[c["suit"]], c["rank"]))


def initial_leader_for(round_num: int) -> str:
    """Round 1 = human leads; rounds alternate thereafter."""
    return HUMAN if round_num % 2 == 1 else BOT


def deal_round(
    round_num: int,
    score: dict,
    rng: Optional[_random_module.Random] = None,
) -> dict:
    rng = rng or _random_module.Random()
    shuffled = _shuffle(create_deck(), rng)
    human_hand = sort_hand(shuffled[0:13])
    bot_hand = sort_hand(shuffled[13:26])
    trump = shuffled[26]
    leader = initial_leader_for(round_num)
    return {
        "humanHand": human_hand,
        "botHand": bot_hand,
        "trump": trump,
        "leader": leader,
        "ledCard": None,
        "awaiting": leader,
        "tricksWon": {"human": 0, "bot": 0},
        "score": score,
        "roundNum": round_num,
        "trickNum": 1,
        "phase": "playing",
        "lastTrick": None,
        "trickHistory": [],
    }


def create_game(rng: Optional[_random_module.Random] = None) -> dict:
    return deal_round(1, {"human": 0, "bot": 0}, rng=rng)


def legal_moves(hand: list[dict], led_card: Optional[dict]) -> list[dict]:
    if led_card is None:
        return list(hand)
    same_suit = [c for c in hand if c["suit"] == led_card["suit"]]
    return same_suit if same_suit else list(hand)


def trick_winner(led_card: dict, follow_card: dict, trump_suit: str) -> str:
    """Returns 'lead' or 'follow'."""
    lead_is_trump = led_card["suit"] == trump_suit
    follow_is_trump = follow_card["suit"] == trump_suit
    if lead_is_trump and not follow_is_trump:
        return "lead"
    if not lead_is_trump and follow_is_trump:
        return "follow"
    # Same trump-ness — either both trump, both led-suit, or the follower
    # threw off (didn't follow suit and didn't trump, so they auto-lose).
    if follow_card["suit"] != led_card["suit"]:
        return "lead"
    return "follow" if follow_card["rank"] > led_card["rank"] else "lead"


def score_for_tricks(n: int) -> int:
    """Lite per-round scoring: non-monotonic; 0-3 and 7-9 are equally rewarded."""
    if n <= 3:
        return 6
    if n == 4:
        return 1
    if n == 5:
        return 2
    if n == 6:
        return 3
    if n <= 9:
        return 6
    return 0


def _remove_card(hand: list[dict], card: dict) -> list[dict]:
    return [c for c in hand if c["id"] != card["id"]]


def _player_key(player: str) -> str:
    return "human" if player == HUMAN else "bot"


def _other_player(player: str) -> str:
    return BOT if player == HUMAN else HUMAN


def play_card(state: dict, card: dict) -> dict:
    """Apply a single card play. Caller is responsible for only calling this
    when state['awaiting'] is the player who owns `card`."""
    player = state["awaiting"]
    hand_key = "humanHand" if player == HUMAN else "botHand"
    new_hand = _remove_card(state[hand_key], card)
    play_event = {"trick": state["trickNum"], "player": player, "card": card}
    trick_history = [*state["trickHistory"], play_event]

    if state["ledCard"] is None:
        # Leading.
        return {
            **state,
            hand_key: new_hand,
            "ledCard": card,
            "awaiting": _other_player(player),
            "trickHistory": trick_history,
        }

    # Following — resolve the trick.
    winner_side = trick_winner(state["ledCard"], card, state["trump"]["suit"])
    winner = state["leader"] if winner_side == "lead" else player
    tricks_won = {
        **state["tricksWon"],
        _player_key(winner): state["tricksWon"][_player_key(winner)] + 1,
    }
    return {
        **state,
        hand_key: new_hand,
        "ledCard": None,
        "awaiting": None,
        "leader": winner,
        "tricksWon": tricks_won,
        "phase": "trick-complete",
        "lastTrick": {
            "leadCard": state["ledCard"],
            "followCard": card,
            "leader": state["leader"],
            "winner": winner,
        },
        "trickHistory": trick_history,
    }


def advance_after_trick(state: dict) -> dict:
    next_trick_num = state["trickNum"] + 1
    if next_trick_num > TRICKS_PER_ROUND:
        return {
            **state,
            "lastTrick": None,
            "trickNum": next_trick_num,
            "awaiting": None,
            "phase": "round-over",
        }
    return {
        **state,
        "lastTrick": None,
        "trickNum": next_trick_num,
        "awaiting": state["leader"],
        "phase": "playing",
    }


def end_round(state: dict, rng: Optional[_random_module.Random] = None) -> dict:
    human_pts = score_for_tricks(state["tricksWon"]["human"])
    bot_pts = score_for_tricks(state["tricksWon"]["bot"])
    new_score = {
        "human": state["score"]["human"] + human_pts,
        "bot": state["score"]["bot"] + bot_pts,
    }
    if new_score["human"] >= TARGET_SCORE or new_score["bot"] >= TARGET_SCORE:
        return {
            **state,
            "score": new_score,
            "awaiting": None,
            "phase": "match-over",
        }
    return deal_round(state["roundNum"] + 1, new_score, rng=rng)


def round_summary(state: dict) -> dict:
    return {
        "human": {
            "tricks": state["tricksWon"]["human"],
            "points": score_for_tricks(state["tricksWon"]["human"]),
        },
        "bot": {
            "tricks": state["tricksWon"]["bot"],
            "points": score_for_tricks(state["tricksWon"]["bot"]),
        },
    }

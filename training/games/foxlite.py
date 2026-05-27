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


# ---------------------------------------------------------------------------
# Search-side helpers (mirror src/engine/nnGame.js).
# Used by the AlphaZero pipeline; not by parity-tested rule functions above.
# ---------------------------------------------------------------------------

CARDS_BY_INDEX: list[dict] = create_deck()
# Cards are listed (suit_outer, rank_inner) matching the JS canonical order so
# index 0 = bells-1, index 32 = moons-11. Used for both action indices and the
# one-hot positions in encode().
CARD_INDEX: dict[str, int] = {c["id"]: i for i, c in enumerate(CARDS_BY_INDEX)}
NUM_CARDS = len(CARDS_BY_INDEX)
TARGET_SCORE = 21


def bot_infoset(state: dict) -> dict:
    """Strip humanHand from a perfect-info state so search code can't peek."""
    return {
        "botHand": state["botHand"],
        "trump": state["trump"],
        "trickHistory": state["trickHistory"],
        "leader": state["leader"],
        "ledCard": state["ledCard"],
        "awaiting": state["awaiting"],
        "tricksWon": state["tricksWon"],
        "score": state["score"],
        "roundNum": state["roundNum"],
        "trickNum": state["trickNum"],
        "phase": state["phase"],
    }


def opponent_void_suits(infoset: dict) -> set[str]:
    """Suits the opponent (HUMAN) has been observed to be void in."""
    voids: set[str] = set()
    by_trick: dict[int, list[dict]] = {}
    for ev in infoset["trickHistory"]:
        by_trick.setdefault(ev["trick"], []).append(ev)
    for events in by_trick.values():
        if len(events) < 2:
            continue
        lead, follow = events[0], events[1]
        if follow["player"] == HUMAN and follow["card"]["suit"] != lead["card"]["suit"]:
            voids.add(lead["card"]["suit"])
    return voids


def sample_determinization(
    infoset: dict, rng: Optional[_random_module.Random] = None
) -> dict:
    """Sample a consistent opponent hand + unused pile from the bot's POV."""
    rng = rng or _random_module.Random()
    seen: set[str] = {c["id"] for c in infoset["botHand"]}
    seen.add(infoset["trump"]["id"])
    for ev in infoset["trickHistory"]:
        seen.add(ev["card"]["id"])

    unseen = [c for c in CARDS_BY_INDEX if c["id"] not in seen]
    voids = opponent_void_suits(infoset)
    allowed_for_opp = [c for c in unseen if c["suit"] not in voids]
    must_go_unused = [c for c in unseen if c["suit"] in voids]

    opp_played = sum(1 for ev in infoset["trickHistory"] if ev["player"] == HUMAN)
    opp_hand_size = 13 - opp_played

    if len(allowed_for_opp) < opp_hand_size:
        raise ValueError(
            f"Determinization impossible: {len(allowed_for_opp)} non-void cards "
            f"available, opponent must hold {opp_hand_size}"
        )

    pool = list(allowed_for_opp)
    rng.shuffle(pool)
    opponent_hand = pool[:opp_hand_size]
    unused_pile = pool[opp_hand_size:] + must_go_unused
    return {"opponentHand": opponent_hand, "unusedPile": unused_pile}


def world_from_determinization(state: dict, det: dict) -> dict:
    """Splice the sampled opponent hand into a perfect-info world."""
    return {**state, "humanHand": det["opponentHand"]}


def step_world(world: dict, card: dict) -> dict:
    """One half-move; auto-advances past trick-complete so MCTS sees plies."""
    nxt = play_card(world, card)
    while nxt["phase"] == "trick-complete":
        nxt = advance_after_trick(nxt)
    return nxt


def is_world_terminal(world: dict) -> bool:
    return world["phase"] in ("round-over", "match-over")


def signed_margin_value(world: dict, mover: str) -> float:
    """End-of-round signed point margin / 6, from `mover`'s perspective."""
    bot_pts = score_for_tricks(world["tricksWon"]["bot"])
    human_pts = score_for_tricks(world["tricksWon"]["human"])
    v_bot = (bot_pts - human_pts) / 6
    return v_bot if mover == BOT else -v_bot


def rollout_value(
    world: dict, mover: str, rng: Optional[_random_module.Random] = None
) -> float:
    """Uniform-random rollout to round-end. Value in `mover`'s frame."""
    rng = rng or _random_module.Random()
    s = world
    while not is_world_terminal(s):
        hand_key = "humanHand" if s["awaiting"] == HUMAN else "botHand"
        legal = legal_moves(s[hand_key], s["ledCard"])
        s = step_world(s, rng.choice(legal))
    return signed_margin_value(s, mover)


def legal_action_indices(world: dict) -> list[int]:
    """Legal moves expressed as integer card indices, ascending."""
    hand_key = "humanHand" if world["awaiting"] == HUMAN else "botHand"
    legal = legal_moves(world[hand_key], world["ledCard"])
    return sorted(CARD_INDEX[c["id"]] for c in legal)


def card_from_index(idx: int) -> dict:
    return CARDS_BY_INDEX[idx]


# ---------------------------------------------------------------------------
# Network input encoding — mirrors src/engine/nnGame.js encode().
# Layout (cursor increments shown alongside, total = INPUT_SIZE):
#   own hand                       33
#   played pile                    33
#   trump suit                      3
#   trump card identity            33
#   led card + "no-led" flag       34
#   self tricks-won (one-hot 0..13) 14
#   opp tricks-won                  14
#   opp suit voids                   3   (only set when mover is the bot)
#   "I led this trick" flag          1
#   trick number (one-hot 1..13)    13
#   self match score (scalar)        1
#   opp match score (scalar)         1
#   total                          183
# ---------------------------------------------------------------------------

INPUT_SIZE = 183


def encode(state: dict, mover: Optional[str] = None) -> list[float]:
    """Encode a state from `mover`'s perspective. Default mover = state.awaiting.

    Mirrors src/engine/nnGame.js encode() byte-for-byte so parity check works.
    """
    if mover is None:
        mover = state["awaiting"]
    out = [0.0] * INPUT_SIZE
    cursor = 0

    mover_is_human = mover == HUMAN
    own_hand = state["humanHand"] if mover_is_human else state["botHand"]
    own_tricks = state["tricksWon"]["human"] if mover_is_human else state["tricksWon"]["bot"]
    opp_tricks = state["tricksWon"]["bot"] if mover_is_human else state["tricksWon"]["human"]
    own_score = state["score"]["human"] if mover_is_human else state["score"]["bot"]
    opp_score = state["score"]["bot"] if mover_is_human else state["score"]["human"]

    # own hand
    for c in own_hand:
        out[cursor + CARD_INDEX[c["id"]]] = 1.0
    cursor += NUM_CARDS

    # played pile
    for ev in state["trickHistory"]:
        out[cursor + CARD_INDEX[ev["card"]["id"]]] = 1.0
    cursor += NUM_CARDS

    # trump suit
    out[cursor + SUITS.index(state["trump"]["suit"])] = 1.0
    cursor += len(SUITS)

    # trump card identity
    out[cursor + CARD_INDEX[state["trump"]["id"]]] = 1.0
    cursor += NUM_CARDS

    # led card
    if state["ledCard"] is not None:
        out[cursor + CARD_INDEX[state["ledCard"]["id"]]] = 1.0
    else:
        out[cursor + NUM_CARDS] = 1.0
    cursor += NUM_CARDS + 1

    # self tricks (one-hot 0..13)
    out[cursor + own_tricks] = 1.0
    cursor += TRICKS_PER_ROUND + 1
    # opp tricks
    out[cursor + opp_tricks] = 1.0
    cursor += TRICKS_PER_ROUND + 1

    # opponent voids — only meaningful from the bot's view
    if not mover_is_human:
        for s in opponent_void_suits({"trickHistory": state["trickHistory"]}):
            out[cursor + SUITS.index(s)] = 1.0
    cursor += len(SUITS)

    # leader-of-this-trick flag
    out[cursor] = 1.0 if (state["leader"] == mover and state["ledCard"] is None) else 0.0
    cursor += 1

    # trick number 1..13
    out[cursor + state["trickNum"] - 1] = 1.0
    cursor += TRICKS_PER_ROUND

    # scores as scalars
    out[cursor] = own_score / TARGET_SCORE
    cursor += 1
    out[cursor] = opp_score / TARGET_SCORE
    cursor += 1

    if cursor != INPUT_SIZE:
        raise RuntimeError(f"encoder cursor mismatch: {cursor} != {INPUT_SIZE}")
    return out

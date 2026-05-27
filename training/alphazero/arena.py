"""Promotion gate: pit a challenger checkpoint against the current best.

Runs ARENA_GAMES matches, swapping seats halfway so each side is bot for
half the run. Returns the challenger's match-win rate; the caller compares
against config.ARENA_WIN_RATE to decide whether to promote.

PIMC is run at training-time budgets here (deeper search would distort the
relative-strength signal we care about).
"""

from __future__ import annotations

import random

import numpy as np

import config
from alphazero.network import PolicyValueNet
from alphazero.pimc import aggregate_visit_counts
from alphazero.selfplay import net_evaluator
from games.foxlite import (
    BOT,
    CARDS_BY_INDEX,
    HUMAN,
    advance_after_trick,
    create_game,
    end_round,
    legal_moves,
    play_card,
)


def _choose_action(net: PolicyValueNet, state: dict, rng: random.Random) -> dict:
    counts, _, _ = aggregate_visit_counts(
        state, net_evaluator(net),
        num_determinizations=config.NUM_DETERMINIZATIONS,
        num_simulations=config.NUM_SIMULATIONS,
        c_puct=config.C_PUCT,
        add_dirichlet_noise=False,
        rng=rng,
    )
    action = int(counts.argmax()) if counts.sum() > 0 else _first_legal_index(state)
    card = CARDS_BY_INDEX[action]
    hand_key = "humanHand" if state["awaiting"] == HUMAN else "botHand"
    legal = legal_moves(state[hand_key], state["ledCard"])
    legal_ids = {c["id"] for c in legal}
    if card["id"] not in legal_ids:
        card = legal[0]
    return card


def _first_legal_index(state: dict) -> int:
    from games.foxlite import CARD_INDEX
    hand_key = "humanHand" if state["awaiting"] == HUMAN else "botHand"
    legal = legal_moves(state[hand_key], state["ledCard"])
    return CARD_INDEX[legal[0]["id"]]


def play_match(net_bot: PolicyValueNet, net_human: PolicyValueNet,
               rng: random.Random) -> dict:
    """Play one full match. `net_bot` always plays the bot seat; `net_human`
    plays the human seat. Returns the final score dict."""
    state = create_game(rng=rng)
    while state["phase"] != "match-over":
        if state["phase"] == "round-over":
            state = end_round(state, rng=rng)
            continue
        if state["phase"] == "trick-complete":
            state = advance_after_trick(state)
            continue
        net = net_bot if state["awaiting"] == BOT else net_human
        card = _choose_action(net, state, rng)
        state = play_card(state, card)
    return state["score"]


def play_series(challenger: PolicyValueNet, champion: PolicyValueNet,
                num_matches: int, rng: random.Random) -> dict:
    """Play `num_matches` head-to-head matches with seats swapping at the
    halfway mark. Returns a result dict including `challenger_win_rate`.
    """
    wins = losses = draws = 0
    margins: list[int] = []  # signed (challenger - champion) per-match point margin
    for i in range(num_matches):
        challenger_is_bot = i < num_matches / 2
        if challenger_is_bot:
            score = play_match(challenger, champion, rng)
            c_pts, h_pts = score["bot"], score["human"]
        else:
            score = play_match(champion, challenger, rng)
            c_pts, h_pts = score["human"], score["bot"]
        margins.append(c_pts - h_pts)
        if c_pts > h_pts:
            wins += 1
        elif h_pts > c_pts:
            losses += 1
        else:
            draws += 1

    total = num_matches
    return {
        "challenger_wins": wins,
        "losses": losses,
        "draws": draws,
        "total": total,
        "challenger_win_rate": wins / total,
        "mean_point_margin": float(np.mean(margins)),
    }

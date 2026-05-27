"""Self-play game generation for AlphaZero training.

Plays a full match (or up to a configurable number of rounds) using the
current network for both seats. At each decision:
  - encode the moving player's view
  - run PIMC with Dirichlet noise on the root prior
  - record (encoded_input, MCTS visit-count distribution, moving_player)
  - sample an action (temperature=1 for the first TEMPERATURE_MOVES plies of
    the round, argmax thereafter)

When a round ends, every recorded decision from that round gets its value
target filled in: signed margin / 6, from the recorder's frame.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch

import config
from alphazero.network import PolicyValueNet, infer
from alphazero.pimc import Evaluator, aggregate_visit_counts
from games.foxlite import (
    BOT,
    CARDS_BY_INDEX,
    HUMAN,
    NUM_CARDS,
    TARGET_SCORE,
    advance_after_trick,
    create_game,
    encode,
    end_round,
    legal_moves,
    play_card,
    score_for_tricks,
    signed_margin_value,
)


def net_evaluator(net: PolicyValueNet) -> Evaluator:
    """Wrap a network into the PIMC evaluator interface."""
    def evaluator(state: dict, mover: str) -> tuple[list[float], float]:
        x = encode(state, mover)
        logits, value = infer(net, x)
        return logits, value
    return evaluator


def play_one_round(
    net: PolicyValueNet,
    state: dict,
    rng: random.Random,
) -> tuple[dict, list[tuple[list[float], np.ndarray, str]]]:
    """Play one round from `state` to round-over, recording each decision.

    Returns (final_state, decisions). Each decision is
    (encoded_input, policy_target, mover). Value targets are filled in after
    the round ends (by the caller, who has access to the final state).
    """
    evaluator = net_evaluator(net)
    decisions: list[tuple[list[float], np.ndarray, str]] = []

    while state["phase"] not in ("round-over", "match-over"):
        if state["phase"] == "trick-complete":
            state = advance_after_trick(state)
            continue
        # phase == 'playing'
        mover = state["awaiting"]
        x = encode(state, mover)
        counts, _ = aggregate_visit_counts(
            state, evaluator,
            num_determinizations=config.NUM_DETERMINIZATIONS,
            num_simulations=config.NUM_SIMULATIONS,
            c_puct=config.C_PUCT,
            add_dirichlet_noise=True,
            dirichlet_alpha=config.DIRICHLET_ALPHA,
            dirichlet_epsilon=config.DIRICHLET_EPSILON,
            rng=rng,
        )

        # Policy target: visit-count proportions over the full 33-card space.
        total = counts.sum()
        policy = counts / total if total > 0 else _uniform_legal(state)
        decisions.append((x, policy, mover))

        # Move selection: temperature 1 for the first N plies of the round,
        # argmax thereafter. Plies are counted from trick 1 — easy proxy.
        ply = sum(1 for ev in state["trickHistory"])
        if ply < config.TEMPERATURE_MOVES and total > 0:
            action = int(np.random.choice(NUM_CARDS, p=counts / total))
        else:
            action = int(counts.argmax())

        card = CARDS_BY_INDEX[action]
        # Safety: action must be legal. If MCTS produced 0 visits everywhere
        # (shouldn't), fall back to a legal card.
        hand_key = "humanHand" if mover == HUMAN else "botHand"
        legal_ids = {c["id"] for c in legal_moves(state[hand_key], state["ledCard"])}
        if card["id"] not in legal_ids:
            card = next(iter(legal_moves(state[hand_key], state["ledCard"])))
        state = play_card(state, card)

    return state, decisions


def _uniform_legal(state: dict) -> np.ndarray:
    """Fallback policy: uniform over legal actions."""
    from games.foxlite import CARD_INDEX
    hand_key = "humanHand" if state["awaiting"] == HUMAN else "botHand"
    legal = legal_moves(state[hand_key], state["ledCard"])
    p = np.zeros(NUM_CARDS, dtype=np.float64)
    if legal:
        for c in legal:
            p[CARD_INDEX[c["id"]]] = 1.0 / len(legal)
    return p


def play_one_match(
    net: PolicyValueNet,
    rng: random.Random,
    max_rounds: int = 30,
) -> list[tuple[list[float], np.ndarray, float]]:
    """Play one full match (to TARGET_SCORE or `max_rounds` rounds), returning
    every decision tuple `(input, policy, value)` ready for the replay buffer.
    """
    state = create_game(rng=rng)
    all_records: list[tuple[list[float], np.ndarray, float]] = []
    rounds = 0
    while state["phase"] != "match-over" and rounds < max_rounds:
        final, decisions = play_one_round(net, state, rng)
        # signed margin from BOT's frame for THIS round only.
        bot_pts = score_for_tricks(final["tricksWon"]["bot"])
        human_pts = score_for_tricks(final["tricksWon"]["human"])
        margin_bot_frame = (bot_pts - human_pts) / 6.0
        for x, policy, mover in decisions:
            value = margin_bot_frame if mover == BOT else -margin_bot_frame
            all_records.append((x, policy, value))
        # Advance: end_round seeds the next round (or ends match).
        state = end_round(final, rng=rng)
        rounds += 1
    return all_records

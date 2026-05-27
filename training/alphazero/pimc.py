"""PIMC: Perfect Information Monte Carlo search.

At decision time:
  1. Sample K determinizations consistent with the mover's information set.
  2. In each, run a PUCT MCTS tree (perfect info inside the determinization).
  3. Sum visit counts at the root across all K trees.
  4. Sample / argmax from the aggregated distribution.

Mirrors src/engine/pimc.js. Values are stored in the *root mover's* frame; we
sign-flip Q at nodes where the in-tree mover differs from the root mover.
Trick-taking doesn't strictly alternate movers (the trick winner leads next),
so the standard "flip on backup" pattern doesn't apply — keeping a fixed
frame is simpler and correct.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Optional

import numpy as np

from games.foxlite import (
    BOT,
    HUMAN,
    CARD_INDEX,
    CARDS_BY_INDEX,
    NUM_CARDS,
    bot_infoset,
    is_world_terminal,
    legal_action_indices,
    rollout_value,
    sample_determinization,
    signed_margin_value,
    step_world,
    world_from_determinization,
)

# An evaluator returns (priors_over_full_action_space, value_in_mover_frame).
# `state` is the world at this MCTS node; `mover` is state['awaiting'].
Evaluator = Callable[[dict, str], tuple[list[float], float]]


class Node:
    __slots__ = (
        "state", "prior", "visit_count", "value_sum",
        "children", "expanded", "is_terminal",
    )

    def __init__(self, state: dict, prior: float):
        self.state = state
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0  # root-mover frame
        self.children: dict[int, Node] = {}  # action_index -> Node
        self.expanded = False
        self.is_terminal = is_world_terminal(state)

    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def _expand(node: Node, evaluator: Evaluator, root_mover: str) -> float:
    """Expand `node` and return the leaf-value estimate in root_mover frame."""
    if node.is_terminal:
        return signed_margin_value(node.state, root_mover)

    mover = node.state["awaiting"]
    priors_full, value_in_mover_frame = evaluator(node.state, mover)
    legal = legal_action_indices(node.state)
    # Masked softmax over legal actions only.
    legal_logits = [priors_full[a] for a in legal]
    m = max(legal_logits)
    exps = [math.exp(lg - m) for lg in legal_logits]
    z = sum(exps)
    for action, e in zip(legal, exps):
        child_state = step_world(node.state, CARDS_BY_INDEX[action])
        node.children[action] = Node(child_state, prior=e / z)
    node.expanded = True

    sign = 1.0 if mover == root_mover else -1.0
    return sign * value_in_mover_frame


def _puct_select_child(node: Node, c_puct: float, root_mover: str) -> tuple[int, Node]:
    """PUCT child selection. Ties broken by lowest action index."""
    mover = node.state["awaiting"]
    sign = 1.0 if mover == root_mover else -1.0
    sqrt_n = math.sqrt(node.visit_count)
    best_score = -math.inf
    best_action = next(iter(node.children))
    best_child = node.children[best_action]
    for action in sorted(node.children.keys()):
        child = node.children[action]
        q = sign * child.mean_value()
        u = c_puct * child.prior * sqrt_n / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def run_mcts(
    root_state: dict,
    evaluator: Evaluator,
    num_simulations: int,
    c_puct: float,
    add_dirichlet_noise: bool = False,
    dirichlet_alpha: float = 0.5,
    dirichlet_epsilon: float = 0.25,
    rng: Optional[random.Random] = None,
) -> Node:
    """Run `num_simulations` PUCT simulations from `root_state`.

    Values are recorded in the root mover's frame so the caller can read off
    visit counts (policy target) and Q-values (informational) directly.
    """
    root_mover = root_state["awaiting"]
    root = Node(root_state, prior=0.0)
    _expand(root, evaluator, root_mover)

    if add_dirichlet_noise and root.children:
        rng_np = np.random.default_rng(rng.randrange(2**31) if rng else None)
        actions = list(root.children.keys())
        noise = rng_np.dirichlet([dirichlet_alpha] * len(actions))
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = child.prior * (1 - dirichlet_epsilon) + float(n) * dirichlet_epsilon

    for _ in range(num_simulations):
        node = root
        path = [node]
        while node.expanded and not node.is_terminal:
            _, child = _puct_select_child(node, c_puct, root_mover)
            node = child
            path.append(node)

        if node.is_terminal:
            value = signed_margin_value(node.state, root_mover)
        else:
            value = _expand(node, evaluator, root_mover)

        for n in path:
            n.visit_count += 1
            n.value_sum += value

    return root


def aggregate_visit_counts(
    state: dict,
    evaluator: Evaluator,
    num_determinizations: int,
    num_simulations: int,
    c_puct: float,
    add_dirichlet_noise: bool = False,
    dirichlet_alpha: float = 0.5,
    dirichlet_epsilon: float = 0.25,
    rng: Optional[random.Random] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run PIMC and return (counts[NUM_CARDS], q_values[NUM_CARDS], root_q).

    `state` is from the mover's seat (its own hand is `state[mover-hand]`).
    `evaluator` is called with perfect-info world states inside each rollout.
    Counts and Q-values cover the full 33-card action space; non-legal /
    non-explored actions are 0. `root_q` is the visit-weighted root value
    averaged across all determinizations — a low-variance estimate of the
    position's value in the mover's frame, suitable as a value-head target.
    """
    rng = rng or random.Random()
    mover = state["awaiting"]
    infoset = bot_infoset(state) if mover == BOT else _infoset_as_human(state)

    counts = np.zeros(NUM_CARDS, dtype=np.float64)
    q_sums = np.zeros(NUM_CARDS, dtype=np.float64)
    root_value_sum = 0.0
    root_visits_sum = 0
    for _ in range(num_determinizations):
        det = sample_determinization(infoset, rng) if mover == BOT \
            else _sample_human_determinization(state, rng)
        world = world_from_determinization(state, det) if mover == BOT \
            else _world_from_human_determinization(state, det)
        root = run_mcts(
            world, evaluator, num_simulations, c_puct,
            add_dirichlet_noise=add_dirichlet_noise,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            rng=rng,
        )
        root_value_sum += root.value_sum
        root_visits_sum += root.visit_count
        for action, child in root.children.items():
            counts[action] += child.visit_count
            q_sums[action] += child.visit_count * child.mean_value()

    q_values = np.zeros(NUM_CARDS, dtype=np.float64)
    nz = counts > 0
    q_values[nz] = q_sums[nz] / counts[nz]
    root_q = root_value_sum / root_visits_sum if root_visits_sum > 0 else 0.0
    return counts, q_values, root_q


# In self-play the *human* seat also makes decisions; symmetry with the bot.
def _infoset_as_human(state: dict) -> dict:
    """Mirror of bot_infoset but for the HUMAN seat."""
    return {
        "humanHand": state["humanHand"],
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


def _sample_human_determinization(state: dict, rng: random.Random) -> dict:
    """Sample a consistent bot-hand + unused pile from the human's POV."""
    seen: set[str] = {c["id"] for c in state["humanHand"]}
    seen.add(state["trump"]["id"])
    for ev in state["trickHistory"]:
        seen.add(ev["card"]["id"])

    # Inferred bot voids from history (mirror of opponent_void_suits but for
    # the other seat).
    voids: set[str] = set()
    by_trick: dict[int, list[dict]] = {}
    for ev in state["trickHistory"]:
        by_trick.setdefault(ev["trick"], []).append(ev)
    for events in by_trick.values():
        if len(events) < 2:
            continue
        lead, follow = events[0], events[1]
        if follow["player"] == BOT and follow["card"]["suit"] != lead["card"]["suit"]:
            voids.add(lead["card"]["suit"])

    unseen = [c for c in CARDS_BY_INDEX if c["id"] not in seen]
    allowed = [c for c in unseen if c["suit"] not in voids]
    must_go_unused = [c for c in unseen if c["suit"] in voids]

    bot_played = sum(1 for ev in state["trickHistory"] if ev["player"] == BOT)
    bot_hand_size = 13 - bot_played

    if len(allowed) < bot_hand_size:
        raise ValueError(
            f"Determinization impossible (human POV): {len(allowed)} non-void "
            f"cards available, bot must hold {bot_hand_size}"
        )

    pool = list(allowed)
    rng.shuffle(pool)
    bot_hand = pool[:bot_hand_size]
    unused = pool[bot_hand_size:] + must_go_unused
    return {"opponentHand": bot_hand, "unusedPile": unused}


def _world_from_human_determinization(state: dict, det: dict) -> dict:
    """Splice the sampled bot hand into a perfect-info world."""
    return {**state, "botHand": det["opponentHand"]}


def uniform_rollout_evaluator(rng: Optional[random.Random] = None) -> Evaluator:
    """Phase-2-equivalent evaluator: uniform prior, rollout value."""
    rng = rng or random.Random()
    def evaluator(state: dict, mover: str) -> tuple[list[float], float]:
        priors = [1.0 / NUM_CARDS] * NUM_CARDS  # PIMC re-normalizes over legal
        return priors, rollout_value(state, mover, rng)
    return evaluator


def hybrid_evaluator(net, rng: Optional[random.Random] = None) -> Evaluator:
    """Self-play training evaluator: NET policy + ROLLOUT value at leaves.

    Breaks the value head's self-bootstrapping during training — leaf values
    come from grounded rollouts, not the network's own predictions. The
    policy head still steers the search via priors, so MCTS visit counts and
    the resulting policy targets carry the policy head's current beliefs.

    Inference (deployed engine) uses pure `net_evaluator` — rollouts at
    leaves would be too slow at PIMC's play-time budget.
    """
    rng = rng or random.Random()
    # Local imports keep pimc.py importable without dragging alphazero.network
    # into modules that just use the search.
    from alphazero.network import infer
    from games.foxlite import encode, rollout_value

    def evaluator(state: dict, mover: str) -> tuple[list[float], float]:
        logits, _ = infer(net, encode(state, mover))
        return logits, rollout_value(state, mover, rng)
    return evaluator

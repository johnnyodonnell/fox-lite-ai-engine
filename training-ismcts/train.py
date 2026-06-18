"""AlphaZero training step: sample a minibatch from the replay buffer, compute
the policy cross-entropy + value MSE loss, and step the optimizer.

Replaces the former REINFORCE update. The self-play worker now produces
search-improved targets: `pi` is the ISMCTS root visit distribution (a soft
target over the 33 canonical card slots, zero on illegal moves) and `z` is the
match outcome (±1) from the deciding seat's perspective. So this is plain
supervised AlphaZero training — no advantage, no entropy bonus, no legal mask
(the target already has zero mass on illegal moves).
"""

import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    """Ring buffer of (state, pi, z) rows in preallocated numpy arrays (a deque's
    O(n) random indexing makes sampling at multi-million capacity prohibitive)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.states = None  # allocated on first add, from the row shapes
        self.pis = None
        self.beliefs = None  # opponent-hand belief targets (sentinel-encoded)
        self.zs = np.empty(capacity, dtype=np.float32)
        self.next = 0
        self.size = 0

    def add_many(self, tuples):
        for state, pi, z, belief in tuples:
            if self.states is None:
                self.states = np.empty((self.capacity, len(state)), dtype=np.float32)
                self.pis = np.empty((self.capacity, len(pi)), dtype=np.float32)
                self.beliefs = np.empty((self.capacity, len(belief)), dtype=np.float32)
            i = self.next
            self.states[i] = state
            self.pis[i] = pi
            self.zs[i] = z
            self.beliefs[i] = belief
            self.next = (i + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size, rng=None):
        rng = rng or np.random
        if self.size == 0:
            return None
        idxs = rng.choice(self.size, size=batch_size, replace=True)
        return self.states[idxs], self.pis[idxs], self.zs[idxs], self.beliefs[idxs]


def train_step(net, opt, batch, device, *, grad_clip=1.0, lambda_belief=1.0):
    states, pis, zs, beliefs = batch
    s = torch.from_numpy(states).to(device)
    pi = torch.from_numpy(pis).to(device)
    z = torch.from_numpy(zs).to(device)
    b = torch.from_numpy(beliefs).to(device)

    logits, v, belief_logits = net(s)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(pi * log_probs).sum(dim=1).mean()
    # H(pi): entropy of the ISMCTS target = irreducible floor of policy_loss.
    # policy_loss - target_entropy == KL(pi || p_net), the reducible fit error.
    target_entropy = -(pi * pi.clamp_min(1e-9).log()).sum(dim=1).mean()
    value_loss = F.mse_loss(v, z)

    # Opponent-hand belief: per-card BCE, masked to the unseen slots. The target is
    # sentinel-encoded (1=opp holds, 0=unseen-not-held, -1=seen/masked), so the mask
    # is (b >= 0) and the regression target is b clamped to {0,1}.
    belief_mask = (b >= 0.0).float()
    belief_tgt = b.clamp_min(0.0)
    belief_bce = F.binary_cross_entropy_with_logits(belief_logits, belief_tgt, reduction="none")
    belief_loss = (belief_bce * belief_mask).sum() / belief_mask.sum().clamp_min(1.0)

    loss = policy_loss + value_loss + lambda_belief * belief_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    opt.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "belief_loss": float(belief_loss.item()),
        "target_entropy": float(target_entropy.item()),
    }

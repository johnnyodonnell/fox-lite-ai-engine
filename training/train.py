"""AlphaZero training step: sample a minibatch from the replay buffer, compute
the policy cross-entropy + value MSE loss, and step the optimizer.

Replaces the former REINFORCE update. The self-play worker now produces
search-improved targets: `pi` is the ISMCTS root visit distribution (a soft
target over the 33 canonical card slots, zero on illegal moves) and `z` is the
match outcome (±1) from the deciding seat's perspective. So this is plain
supervised AlphaZero training — no advantage, no entropy bonus, no legal mask
(the target already has zero mass on illegal moves).
"""

import collections

import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    """Ring buffer of (state, pi, z) tuples."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = collections.deque(maxlen=capacity)

    def add_many(self, tuples):
        self.buf.extend(tuples)

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size, rng=None):
        rng = rng or np.random
        if len(self.buf) == 0:
            return None
        idxs = rng.choice(len(self.buf), size=batch_size, replace=True)
        states = np.stack([self.buf[i][0] for i in idxs])
        pis = np.stack([self.buf[i][1] for i in idxs])
        zs = np.array([self.buf[i][2] for i in idxs], dtype=np.float32)
        return states, pis, zs


def train_step(net, opt, batch, device, *, grad_clip=1.0):
    states, pis, zs = batch
    s = torch.from_numpy(states).to(device)
    pi = torch.from_numpy(pis).to(device)
    z = torch.from_numpy(zs).to(device)

    logits, v = net(s)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(pi * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(v, z)
    loss = policy_loss + value_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    opt.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
    }

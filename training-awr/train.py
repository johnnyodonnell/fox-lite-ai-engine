"""AWR training step: sample a minibatch from the replay buffer and apply
Advantage-Weighted Regression (Peng et al. 2019, arXiv:1910.00177).

Replaces the AlphaZero update (CE vs ISMCTS visit targets). Self-play is
search-free — the worker samples moves straight from the policy — so each row
is (state, legal_mask, action, z) and the update is the original REINFORCE
step with the linear advantage swapped for a clipped exponential weight:

  adv         = z - value(state).detach()
  w           = exp(adv / beta).clamp(max=w_max)     # always positive
  policy_loss = -(w * log pi(action|state)).mean()   # over legal moves
  value_loss  = MSE(value(state), z)
  loss        = policy_loss + value_loss - entropy_coef * entropy

Being weighted behavior cloning of better-than-expected actions, the update is
explicitly off-policy: a large replay buffer holding many stale policies is
part of the algorithm, not a compromise — so the buffer reuse pacing carries
over from the AlphaZero setup unchanged.
"""

import numpy as np
import torch
import torch.nn.functional as F

MASK_NEG = 1.0e9


class ReplayBuffer:
    """Ring buffer of (state, mask, action, z) rows in preallocated numpy
    arrays. Ingest is block-wise (`add_block`): search-free self-play produces
    ~100x the rows/sec of ISMCTS, so per-row Python loops can't keep up."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.states = None  # allocated on first add, from the block shapes
        self.masks = None
        self.actions = np.empty(capacity, dtype=np.int64)
        self.zs = np.empty(capacity, dtype=np.float32)
        self.next = 0
        self.size = 0

    def add_block(self, states, masks, actions, zs):
        n = len(zs)
        if n == 0:
            return
        if n > self.capacity:  # keep only the newest capacity rows
            states, masks = states[-self.capacity:], masks[-self.capacity:]
            actions, zs = actions[-self.capacity:], zs[-self.capacity:]
            n = self.capacity
        if self.states is None:
            self.states = np.empty((self.capacity, states.shape[1]), dtype=np.float32)
            self.masks = np.empty((self.capacity, masks.shape[1]), dtype=np.float32)
        first = min(n, self.capacity - self.next)  # rows before the ring wraps
        for dst, src in ((self.states, states), (self.masks, masks),
                         (self.actions, actions), (self.zs, zs)):
            dst[self.next:self.next + first] = src[:first]
            if first < n:
                dst[:n - first] = src[first:]
        self.next = (self.next + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size, rng=None):
        rng = rng or np.random
        if self.size == 0:
            return None
        idxs = rng.choice(self.size, size=batch_size, replace=True)
        return self.states[idxs], self.masks[idxs], self.actions[idxs], self.zs[idxs]


def train_step(net, opt, batch, device, *, beta=1.0, w_max=20.0,
               entropy_coef=0.0, grad_clip=1.0):
    states, masks, actions, zs = batch
    s = torch.from_numpy(states).to(device)
    m = torch.from_numpy(masks).to(device)
    a = torch.from_numpy(actions).to(device)
    z = torch.from_numpy(zs).to(device)

    logits, v = net(s)
    logp = F.log_softmax(logits + (m - 1.0) * MASK_NEG, dim=1)
    logp_a = logp.gather(1, a[:, None]).squeeze(1)
    adv = z - v.detach()
    w = (adv / beta).exp().clamp(max=w_max)
    policy_loss = -(w * logp_a).mean()
    value_loss = F.mse_loss(v, z)
    # Entropy of the legal-masked policy. Logged always (collapse watch);
    # entropy_coef=0 is faithful AWR, >0 adds the REINFORCE-style bonus.
    entropy = -(logp.exp() * logp).sum(dim=1).mean()
    loss = policy_loss + value_loss - entropy_coef * entropy

    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    opt.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        # Mean AWR weight: ~1 with an uninformative critic; drifting toward 0
        # or pinning at w_max signals beta is mis-scaled for the advantages.
        "w_mean": float(w.mean().item()),
    }

"""REINFORCE training step over a self-play cohort (Phase 5).

On-policy policy gradient with a learned value baseline + entropy bonus:
  advantage   = z - value(state).detach()
  policy_loss = -(advantage * log pi(action|state)).mean()      # over legal moves
  value_loss  = MSE(value(state), z)
  entropy     = mean entropy of the (legal-masked) policy
  loss        = policy_loss + c_value*value_loss - c_entropy*entropy

The cohort is on-policy data from the *current* weights. Cohorts are sized so
each is a single SGD step (rows <= sgd_batch), keeping every update strictly
on-policy — the gradient is only unbiased for the policy that generated the data.
"""

import numpy as np
import torch
import torch.nn.functional as F

MASK_NEG = 1.0e9


def train_on_cohort(net, opt, cohort, device, *, sgd_batch=1024, epochs=1,
                    c_value=1.0, c_entropy=0.05, grad_clip=1.0, rng=None):
    states = torch.from_numpy(cohort["states"]).to(device)
    masks = torch.from_numpy(cohort["masks"]).to(device)
    actions = torch.from_numpy(cohort["actions"]).to(device)
    z = torch.from_numpy(cohort["z"]).to(device).float()
    n = cohort["n"]
    if rng is None:
        rng = np.random.default_rng()

    agg = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0}
    steps = 0
    net.train()
    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(n)).to(device)
        for start in range(0, n, sgd_batch):
            idx = perm[start:start + sgd_batch]
            s, m, a, zz = states[idx], masks[idx], actions[idx], z[idx]
            logits, v = net(s)
            masked = logits + (m - 1.0) * MASK_NEG
            logp = F.log_softmax(masked, dim=1)
            logp_a = logp.gather(1, a[:, None]).squeeze(1)
            adv = zz - v.detach()
            policy_loss = -(adv * logp_a).mean()
            value_loss = F.mse_loss(v, zz)
            p = logp.exp()
            entropy = -(p * logp).sum(dim=1).mean()
            loss = policy_loss + c_value * value_loss - c_entropy * entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()

            agg["loss"] += float(loss.item())
            agg["policy"] += float(policy_loss.item())
            agg["value"] += float(value_loss.item())
            agg["entropy"] += float(entropy.item())
            steps += 1

    if steps:
        for k in agg:
            agg[k] /= steps
    agg["steps"] = steps
    return agg

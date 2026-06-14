"""REINFORCE training step over a self-play cohort (Phase 5).

On-policy policy gradient with a learned value baseline + entropy bonus:
  advantage   = z - value(state).detach()
  policy_loss = -(advantage * log pi(action|state)).mean()      # over legal moves
  value_loss  = MSE(value(state), z)
  entropy     = mean entropy of the (legal-masked) policy
  loss        = policy_loss + c_value*value_loss - alpha*entropy

`alpha` (the entropy coefficient) is *adaptive*, not fixed: a SAC-style dual
variable auto-tuned to hold the measured entropy at a target. The target is a
fraction of the mean max-entropy log(n_legal), so it auto-scales to how
constrained positions are (legal-move counts vary widely here). This is what
keeps the policy from collapsing under the dense, high-magnitude per-round
reward — a fixed coefficient is an open-loop tug-of-war the reward wins.

  alpha      = exp(log_alpha)
  H_target   = ent_target_frac * mean(log n_legal)
  alpha_loss = alpha * (entropy.detach() - H_target)   # GD on log_alpha:
                                                       # H<target -> alpha up

The cohort is on-policy data from the *current* weights, so we make only a small
number of passes (epochs) per cohort to avoid drifting off-policy.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

MASK_NEG = 1.0e9


def train_on_cohort(net, opt, cohort, device, *, sgd_batch=1024, epochs=1,
                    c_value=1.0, log_alpha=None, alpha_opt=None,
                    ent_target_frac=0.5, alpha_min=1e-3, alpha_max=0.5,
                    grad_clip=1.0, rng=None):
    states = torch.from_numpy(cohort["states"]).to(device)
    masks = torch.from_numpy(cohort["masks"]).to(device)
    actions = torch.from_numpy(cohort["actions"]).to(device)
    z = torch.from_numpy(cohort["z"]).to(device).float()
    n = cohort["n"]
    if rng is None:
        rng = np.random.default_rng()
    log_alpha_min, log_alpha_max = math.log(alpha_min), math.log(alpha_max)

    agg = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0,
           "alpha": 0.0, "ent_target": 0.0}
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
            # Target a fraction of the achievable entropy: log(#legal moves),
            # averaged over the batch. Forced moves (n_legal=1) contribute 0.
            n_legal = m.sum(dim=1).clamp_min(1.0)
            ent_target = ent_target_frac * n_legal.log().mean()

            alpha = log_alpha.exp()
            loss = policy_loss + c_value * value_loss - alpha.detach() * entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()

            # Dual update for the entropy coefficient (independent of the net
            # update above; `alpha` is detached there, so no graph conflict).
            alpha_loss = alpha * (entropy.detach() - ent_target.detach())
            alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            alpha_opt.step()
            with torch.no_grad():
                log_alpha.clamp_(log_alpha_min, log_alpha_max)

            agg["loss"] += float(loss.item())
            agg["policy"] += float(policy_loss.item())
            agg["value"] += float(value_loss.item())
            agg["entropy"] += float(entropy.item())
            agg["alpha"] += float(alpha.item())
            agg["ent_target"] += float(ent_target.item())
            steps += 1

    if steps:
        for k in agg:
            agg[k] /= steps
    agg["steps"] = steps
    return agg

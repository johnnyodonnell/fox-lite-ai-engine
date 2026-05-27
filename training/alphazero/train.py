"""Train one iteration of the policy/value net from the replay buffer.

Loss = masked cross-entropy(policy) + MSE(value). Standard AlphaZero objective;
we don't mask logits on the network side (it outputs the full 33-action space)
because policy_target distributions sum to 1 *over legal moves*, and that's
all the cross-entropy needs.

Safety: per-step NaN check; weight-norm tripwire halts training if any
parameter's L2 explodes past config.MAX_WEIGHT_NORM (a quiet AlphaZero
divergence pattern).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import config
from alphazero.network import PolicyValueNet
from alphazero.replay_buffer import ReplayBuffer


@dataclass
class TrainStats:
    steps: int
    policy_loss: float
    value_loss: float
    total_loss: float


def train_iteration(
    net: PolicyValueNet,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    device: torch.device,
) -> TrainStats:
    net.train()
    p_acc = v_acc = t_acc = 0.0
    steps = 0
    for _ in range(config.TRAIN_STEPS_PER_ITER):
        if len(buffer) < config.BATCH_SIZE:
            break
        x_np, p_np, v_np = buffer.sample(config.BATCH_SIZE, rng)
        x = torch.from_numpy(x_np).to(device)
        p_target = torch.from_numpy(p_np).to(device)
        v_target = torch.from_numpy(v_np).to(device)

        logits, value = net(x)
        # Cross-entropy on full action space. -E[p_target * log_softmax(logits)].
        log_p = F.log_softmax(logits, dim=-1)
        policy_loss = -(p_target * log_p).sum(dim=-1).mean()
        value_loss = F.mse_loss(value, v_target)
        loss = policy_loss + value_loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {steps}: "
                f"policy={policy_loss.item()}, value={value_loss.item()}"
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weight-norm tripwire (silent divergence is the AlphaZero footgun).
        for name, p in net.named_parameters():
            n = p.norm().item()
            if n > config.MAX_WEIGHT_NORM:
                raise RuntimeError(f"weight-norm tripwire: {name} L2={n}")

        p_acc += policy_loss.item()
        v_acc += value_loss.item()
        t_acc += loss.item()
        steps += 1

    if steps == 0:
        return TrainStats(0, 0.0, 0.0, 0.0)
    return TrainStats(
        steps=steps,
        policy_loss=p_acc / steps,
        value_loss=v_acc / steps,
        total_loss=t_acc / steps,
    )

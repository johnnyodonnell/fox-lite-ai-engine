"""The policy/value network — a small MLP, mover-frame canonicalized.

Input  : INPUT_SIZE-dim float vector from games.foxlite.encode().
Output : (policy_logits[NUM_CARDS], value scalar in [-1, 1] via tanh).

Layer sizes come from config (HIDDEN_SIZE, TRUNK_LAYERS). forward() returns
*raw* policy logits — softmax and illegal-move masking are done by the caller
(PIMC at inference, the loss function at training time). This keeps the
network identical to the hand-written JS forward pass in src/engine/nn.js,
which the parity check relies on.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from games.foxlite import INPUT_SIZE, NUM_CARDS


class PolicyValueNet(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE, action_size: int = NUM_CARDS):
        super().__init__()
        hidden = config.HIDDEN_SIZE

        trunk: list[nn.Module] = []
        prev = input_size
        for _ in range(config.TRUNK_LAYERS):
            trunk.append(nn.Linear(prev, hidden))
            prev = hidden
        self.trunk = nn.ModuleList(trunk)

        self.policy_head = nn.Linear(hidden, action_size)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [batch, input_size]; returns (policy_logits, value_scalar).

        value_scalar is in (-1, 1) via tanh — matches the bot's signed-margin
        target divided by the max-per-round-margin of 6.
        """
        for layer in self.trunk:
            x = F.relu(layer(x))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x)).squeeze(-1)
        return policy_logits, value


@torch.no_grad()
def infer(net: PolicyValueNet, encoded: list[float]) -> tuple[list[float], float]:
    """Single-state inference returning raw logits + value as Python types.

    The input adopts the network's dtype, so a net moved to float64 (via
    `.double()`, used in parity checks) runs the whole forward pass in
    float64 to match the JS engine.
    """
    dtype = next(net.parameters()).dtype
    device = next(net.parameters()).device
    x = torch.tensor([encoded], dtype=dtype, device=device)
    logits, value = net(x)
    return logits[0].tolist(), float(value[0])

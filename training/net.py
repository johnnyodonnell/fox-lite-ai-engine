"""Residual MLP for Fox-Lite (policy + value heads).

Input is the ~230-dim canonical encoding (see encode.py / encode.js / encode.rs).
Pre-activation residual fully-connected blocks (LayerNorm -> GELU -> Linear),
two heads: policy (33 canonical card logits) and value (scalar in [-1,1] via tanh,
target = match outcome z in {-1,+1}).

State-dict key names are chosen to be loaded directly by the Rust tch forward
(selfplay_rs/src/net.rs): stem, blocks.{i}.{ln1,fc1,ln2,fc2}, policy_ln/policy_fc,
value_ln/value_fc1/value_fc2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encode import INPUT_SIZE, NUM_CARDS

WIDTH = 512
N_BLOCKS = 4
POLICY_SIZE = NUM_CARDS  # 33
VALUE_HIDDEN = 256


class ResBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        h = self.fc1(F.gelu(self.ln1(x)))
        h = self.fc2(F.gelu(self.ln2(h)))
        return x + h


class FoxNet(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, width=WIDTH, n_blocks=N_BLOCKS,
                 policy_size=POLICY_SIZE):
        super().__init__()
        self.stem = nn.Linear(input_size, width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(n_blocks)])
        self.policy_ln = nn.LayerNorm(width)
        self.policy_fc = nn.Linear(width, policy_size)
        self.value_ln = nn.LayerNorm(width)
        self.value_fc1 = nn.Linear(width, VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN, 1)

    def forward(self, x):
        """x: [B, INPUT_SIZE] -> (policy_logits [B, 33], value [B])."""
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        policy = self.policy_fc(self.policy_ln(h))
        v = F.gelu(self.value_fc1(self.value_ln(h)))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return policy, v


def n_params(net) -> int:
    return sum(p.numel() for p in net.parameters())

"""Fox-Lite net v2: transformer history encoder + residual MLP trunk
(policy + value heads).

Input is the flat ~205-dim v2 encoding (see encode.py / encode.js / encode.rs):
26 history tokens of [card index, played-by-self, valid] followed by a 127-dim
static one-hot block. `forward` slices the flat row internally so the trainer,
cohort format, ONNX export, and the Rust pipeline all keep a single [B, 205]
input tensor.

History encoder: token = card embedding + played-by-self embedding + learned
positional embedding; N pre-LN self-attention blocks (explicit q/k/v/o Linears
— no nn.MultiheadAttention, so the safetensors keys stay simple FQNs the Rust
forward can mirror); additive key mask (valid-1)*1e9; final LayerNorm; masked
mean-pool. An empty history (first lead of a round) pools to a zero vector by
construction. The pooled vector is concatenated with the static block and fed
to the same residual MLP trunk as v1.

State-dict key names are chosen to be loaded directly by the Rust tch forward
(selfplay_rs/src/net.rs): hist_embed, hist_self_embed, hist_pos,
hist_layers.{i}.{ln1,q,k,v,o,ln2,fc1,fc2}, hist_ln, stem,
blocks.{i}.{ln1,fc1,ln2,fc2}, policy_ln/policy_fc, value_ln/value_fc1/value_fc2.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from encode import HIST, HIST_TOKENS, INPUT_SIZE, NUM_CARDS, TOKEN_FEATS

STATIC_SIZE = INPUT_SIZE - HIST  # 127

WIDTH = 512
N_BLOCKS = 4
POLICY_SIZE = NUM_CARDS  # 33
VALUE_HIDDEN = 256

D_MODEL = 128
N_LAYERS = 2
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 32
FFN = 256

MASK_NEG = 1.0e9  # same additive-mask constant as train.py's legal masking


class HistLayer(nn.Module):
    """Pre-LN self-attention block over the history tokens."""

    def __init__(self, d=D_MODEL, ffn=FFN):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)

    def forward(self, x, addmask):
        """x: [B, T, d]; addmask: [B, 1, 1, T] additive key-padding mask."""
        h = self.ln1(x)
        q = self.q(h).reshape(-1, HIST_TOKENS, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k(h).reshape(-1, HIST_TOKENS, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v(h).reshape(-1, HIST_TOKENS, N_HEADS, HEAD_DIM).transpose(1, 2)
        att = q.matmul(k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)  # [B,H,T,T]
        att = F.softmax(att + addmask, dim=-1)
        a = att.matmul(v).transpose(1, 2).reshape(-1, HIST_TOKENS, D_MODEL)
        x = x + self.o(a)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


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
    def __init__(self, width=WIDTH, n_blocks=N_BLOCKS, policy_size=POLICY_SIZE,
                 d_model=D_MODEL, n_layers=N_LAYERS):
        super().__init__()
        self.hist_embed = nn.Embedding(NUM_CARDS, d_model)
        self.hist_self_embed = nn.Embedding(2, d_model)
        self.hist_pos = nn.Parameter(torch.zeros(HIST_TOKENS, d_model))
        self.hist_layers = nn.ModuleList([HistLayer(d_model) for _ in range(n_layers)])
        self.hist_ln = nn.LayerNorm(d_model)
        self.stem = nn.Linear(STATIC_SIZE + d_model, width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(n_blocks)])
        self.policy_ln = nn.LayerNorm(width)
        self.policy_fc = nn.Linear(width, policy_size)
        self.value_ln = nn.LayerNorm(width)
        self.value_fc1 = nn.Linear(width, VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN, 1)

    def forward(self, x):
        """x: [B, INPUT_SIZE] -> (policy_logits [B, 33], value [B])."""
        tok = x[:, :HIST].reshape(-1, HIST_TOKENS, TOKEN_FEATS)
        static = x[:, HIST:]
        card = tok[:, :, 0].long()
        self_bit = tok[:, :, 1].long()
        valid = tok[:, :, 2]  # [B, T]

        h = self.hist_embed(card) + self.hist_self_embed(self_bit) + self.hist_pos
        addmask = ((valid - 1.0) * MASK_NEG).reshape(-1, 1, 1, HIST_TOKENS)
        for layer in self.hist_layers:
            h = layer(h, addmask)
        h = self.hist_ln(h)
        vm = valid.unsqueeze(-1)
        pooled = (h * vm).sum(dim=1) / vm.sum(dim=1).clamp(min=1.0)

        t = self.stem(torch.cat([static, pooled], dim=1))
        for block in self.blocks:
            t = block(t)
        policy = self.policy_fc(self.policy_ln(t))
        v = F.gelu(self.value_fc1(self.value_ln(t)))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return policy, v


def n_params(net) -> int:
    return sum(p.numel() for p in net.parameters())

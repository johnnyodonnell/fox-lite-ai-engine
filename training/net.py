"""Fox-Lite net v3: transformer history encoder over trick tokens +
learned-query attention readout + residual MLP trunk (policy + value heads).

Input is the flat 209-dim v3 encoding (see encode.py / encode.js / encode.rs):
12 trick tokens of [lead card index, follow card index, led-by-self, valid]
(slot 0 = most recent completed trick) followed by a 161-dim static one-hot
block. `forward` slices the flat row internally so the trainer, cohort format,
ONNX export, and the Rust pipeline all keep a single [B, 209] input tensor.

History encoder: token = lead-card embedding + follow-card embedding +
led-by-self embedding + learned positional embedding; N pre-LN self-attention
blocks (explicit q/k/v/o Linears — no nn.MultiheadAttention, so the
safetensors keys stay simple FQNs the Rust forward can mirror); additive key
mask (valid-1)*1e9; final LayerNorm; then a pooled summary instead of a bare
mean-pool: masked MEAN (counting) + masked per-dim MAX (ever-happened facts)
+ N_READOUT learned query vectors that softmax-attend over the tokens, all
concatenated. No key/value projections: with learned constant queries a key
projection is absorbed into the query, and a value projection into the stem.
An empty history (no completed tricks yet) pools to a zero vector by
construction (the summary is gated on any-valid). The pooled vectors are
concatenated with the static block and fed to the same residual MLP trunk as
v1/v2. mean_max_pool=False reproduces the readout-only flavor (run5); the
mean+max blocks add no parameters, so the stem width is the only witness —
use foxnet_for_state to rebuild the right flavor from a state dict.

State-dict key names are chosen to be loaded directly by the Rust tch forward
(selfplay_rs/src/net.rs): hist_lead_embed, hist_follow_embed, hist_led_embed,
hist_pos, hist_layers.{i}.{ln1,q,k,v,o,ln2,fc1,fc2}, hist_ln, readout_q, stem,
blocks.{i}.{ln1,fc1,ln2,fc2}, policy_ln/policy_fc, value_ln/value_fc1/value_fc2.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from encode import HIST, HIST_TOKENS, INPUT_SIZE, NUM_CARDS, TOKEN_FEATS

STATIC_SIZE = INPUT_SIZE - HIST  # 161

WIDTH = 512
N_BLOCKS = 4
POLICY_SIZE = NUM_CARDS  # 33
VALUE_HIDDEN = 256

D_MODEL = 128
N_LAYERS = 2
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 32
FFN = 256
N_READOUT = 4

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
                 d_model=D_MODEL, n_layers=N_LAYERS, n_readout=N_READOUT,
                 mean_max_pool=True):
        super().__init__()
        self.mean_max_pool = mean_max_pool
        self.hist_lead_embed = nn.Embedding(NUM_CARDS, d_model)
        self.hist_follow_embed = nn.Embedding(NUM_CARDS, d_model)
        self.hist_led_embed = nn.Embedding(2, d_model)
        self.hist_pos = nn.Parameter(torch.zeros(HIST_TOKENS, d_model))
        self.hist_layers = nn.ModuleList([HistLayer(d_model) for _ in range(n_layers)])
        self.hist_ln = nn.LayerNorm(d_model)
        self.readout_q = nn.Parameter(torch.randn(n_readout, d_model) * 0.02)
        pool_d = (n_readout + (2 if mean_max_pool else 0)) * d_model
        self.stem = nn.Linear(STATIC_SIZE + pool_d, width)
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
        lead = tok[:, :, 0].long()
        follow = tok[:, :, 1].long()
        led_self = tok[:, :, 2].long()
        valid = tok[:, :, 3]  # [B, T]

        h = (self.hist_lead_embed(lead) + self.hist_follow_embed(follow)
             + self.hist_led_embed(led_self) + self.hist_pos)
        addmask = ((valid - 1.0) * MASK_NEG).reshape(-1, 1, 1, HIST_TOKENS)
        for layer in self.hist_layers:
            h = layer(h, addmask)
        h = self.hist_ln(h)

        # Attention readout: scores [B,T,Q], softmax over tokens, pooled [B,Q,d].
        scores = h.matmul(self.readout_q.t()) / math.sqrt(D_MODEL)
        scores = scores + ((valid - 1.0) * MASK_NEG).unsqueeze(-1)
        att = F.softmax(scores, dim=1)
        pooled = att.transpose(1, 2).matmul(h).flatten(1)  # [B, Q*d]
        if self.mean_max_pool:
            vm = valid.unsqueeze(-1)  # [B,T,1]
            mean = (h * vm).sum(dim=1) / vm.sum(dim=1).clamp(min=1.0)
            mx = (h + (vm - 1.0) * MASK_NEG).amax(dim=1)
            pooled = torch.cat([mean, mx, pooled], dim=1)
        # Empty history: softmax over all-masked slots is uniform over padding
        # (and the masked max bottoms out at -MASK_NEG), so gate the summary to
        # an exact zero vector when no token is valid.
        pooled = pooled * valid.amax(dim=1, keepdim=True)

        t = self.stem(torch.cat([static, pooled], dim=1))
        for block in self.blocks:
            t = block(t)
        policy = self.policy_fc(self.policy_ln(t))
        v = F.gelu(self.value_fc1(self.value_ln(t)))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return policy, v


def foxnet_for_state(sd) -> FoxNet:
    """Build a FoxNet matching a state dict's pooling flavor and load it.

    mean+max pooling adds no parameters, so the stem input width is the only
    way to tell the flavors apart. Used by resume/promote so checkpoints from
    readout-only runs (run5) and mean+max runs both load under current code.
    """
    d_model = sd["hist_lead_embed.weight"].shape[1]
    n_readout = sd["readout_q"].shape[0]
    stem_in = sd["stem.weight"].shape[1]
    mean_max = stem_in == STATIC_SIZE + (n_readout + 2) * d_model
    if not mean_max and stem_in != STATIC_SIZE + n_readout * d_model:
        raise ValueError(f"stem width {stem_in} matches neither pooling flavor")
    net = FoxNet(n_readout=n_readout, mean_max_pool=mean_max)
    net.load_state_dict(sd)
    return net


def n_params(net) -> int:
    return sum(p.numel() for p in net.parameters())

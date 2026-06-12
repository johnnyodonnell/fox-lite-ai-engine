"""Eager vs torch.compile timing of the train_on_cohort SGD step.

Mirrors train.py's step exactly (REINFORCE loss, grad clip, AdamW) at the
production --sgd-batch, plus one ragged tail batch so the compiled run pays
the same dynamic-shape recompile it would see in the orchestrator loop.
Run on an idle GPU; prints ms/step for each mode and the speedup.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from encode import HIST, HIST_TOKENS, INPUT_SIZE, NUM_CARDS, TOKEN_FEATS
from net import FoxNet

MASK_NEG = 1.0e9


def make_batch(n, rng, device):
    tok = np.zeros((n, HIST_TOKENS, TOKEN_FEATS), dtype=np.float32)
    tok[:, :, 0] = rng.integers(0, NUM_CARDS, (n, HIST_TOKENS))
    tok[:, :, 1] = rng.integers(0, 2, (n, HIST_TOKENS))
    tok[:, :, 2] = rng.integers(0, 2, (n, HIST_TOKENS))
    static = rng.random((n, INPUT_SIZE - HIST), dtype=np.float32)
    x = np.concatenate([tok.reshape(n, HIST), static], axis=1)
    masks = (rng.random((n, NUM_CARDS)) < 0.3).astype(np.float32)
    masks[:, 0] = 1.0  # keep at least one legal move so action 0 is valid
    actions = np.zeros(n, dtype=np.int64)
    z = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), n)
    return (torch.from_numpy(x).to(device), torch.from_numpy(masks).to(device),
            torch.from_numpy(actions).to(device), torch.from_numpy(z).to(device))


def step(net, opt, s, m, a, zz, c_entropy):
    logits, v = net(s)
    masked = logits + (m - 1.0) * MASK_NEG
    logp = F.log_softmax(masked, dim=1)
    logp_a = logp.gather(1, a[:, None]).squeeze(1)
    adv = zz - v.detach()
    policy_loss = -(adv * logp_a).mean()
    value_loss = F.mse_loss(v, zz)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=1).mean()
    loss = policy_loss + value_loss - c_entropy * entropy
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()


def bench(label, net, args, device):
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    batches = [make_batch(args.batch, rng, device) for _ in range(2)]
    batches.append(make_batch(args.tail, rng, device))
    net.train()
    t0 = time.time()
    for i in range(args.warmup):
        step(net, opt, *batches[i % len(batches)], args.c_entropy)
    torch.cuda.synchronize()
    print(f"{label}: warmup {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    for i in range(args.steps):
        step(net, opt, *batches[i % len(batches)], args.c_entropy)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.steps
    print(f"{label}: {dt * 1000:.0f} ms/step "
          f"({args.steps} steps, batch {args.batch})", flush=True)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--tail", type=int, default=62466, help="ragged last-minibatch size")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--c-entropy", type=float, default=0.01)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)
    eager = bench("eager", FoxNet().to(device), args, device)
    torch.manual_seed(0)
    compiled = bench("compiled", torch.compile(FoxNet().to(device)), args, device)
    print(f"speedup: {eager / compiled:.2f}x")


if __name__ == "__main__":
    main()

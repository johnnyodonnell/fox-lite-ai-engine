"""Export FoxNet to an AOTInductor .pt2 package (fused, Python-free forward).

The .pt2 bakes in the forward graph + the weights from `weights` at export time.
The self-play worker loads it ONCE; fresh weights then arrive via the safetensors
sidecar and are pushed in place with update_constant_buffer + run_const_fold
(no recompile) -- which requires use_runtime_constant_folding=True here.

Run with the training venv:
    .venv/bin/python export_pt2.py [weights.safetensors] [out.pt2] [batch]
"""
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, "..")
from net import FoxNet, SERVING_BATCH  # noqa: E402
from encode import INPUT_SIZE  # noqa: E402

W = sys.argv[1] if len(sys.argv) > 1 else "../../runs/run3/serving_weights.safetensors"
OUT = sys.argv[2] if len(sys.argv) > 2 else "serving_model.pt2"
# Default to the production batch (must match SERVING_BATCH / pipeline Config::batch);
# the argv override exists only for ad-hoc spikes.
B = int(sys.argv[3]) if len(sys.argv) > 3 else SERVING_BATCH

net = FoxNet()
net.load_state_dict(load_file(W))
net.eval().to("cuda").bfloat16()

x = torch.zeros(B, INPUT_SIZE, device="cuda", dtype=torch.bfloat16)
# Keep folded constants runtime-updatable so the worker can hot-swap raw weights
# via update_constant_buffer + run_const_fold (no recompile).
import torch._inductor  # noqa: E402

torch._inductor.config.aot_inductor.use_runtime_constant_folding = True
with torch.no_grad():
    ep = torch.export.export(net, (x,))
    path = torch._inductor.aoti_compile_and_package(ep, package_path=OUT)
print(f"wrote {path} (batch={B}, input={INPUT_SIZE})")

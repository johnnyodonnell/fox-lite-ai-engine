"""De-risk spike: export FoxNet -> .pt2, load it back, and compare the fused AOTI
forward against the eager bf16 FoxNet on random input. Pure Python (no Rust) -- it
validates that AOTInductor export + the fused forward are numerically sound on this
torch/GPU before we wire the C++ shim. Run with the training venv:

    .venv/bin/python aoti_spike.py [weights.safetensors] [batch]
"""
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, "..")
from net import FoxNet  # noqa: E402
from encode import INPUT_SIZE  # noqa: E402

W = sys.argv[1] if len(sys.argv) > 1 else "../../runs/run3/serving_weights.safetensors"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 64

sd = load_file(W)
# fp32 reference (ground truth) and the eager bf16 path we already run in prod.
net_fp32 = FoxNet().eval().to("cuda")
net_fp32.load_state_dict(sd)
net_bf16 = FoxNet().eval().to("cuda").bfloat16()
net_bf16.load_state_dict(sd)

x32 = torch.randn(B, INPUT_SIZE, device="cuda")
x16 = x32.bfloat16()
with torch.no_grad():
    ref_l, ref_v = net_fp32(x32)
    eager_l, eager_v = net_bf16(x16)

# Export at this spike batch and load the package back (the runtime path the worker
# will use, minus the C++ shim).
import torch._inductor  # noqa: E402

torch._inductor.config.aot_inductor.use_runtime_constant_folding = True
with torch.no_grad():
    ep = torch.export.export(net_bf16, (x16,))
    pkg = torch._inductor.aoti_compile_and_package(ep, package_path="/tmp/fox_spike.pt2")
runner = torch._inductor.aoti_load_package(pkg)
with torch.no_grad():
    aoti_l, aoti_v = runner(x16)


def md(a, b):
    return (a.float() - b.float()).abs().max().item()


aoti_vs_fp32 = (md(aoti_l, ref_l), md(aoti_v, ref_v))
eager_vs_fp32 = (md(eager_l, ref_l), md(eager_v, ref_v))
print(f"batch={B} input={INPUT_SIZE}  (random N(0,1) input -- stress test)")
print(f"  shapes: logits={tuple(aoti_l.shape)}  value={tuple(aoti_v.shape)}")
print(f"  eager bf16 vs fp32:  max|dlogits|={eager_vs_fp32[0]:.3e}  max|dvalue|={eager_vs_fp32[1]:.3e}")
print(f"  AOTI  bf16 vs fp32:  max|dlogits|={aoti_vs_fp32[0]:.3e}  max|dvalue|={aoti_vs_fp32[1]:.3e}")
print(f"  AOTI  vs eager bf16: max|dlogits|={md(aoti_l, eager_l):.3e}  max|dvalue|={md(aoti_v, eager_v):.3e}")
# Gate: AOTI must be no meaningfully worse than the eager bf16 path already in prod
# (same dtype; only fusion/accumulation-order differs), and the right shapes.
shapes_ok = tuple(aoti_l.shape) == (B, 33) and tuple(aoti_v.shape) == (B,)
margin = 0.05
ok = (
    shapes_ok
    and aoti_vs_fp32[0] <= eager_vs_fp32[0] + margin
    and aoti_vs_fp32[1] <= eager_vs_fp32[1] + margin
)
print("SPIKE OK" if ok else "SPIKE FAILED")
sys.exit(0 if ok else 1)

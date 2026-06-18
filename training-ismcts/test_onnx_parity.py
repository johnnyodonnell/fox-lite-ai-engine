"""Parity test: the exported ONNX (what the browser runs via onnxruntime-web)
must match the PyTorch reference outputs from the forward fixture.

Run (project venv):
  python training/make_forward_fixture.py   # writes fixtures/fwd_model.onnx + fwd_fixture.safetensors
  python training/test_onnx_parity.py
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from safetensors.numpy import load_file

FIX = Path(__file__).resolve().parent / "fixtures"


def main() -> int:
    fx = load_file(str(FIX / "fwd_fixture.safetensors"))
    x = fx["input"].astype(np.float32)
    ref_logits = fx["ref_logits"].astype(np.float32)
    ref_value = fx["ref_value"].astype(np.float32)
    ref_belief = fx["ref_belief"].astype(np.float32)

    sess = ort.InferenceSession(str(FIX / "fwd_model.onnx"),
                                providers=["CPUExecutionProvider"])
    out = sess.run(["policy", "value", "belief"], {"input": x})
    policy, value, belief = out[0], out[1].reshape(-1), out[2]

    dl = float(np.abs(policy - ref_logits).max())
    dv = float(np.abs(value - ref_value).max())
    db = float(np.abs(belief - ref_belief).max())
    print(f"ONNX vs PyTorch on {x.shape[0]} positions: "
          f"max|Δlogits|={dl:.3e} max|Δvalue|={dv:.3e} max|Δbelief|={db:.3e}")
    ok = dl < 1e-4 and dv < 1e-4 and db < 1e-4
    print("ONNX-PARITY OK" if ok else "ONNX-PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

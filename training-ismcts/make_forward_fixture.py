"""Build the forward-parity fixture for Phase 3.

Encodes the first N states from the encoder fixture, runs a freshly-initialized
FoxNet in fp32, and writes:
  fixtures/fwd_weights.safetensors   - net params (Rust tch loads these)
  fixtures/fwd_fixture.safetensors    - input[N,INPUT_SIZE], ref_logits[N,33], ref_value[N]
  fixtures/fwd_model.onnx             - ONNX export (browser parity, checked in Phase 7)

Run (on asus-nvidia, in the project venv):
  python training/make_forward_fixture.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from encode import encode
from export import export_onnx, save_weights_st
from net import FoxNet, n_params

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"


def main(n: int = 512, seed: int = 0):
    enc_fix = json.loads((FIX / "encode_fixtures.json").read_text())
    cases = enc_fix["cases"][:n]
    inputs = np.stack([encode(c["state"], c["mover"]) for c in cases]).astype(np.float32)

    torch.manual_seed(seed)
    net = FoxNet().eval()
    print(f"net params: {n_params(net):,}")

    x = torch.from_numpy(inputs)
    with torch.no_grad():
        logits, value, belief = net(x)

    save_weights_st(net, str(FIX / "fwd_weights.safetensors"))
    save_file(
        {
            "input": x.contiguous(),
            "ref_logits": logits.float().contiguous(),
            "ref_value": value.float().contiguous(),
            "ref_belief": belief.float().contiguous(),
        },
        str(FIX / "fwd_fixture.safetensors"),
    )
    export_onnx(net, str(FIX / "fwd_model.onnx"))
    print(f"wrote fixture for {len(cases)} states -> {FIX}")


if __name__ == "__main__":
    main()

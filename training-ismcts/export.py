"""ONNX export + safetensors weight I/O for the Fox-Lite net.

- export_onnx: write a browser-loadable ONNX (named inputs/outputs, dynamic batch).
- save_weights_st / load_weights_st: fp32 safetensors keyed by state_dict FQN, the
  format the Rust self-play worker (selfplay_rs) and evaluator load.
"""

import torch

from encode import INPUT_SIZE


def export_onnx(net, path: str, input_size: int = INPUT_SIZE):
    """Export `net` to a self-contained ONNX at `path` (CPU fp32, dynamic batch).

    The torch dynamo exporter externalizes weights into a sibling `.data` file;
    the browser needs a single file, so we reload and re-save with weights
    embedded, then remove the stray external-data file.
    """
    import copy
    import os

    import onnx

    ex = copy.deepcopy(net).to("cpu").float().eval()
    dummy = torch.zeros(1, input_size, dtype=torch.float32)
    torch.onnx.export(
        ex,
        dummy,
        path,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )
    # Embed weights into a single self-contained file for the browser.
    model = onnx.load(path)  # resolves any sibling external data
    onnx.save_model(model, path, save_as_external_data=False)
    data_file = path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)


def save_weights_st(net, path: str):
    """Save params as fp32 safetensors keyed by state_dict FQN (Rust reads these)."""
    from safetensors.torch import save_file

    sd = {k: v.detach().to("cpu", torch.float32).contiguous()
          for k, v in net.state_dict().items()}
    save_file(sd, path)


def load_weights_st(net, path: str, device="cpu"):
    from safetensors.torch import load_file

    sd = load_file(path, device=str(device))
    net.load_state_dict(sd)
    return net

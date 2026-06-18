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
    # Batch-2 dummy + dynamic_shapes: the dynamo exporter specializes size-1
    # dims, which bakes static batch reshapes into the attention layers.
    dummy = torch.zeros(2, input_size, dtype=torch.float32)
    # The net returns three heads (policy, value, belief); the browser single-forward
    # path only consumes policy+value, but the ONNX must name every output it emits.
    torch.onnx.export(
        ex,
        (dummy,),
        path,
        input_names=["input"],
        output_names=["policy", "value", "belief"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
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

import torch
import torch.nn as nn
import onnx
from typing import Tuple


def export_model_to_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...],
    opset_version: int = 14,
    export_params: bool = True,
    do_constant_folding: bool = True,
    verbose: bool = False,
    use_camera: bool = False,
    use_view: bool = False,
) -> str:
    """
    Export a PyTorch model to ONNX format.
    """
    input_names = ["input"]
    if use_camera:
        input_names.append("camera_label")
    if use_view:
        input_names.append("view_label")
    output_names = ["features"]

    dynamic_axes = {
        "input": {0: "batch_size"},
        "features": {0: "batch_size"}
    }
    if use_camera:
        dynamic_axes["camera_label"] = {0: "batch_size"}
    if use_view:
        dynamic_axes["view_label"] = {0: "batch_size"}
    
    cam_label = torch.zeros(1, dtype=torch.int64) if use_camera else None
    view_label = torch.zeros(1, dtype=torch.int64) if use_view else None
    
    dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)

    model.eval()
    if cam_label is not None and view_label is not None:
        torch.onnx.export(
            model,
            (dummy_input, cam_label, view_label),
            save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            export_params=export_params,
            do_constant_folding=do_constant_folding,
            verbose=verbose
        )
    elif cam_label is not None:
        torch.onnx.export(
            model,
            (dummy_input, cam_label),
            save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            export_params=export_params,
            do_constant_folding=do_constant_folding,
            verbose=verbose
        )
    elif view_label is not None:
        torch.onnx.export(
            model,
            (dummy_input, view_label),
            save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            export_params=export_params,
            do_constant_folding=do_constant_folding,
            verbose=verbose
        )
    else:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            export_params=export_params,
            do_constant_folding=do_constant_folding,
            verbose=verbose
        )
    
    if verbose:
        print(f"Model exported to ONNX format at: {save_path}")

        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model checked successfully!")

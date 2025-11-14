"""
Convert PyTorch models to TensorRT for optimized inference

Utilizes tensor cores for maximum throughput
"""

import torch
import tensorrt as trt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import onnx


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def convert_to_onnx(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    input_names: List[str] = ['input'],
    output_names: List[str] = ['output'],
    dynamic_axes: Optional[dict] = None,
    opset_version: int = 14
) -> str:
    """
    Convert PyTorch model to ONNX format

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)
        output_path: Path to save ONNX model
        input_names: Names of input tensors
        output_names: Names of output tensors
        dynamic_axes: Dynamic axes for variable-size inputs
        opset_version: ONNX opset version

    Returns:
        Path to ONNX model
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape, device='cuda')

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"ONNX model exported to {output_path}")
    return output_path


def convert_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    fp16_mode: bool = True,
    max_batch_size: int = 1,
    max_workspace_size: int = 1 << 30,  # 1GB
    input_shape: Optional[Tuple[int, ...]] = None,
    dynamic_shapes: bool = False
) -> str:
    """
    Convert ONNX model to TensorRT engine

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        fp16_mode: Enable FP16 precision (tensor core acceleration)
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace size in bytes
        input_shape: Input shape for optimization (optional)
        dynamic_shapes: Enable dynamic shape support

    Returns:
        Path to TensorRT engine
    """
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print('Failed to parse ONNX model')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError('ONNX parse failed')

    # Create builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    # Enable FP16 mode for tensor cores
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled - tensor cores will be utilized")
    else:
        print("FP16 mode not available or disabled")

    # Enable optimization profiles for dynamic shapes
    if dynamic_shapes and input_shape:
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name

        # Set min, optimal, and max shapes
        min_shape = (1, *input_shape)
        opt_shape = (max_batch_size, *input_shape)
        max_shape = (max_batch_size, *input_shape)

        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    # Build engine
    print("Building TensorRT engine... This may take a while.")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError('Failed to build TensorRT engine')

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {engine_path}")
    return engine_path


def pytorch_to_tensorrt(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    output_dir: str,
    model_name: str = 'model',
    fp16_mode: bool = True,
    max_batch_size: int = 1,
    dynamic_shapes: bool = False
) -> Tuple[str, str]:
    """
    Convert PyTorch model directly to TensorRT

    Args:
        model: PyTorch model
        input_shape: Input shape (without batch dimension)
        output_dir: Directory to save outputs
        model_name: Base name for output files
        fp16_mode: Enable FP16 precision
        max_batch_size: Maximum batch size
        dynamic_shapes: Enable dynamic shape support

    Returns:
        Tuple of (onnx_path, engine_path)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    onnx_path = f"{output_dir}/{model_name}.onnx"
    engine_path = f"{output_dir}/{model_name}.engine"

    # Dynamic axes for variable sequence length (if needed)
    dynamic_axes = None
    if dynamic_shapes:
        dynamic_axes = {
            'input': {0: 'batch', 1: 'sequence'},
            'output': {0: 'batch', 1: 'sequence'}
        }

    # Convert to ONNX
    convert_to_onnx(
        model,
        input_shape,
        onnx_path,
        dynamic_axes=dynamic_axes
    )

    # Convert to TensorRT
    convert_to_tensorrt(
        onnx_path,
        engine_path,
        fp16_mode=fp16_mode,
        max_batch_size=max_batch_size,
        input_shape=input_shape,
        dynamic_shapes=dynamic_shapes
    )

    return onnx_path, engine_path


def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize PyTorch model for inference

    Args:
        model: PyTorch model

    Returns:
        Optimized model
    """
    model.eval()

    # Fuse common operations
    model = torch.jit.script(model)
    model = torch.jit.freeze(model)

    return model

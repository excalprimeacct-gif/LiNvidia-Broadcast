#!/usr/bin/env python3
"""
Export models to TensorRT for optimized inference

Converts PyTorch models to ONNX and then to TensorRT engines with FP16 precision
for maximum performance on tensor cores.
"""

import sys
import argparse
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from linvidia.models.noise_suppression import NoiseSuppressionRNN
from linvidia.models.segmentation import MobileNetV3Segmentation
from linvidia.inference.converter import pytorch_to_tensorrt


def export_noise_suppression(
    output_dir: str,
    freq_bins: int = 257,
    hidden_size: int = 256,
    num_layers: int = 2,
    fp16_mode: bool = True,
    checkpoint_path: str = None
):
    """
    Export noise suppression model to TensorRT

    Args:
        output_dir: Directory to save outputs
        freq_bins: Number of frequency bins
        hidden_size: Hidden size of RNN
        num_layers: Number of RNN layers
        fp16_mode: Enable FP16 precision
        checkpoint_path: Optional checkpoint to load
    """
    print("=" * 60)
    print("Exporting Noise Suppression Model to TensorRT")
    print("=" * 60)

    # Create model
    model = NoiseSuppressionRNN(
        input_size=freq_bins,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).cuda().eval()

    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No checkpoint provided, exporting untrained model")

    # Input shape: (freq_bins,) - single frame
    input_shape = (freq_bins,)

    # Convert to TensorRT
    onnx_path, engine_path = pytorch_to_tensorrt(
        model=model,
        input_shape=input_shape,
        output_dir=output_dir,
        model_name='noise_suppression',
        fp16_mode=fp16_mode,
        max_batch_size=1,
        dynamic_shapes=False
    )

    print("\n✓ Noise Suppression Model Export Complete!")
    print(f"  ONNX:   {onnx_path}")
    print(f"  Engine: {engine_path}")
    print(f"  FP16:   {fp16_mode}")

    return onnx_path, engine_path


def export_segmentation(
    output_dir: str,
    input_size: tuple = (256, 256),
    fp16_mode: bool = True,
    checkpoint_path: str = None
):
    """
    Export background segmentation model to TensorRT

    Args:
        output_dir: Directory to save outputs
        input_size: Input image size (H, W)
        fp16_mode: Enable FP16 precision
        checkpoint_path: Optional checkpoint to load
    """
    print("=" * 60)
    print("Exporting Background Segmentation Model to TensorRT")
    print("=" * 60)

    # Create model
    model = MobileNetV3Segmentation(
        input_size=input_size,
        pretrained=True
    ).cuda().eval()

    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Using pretrained ImageNet weights")

    # Input shape: (3, H, W) - RGB image
    input_shape = (3, *input_size)

    # Convert to TensorRT
    onnx_path, engine_path = pytorch_to_tensorrt(
        model=model,
        input_shape=input_shape,
        output_dir=output_dir,
        model_name='segmentation',
        fp16_mode=fp16_mode,
        max_batch_size=1,
        dynamic_shapes=False
    )

    print("\n✓ Segmentation Model Export Complete!")
    print(f"  ONNX:   {onnx_path}")
    print(f"  Engine: {engine_path}")
    print(f"  FP16:   {fp16_mode}")

    return onnx_path, engine_path


def main():
    parser = argparse.ArgumentParser(
        description='Export LiNvidia Broadcast models to TensorRT'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['noise', 'segmentation', 'both'],
        help='Model to export'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models/tensorrt',
        help='Output directory for TensorRT engines (default: ./models/tensorrt)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (optional)'
    )

    parser.add_argument(
        '--no-fp16',
        action='store_true',
        help='Disable FP16 precision (not recommended for RTX GPUs)'
    )

    parser.add_argument(
        '--freq-bins',
        type=int,
        default=257,
        help='Number of frequency bins for noise suppression (default: 257)'
    )

    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=('HEIGHT', 'WIDTH'),
        help='Input size for segmentation model (default: 256 256)'
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    fp16_mode = not args.no_fp16

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        print("TensorRT export requires CUDA")
        sys.exit(1)

    # Print GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"FP16 Mode: {fp16_mode}\n")

    try:
        # Export models
        if args.model in ['noise', 'both']:
            export_noise_suppression(
                output_dir=args.output_dir,
                freq_bins=args.freq_bins,
                fp16_mode=fp16_mode,
                checkpoint_path=args.checkpoint
            )

        if args.model in ['segmentation', 'both']:
            export_segmentation(
                output_dir=args.output_dir,
                input_size=tuple(args.input_size),
                fp16_mode=fp16_mode,
                checkpoint_path=args.checkpoint
            )

        print("\n" + "=" * 60)
        print("✓ All exports completed successfully!")
        print("=" * 60)
        print(f"\nEngines saved in: {args.output_dir}")
        print("\nTo use these engines, update your config to point to the .engine files")

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

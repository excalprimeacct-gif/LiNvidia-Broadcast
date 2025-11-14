#!/usr/bin/env python3
"""
Download pre-trained noise suppression models

This script will download pre-trained models when they become available.
For now, it creates an untrained model that can be fine-tuned.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from linvidia.models import create_noise_suppression_model
from linvidia.inference.converter import pytorch_to_tensorrt


def create_initial_model(output_dir: str, freq_bins: int = 257):
    """
    Create initial (untrained) model

    Args:
        output_dir: Directory to save model
        freq_bins: Number of frequency bins
    """
    print("Creating initial noise suppression model...")

    # Create model
    model = create_noise_suppression_model(
        freq_bins=freq_bins,
        hidden_size=256,
        num_layers=2,
        device='cuda',
        use_fp16=True
    )

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = f"{output_dir}/noise_suppression_initial.pth"
    model.save(model_path)
    print(f"✓ Model saved to {model_path}")

    # Convert to TensorRT
    print("\nConverting to TensorRT for tensor core acceleration...")
    try:
        onnx_path, engine_path = pytorch_to_tensorrt(
            model.model,
            input_shape=(freq_bins,),
            output_dir=output_dir,
            model_name='noise_suppression',
            fp16_mode=True,
            max_batch_size=1
        )
        print(f"✓ ONNX model: {onnx_path}")
        print(f"✓ TensorRT engine: {engine_path}")
    except Exception as e:
        print(f"Warning: TensorRT conversion failed: {e}")
        print("You can still use the PyTorch model.")

    print("\n✓ Model setup complete!")
    print("\nNote: This is an untrained model. For best results, train on your dataset.")
    print("See scripts/train_noise_suppressor.py for training instructions.")


def main():
    parser = argparse.ArgumentParser(description='Download or create noise suppression models')
    parser.add_argument('--output-dir', default='models', help='Output directory')
    parser.add_argument('--freq-bins', type=int, default=257, help='Number of frequency bins')
    args = parser.parse_args()

    try:
        create_initial_model(args.output_dir, args.freq_bins)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

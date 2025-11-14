#!/usr/bin/env python3
"""
TensorRT Segmentation Example

Demonstrates how to use TensorRT-accelerated background segmentation
for 2-3x faster inference compared to PyTorch.
"""

import numpy as np
import cv2
from pathlib import Path

from linvidia.models.segmentation import create_segmentation_model
from linvidia.background_effects import create_background_effects


def example_1_direct_tensorrt_engine():
    """Example 1: Using TensorRT segmentation engine directly"""
    print("=== Example 1: Direct TensorRT Engine Usage ===\n")

    # Create TensorRT-based segmentation model
    # First, ensure you've exported the model with:
    #   linvidia export-tensorrt --model segmentation

    engine_path = "models/tensorrt/segmentation.engine"

    if not Path(engine_path).exists():
        print(f"Error: TensorRT engine not found at {engine_path}")
        print("Please run: linvidia export-tensorrt --model segmentation")
        return

    # Create segmentation model with TensorRT backend
    segmentation_model = create_segmentation_model(
        use_tensorrt=True,
        tensorrt_engine_path=engine_path,
        input_size=(256, 256)
    )

    # Load test image
    image = cv2.imread("test_image.jpg")
    if image is None:
        print("Error: test_image.jpg not found")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Segment person from background
    mask = segmentation_model.segment(image, smooth=True)

    print(f"Input shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Average latency: {segmentation_model.get_latency():.3f} ms")

    # Visualize mask
    mask_vis = (mask * 255).astype(np.uint8)
    cv2.imwrite("mask_tensorrt.png", mask_vis)
    print("Saved mask to: mask_tensorrt.png\n")


def example_2_background_effects_with_tensorrt():
    """Example 2: Using TensorRT with background effects pipeline"""
    print("=== Example 2: Background Effects with TensorRT ===\n")

    engine_path = "models/tensorrt/segmentation.engine"

    if not Path(engine_path).exists():
        print(f"Error: TensorRT engine not found at {engine_path}")
        print("Please run: linvidia export-tensorrt --model segmentation")
        return

    # Create background effects system with TensorRT acceleration
    bg_effects = create_background_effects(
        camera_id=0,
        output_type='window',
        effect='blur',
        blur_strength=35,
        use_tensorrt=True,
        tensorrt_engine_path=engine_path
    )

    print("Starting background effects with TensorRT acceleration...")
    print("Press 'q' to quit\n")

    # Run for 30 seconds with FPS display
    bg_effects.run(duration=30, show_fps=True)


def example_3_performance_comparison():
    """Example 3: Compare PyTorch vs TensorRT performance"""
    print("=== Example 3: Performance Comparison ===\n")

    engine_path = "models/tensorrt/segmentation.engine"

    if not Path(engine_path).exists():
        print(f"Error: TensorRT engine not found at {engine_path}")
        print("Please run: linvidia export-tensorrt --model segmentation")
        return

    # Load test image
    image = cv2.imread("test_image.jpg")
    if image is None:
        print("Error: test_image.jpg not found")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Test PyTorch model
    print("Testing PyTorch model (FP16)...")
    pytorch_model = create_segmentation_model(
        model_type='mobilenet',
        device='cuda',
        use_fp16=True,
        input_size=(256, 256)
    )

    # Warmup
    for _ in range(10):
        _ = pytorch_model.segment(image)

    # Benchmark
    pytorch_model.reset_stats()
    for _ in range(100):
        _ = pytorch_model.segment(image)

    pytorch_latency = pytorch_model.get_latency()
    print(f"PyTorch FP16 latency: {pytorch_latency:.3f} ms")

    # Test TensorRT model
    print("\nTesting TensorRT model (FP16)...")
    tensorrt_model = create_segmentation_model(
        use_tensorrt=True,
        tensorrt_engine_path=engine_path,
        input_size=(256, 256)
    )

    # Warmup
    for _ in range(10):
        _ = tensorrt_model.segment(image)

    # Benchmark
    tensorrt_model.engine.reset_stats()
    for _ in range(100):
        _ = tensorrt_model.segment(image)

    tensorrt_latency = tensorrt_model.get_latency()
    print(f"TensorRT FP16 latency: {tensorrt_latency:.3f} ms")

    # Calculate speedup
    speedup = pytorch_latency / tensorrt_latency
    print(f"\nSpeedup: {speedup:.2f}x faster with TensorRT")
    print(f"FPS improvement: {1000/pytorch_latency:.1f} -> {1000/tensorrt_latency:.1f} FPS\n")


def example_4_custom_postprocessing():
    """Example 4: Custom postprocessing with TensorRT"""
    print("=== Example 4: Custom Postprocessing ===\n")

    engine_path = "models/tensorrt/segmentation.engine"

    if not Path(engine_path).exists():
        print(f"Error: TensorRT engine not found at {engine_path}")
        print("Please run: linvidia export-tensorrt --model segmentation")
        return

    # Create segmentation model
    segmentation_model = create_segmentation_model(
        use_tensorrt=True,
        tensorrt_engine_path=engine_path,
        input_size=(256, 256)
    )

    # Load test image
    image = cv2.imread("test_image.jpg")
    if image is None:
        print("Error: test_image.jpg not found")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get soft mask (default)
    soft_mask = segmentation_model.segment(
        image,
        smooth=True,
        return_binary=False
    )

    # Get binary mask
    binary_mask = segmentation_model.segment(
        image,
        smooth=False,
        return_binary=True,
        threshold=0.5
    )

    # Get sharp mask (no smoothing)
    sharp_mask = segmentation_model.segment(
        image,
        smooth=False,
        return_binary=False
    )

    # Save visualizations
    cv2.imwrite("mask_soft.png", (soft_mask * 255).astype(np.uint8))
    cv2.imwrite("mask_binary.png", (binary_mask * 255).astype(np.uint8))
    cv2.imwrite("mask_sharp.png", (sharp_mask * 255).astype(np.uint8))

    print("Saved masks:")
    print("  - mask_soft.png (default, smooth edges)")
    print("  - mask_binary.png (hard threshold at 0.5)")
    print("  - mask_sharp.png (no edge smoothing)\n")


def example_5_cli_usage():
    """Example 5: Using TensorRT via CLI"""
    print("=== Example 5: CLI Usage ===\n")

    print("Using TensorRT with the CLI:\n")

    print("1. Export segmentation model to TensorRT:")
    print("   linvidia export-tensorrt --model segmentation\n")

    print("2. Run background effects with TensorRT:")
    print("   linvidia background-effects --effect blur --tensorrt\n")

    print("3. Use custom TensorRT engine:")
    print("   linvidia background-effects --effect blur \\")
    print("     --tensorrt \\")
    print("     --tensorrt-engine ./my_custom_engine.engine\n")

    print("4. Benchmark TensorRT performance:")
    print("   linvidia benchmark-tensorrt \\")
    print("     --engine models/tensorrt/segmentation.engine \\")
    print("     --model-type segmentation \\")
    print("     --compare-pytorch\n")

    print("5. Virtual camera with TensorRT:")
    print("   linvidia background-effects \\")
    print("     --effect blur \\")
    print("     --output-type virtual \\")
    print("     --virtual-device /dev/video2 \\")
    print("     --tensorrt\n")


if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("TensorRT Segmentation Examples")
    print("=" * 60)
    print()

    examples = {
        '1': ('Direct TensorRT Engine Usage', example_1_direct_tensorrt_engine),
        '2': ('Background Effects with TensorRT', example_2_background_effects_with_tensorrt),
        '3': ('Performance Comparison', example_3_performance_comparison),
        '4': ('Custom Postprocessing', example_4_custom_postprocessing),
        '5': ('CLI Usage', example_5_cli_usage),
    }

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            func()
        else:
            print(f"Error: Unknown example '{example_num}'")
            print(f"Available examples: {', '.join(examples.keys())}")
    else:
        print("Available examples:")
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")
        print()
        print("Usage: python tensorrt_segmentation_example.py <example_number>")
        print("Example: python tensorrt_segmentation_example.py 1")

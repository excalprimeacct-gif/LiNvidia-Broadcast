#!/usr/bin/env python3
"""
Benchmark TensorRT engines for performance testing

Measures latency, throughput, and FPS for TensorRT inference
"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from linvidia.inference.engine import TensorRTEngine, TensorRTNoiseSuppressionEngine


def benchmark_engine(
    engine_path: str,
    input_shape: tuple,
    warmup_runs: int = 10,
    test_runs: int = 100,
    name: str = "Model"
):
    """
    Benchmark a TensorRT engine

    Args:
        engine_path: Path to TensorRT engine
        input_shape: Input shape (without batch dimension)
        warmup_runs: Number of warmup runs
        test_runs: Number of test runs
        name: Model name for display
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {name}")
    print(f"{'=' * 60}")
    print(f"Engine: {engine_path}")
    print(f"Input shape: {input_shape}")

    # Load engine
    print("\nLoading TensorRT engine...")
    engine = TensorRTEngine(engine_path, use_cuda_stream=True)

    print(f"Input shape:  {engine.get_input_shape()}")
    print(f"Output shape: {engine.get_output_shape()}")

    # Create random input
    input_data = np.random.randn(1, *input_shape).astype(np.float32)

    # Warmup
    print(f"\nWarming up ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        _ = engine.infer(input_data)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{warmup_runs} warmup runs completed")

    # Reset stats after warmup
    engine.reset_stats()

    # Benchmark
    print(f"\nRunning benchmark ({test_runs} runs)...")
    latencies = []

    for i in range(test_runs):
        start_time = time.perf_counter()
        _ = engine.infer(input_data)
        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000  # Convert to ms
        latencies.append(latency)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{test_runs} runs completed")

    # Calculate statistics
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Results for {name}")
    print(f"{'=' * 60}")
    print(f"Average Latency:  {avg_latency:.3f} ms")
    print(f"Std Dev:          {std_latency:.3f} ms")
    print(f"Min Latency:      {min_latency:.3f} ms")
    print(f"Max Latency:      {max_latency:.3f} ms")
    print(f"P50 Latency:      {p50_latency:.3f} ms")
    print(f"P95 Latency:      {p95_latency:.3f} ms")
    print(f"P99 Latency:      {p99_latency:.3f} ms")
    print(f"Throughput:       {fps:.1f} FPS")
    print(f"{'=' * 60}\n")

    return {
        'avg_latency': avg_latency,
        'std_latency': std_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'p50_latency': p50_latency,
        'p95_latency': p95_latency,
        'p99_latency': p99_latency,
        'fps': fps
    }


def benchmark_noise_suppression(engine_path: str, warmup: int, test: int):
    """Benchmark noise suppression engine"""
    return benchmark_engine(
        engine_path=engine_path,
        input_shape=(257,),  # Frequency bins
        warmup_runs=warmup,
        test_runs=test,
        name="Noise Suppression"
    )


def benchmark_segmentation(engine_path: str, warmup: int, test: int):
    """Benchmark segmentation engine"""
    return benchmark_engine(
        engine_path=engine_path,
        input_shape=(3, 256, 256),  # RGB 256x256
        warmup_runs=warmup,
        test_runs=test,
        name="Background Segmentation"
    )


def compare_pytorch_vs_tensorrt(
    model_type: str,
    tensorrt_engine: str,
    warmup: int,
    test: int
):
    """Compare PyTorch vs TensorRT performance"""
    print(f"\n{'=' * 60}")
    print("PyTorch vs TensorRT Performance Comparison")
    print(f"{'=' * 60}")

    if model_type == 'noise':
        # Import model
        from linvidia.models.noise_suppression import NoiseSuppressionRNN

        # Create PyTorch model
        model = NoiseSuppressionRNN(input_size=257, hidden_size=256, num_layers=2)
        model = model.cuda().eval()

        input_shape = (257,)
        input_data = torch.randn(1, *input_shape, device='cuda')

        # Benchmark PyTorch
        print("\nBenchmarking PyTorch model...")
        latencies = []

        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                _ = model(input_data)

            # Benchmark
            for _ in range(test):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                _ = model(input_data)
                torch.cuda.synchronize()
                end_time = time.perf_counter()

                latencies.append((end_time - start_time) * 1000)

        pytorch_latency = np.mean(latencies)
        pytorch_fps = 1000.0 / pytorch_latency

        print(f"PyTorch Average Latency: {pytorch_latency:.3f} ms")
        print(f"PyTorch Throughput:      {pytorch_fps:.1f} FPS")

        # Benchmark TensorRT
        trt_results = benchmark_noise_suppression(tensorrt_engine, warmup, test)

        # Calculate speedup
        speedup = pytorch_latency / trt_results['avg_latency']

        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        print(f"PyTorch:   {pytorch_latency:.3f} ms ({pytorch_fps:.1f} FPS)")
        print(f"TensorRT:  {trt_results['avg_latency']:.3f} ms ({trt_results['fps']:.1f} FPS)")
        print(f"Speedup:   {speedup:.2f}x faster with TensorRT")
        print(f"{'=' * 60}\n")

    elif model_type == 'segmentation':
        print("Segmentation comparison not yet implemented")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark TensorRT engines for LiNvidia Broadcast'
    )

    parser.add_argument(
        '--engine',
        type=str,
        required=True,
        help='Path to TensorRT engine file'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        choices=['noise', 'segmentation'],
        required=True,
        help='Type of model to benchmark'
    )

    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup runs (default: 10)'
    )

    parser.add_argument(
        '--test-runs',
        type=int,
        default=100,
        help='Number of test runs (default: 100)'
    )

    parser.add_argument(
        '--compare-pytorch',
        action='store_true',
        help='Compare with PyTorch performance'
    )

    args = parser.parse_args()

    # Check if engine exists
    if not Path(args.engine).exists():
        print(f"Error: Engine file not found: {args.engine}")
        print("\nFirst export models with:")
        print("  python scripts/export_to_tensorrt.py --model both")
        sys.exit(1)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)

    # Print system info
    print(f"\nSystem Information:")
    print(f"{'=' * 60}")
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch:      {torch.__version__}")

    try:
        import tensorrt as trt
        print(f"TensorRT:     {trt.__version__}")
    except ImportError:
        print("TensorRT:     Not installed")

    print(f"{'=' * 60}")

    try:
        if args.compare_pytorch:
            compare_pytorch_vs_tensorrt(
                args.model_type,
                args.engine,
                args.warmup,
                args.test_runs
            )
        else:
            # Benchmark TensorRT only
            if args.model_type == 'noise':
                benchmark_noise_suppression(args.engine, args.warmup, args.test_runs)
            elif args.model_type == 'segmentation':
                benchmark_segmentation(args.engine, args.warmup, args.test_runs)

    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

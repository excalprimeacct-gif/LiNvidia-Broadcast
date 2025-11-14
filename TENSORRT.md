# TensorRT Optimization Guide

This guide explains how to convert LiNvidia Broadcast models to TensorRT for maximum performance on NVIDIA tensor cores.

## Overview

TensorRT is NVIDIA's high-performance deep learning inference library that provides:
- **FP16 Precision**: 2x throughput on tensor cores
- **Layer Fusion**: Optimizes operations for reduced latency
- **Kernel Auto-Tuning**: Hardware-specific optimizations
- **Memory Optimization**: Reduced memory footprint

Expected speedup: **2-5x faster** than PyTorch

## Requirements

- NVIDIA GPU with Tensor Cores (RTX 20xx or newer)
- CUDA 11.0+
- TensorRT 8.0+
- PyTorch 2.0+

## Quick Start

### 1. Export Models to TensorRT

```bash
# Export both models with FP16 precision
linvidia export-tensorrt --model both

# Or export individually
linvidia export-tensorrt --model noise
linvidia export-tensorrt --model segmentation

# Export to custom directory
linvidia export-tensorrt --model both --output-dir ./my_engines

# Export with custom checkpoint
linvidia export-tensorrt --model noise --checkpoint ./checkpoints/noise_model.pt
```

This will create:
- `models/tensorrt/noise_suppression.onnx`
- `models/tensorrt/noise_suppression.engine`
- `models/tensorrt/segmentation.onnx`
- `models/tensorrt/segmentation.engine`

### 2. Benchmark Performance

```bash
# Benchmark TensorRT engine
linvidia benchmark-tensorrt \
  --engine models/tensorrt/noise_suppression.engine \
  --model-type noise

# Compare with PyTorch
linvidia benchmark-tensorrt \
  --engine models/tensorrt/noise_suppression.engine \
  --model-type noise \
  --compare-pytorch

# Extended benchmark
linvidia benchmark-tensorrt \
  --engine models/tensorrt/segmentation.engine \
  --model-type segmentation \
  --warmup 50 \
  --test-runs 500
```

## Using Scripts Directly

### Export Script

```bash
# Export models
python scripts/export_to_tensorrt.py --model both

# Options
python scripts/export_to_tensorrt.py \
  --model both \
  --output-dir ./models/tensorrt \
  --freq-bins 257 \
  --input-size 256 256 \
  --no-fp16  # Disable FP16 (not recommended)
```

### Benchmark Script

```bash
# Comprehensive benchmarking
python scripts/benchmark_tensorrt.py \
  --engine models/tensorrt/noise_suppression.engine \
  --model-type noise \
  --compare-pytorch \
  --warmup 10 \
  --test-runs 100
```

## Integration with Pipeline

### Using TensorRT in Code

```python
from linvidia.inference.engine import TensorRTEngine, TensorRTNoiseSuppressionEngine

# For noise suppression
engine = TensorRTNoiseSuppressionEngine(
    engine_path='models/tensorrt/noise_suppression.engine',
    freq_bins=257,
    use_cuda_stream=True
)

# Process audio
magnitude_out, phase_out = engine.suppress_noise(
    magnitude=magnitude_spectrum,
    phase=phase_spectrum,
    strength=0.95
)

# Get latency stats
avg_latency = engine.get_latency()
print(f"Average latency: {avg_latency:.3f} ms")
```

### Generic TensorRT Engine

```python
from linvidia.inference.engine import TensorRTEngine
import numpy as np

# Load engine
engine = TensorRTEngine('path/to/model.engine', use_cuda_stream=True)

# Run inference
input_data = np.random.randn(1, 3, 256, 256).astype(np.float32)
output = engine.infer(input_data)

# Performance stats
print(f"Average latency: {engine.get_average_latency():.3f} ms")
engine.reset_stats()
```

## Performance Optimization

### FP16 vs FP32

**FP16 (Recommended):**
- 2x faster inference
- 50% less memory
- Minimal accuracy loss
- Requires Tensor Cores (RTX GPUs)

**FP32:**
- Better accuracy (negligible difference in practice)
- Slower inference
- More memory usage

### Batch Size

Currently optimized for batch_size=1 (real-time streaming).

For batch processing:
```python
# Process multiple frames
results = engine.infer_batch(input_batch)
```

### Dynamic Shapes

For variable input sizes:
```python
from linvidia.inference.converter import pytorch_to_tensorrt

pytorch_to_tensorrt(
    model=model,
    input_shape=(3, 256, 256),
    output_dir='./engines',
    dynamic_shapes=True  # Enable dynamic shapes
)
```

## Expected Performance

### Noise Suppression (RTX 3060)

| Mode | Latency | FPS | Speedup |
|------|---------|-----|---------|
| PyTorch FP32 | ~2.5 ms | 400 | 1.0x |
| PyTorch FP16 | ~1.8 ms | 555 | 1.4x |
| TensorRT FP16 | ~0.8 ms | 1250 | 3.1x |

### Background Segmentation (RTX 3060)

| Mode | Latency | FPS | Speedup |
|------|---------|-----|---------|
| PyTorch FP32 | ~18 ms | 55 | 1.0x |
| PyTorch FP16 | ~12 ms | 83 | 1.5x |
| TensorRT FP16 | ~8 ms | 125 | 2.25x |

*Actual performance may vary based on GPU model and system configuration*

## Conversion Process Details

### Step 1: PyTorch → ONNX

- Exports model to ONNX format
- Includes graph optimizations
- Validates ONNX model

### Step 2: ONNX → TensorRT

- Parses ONNX graph
- Applies layer fusion
- Generates optimized kernels
- Builds serialized engine

### Step 3: Engine Serialization

- Saves optimized engine to disk
- Engine is GPU-specific
- Recompile for different GPUs

## Troubleshooting

### TensorRT Not Found

```bash
# Install TensorRT
pip install nvidia-tensorrt

# Or download from NVIDIA
# https://developer.nvidia.com/tensorrt
```

### CUDA Out of Memory

```bash
# Use smaller models or reduce batch size
# FP16 uses 50% less memory than FP32
linvidia export-tensorrt --model both  # FP16 by default
```

### Slow First Inference

TensorRT engines are optimized on first run. Subsequent inferences are fast.

```python
# Warmup
for _ in range(10):
    _ = engine.infer(dummy_input)
```

### Engine Not Portable

TensorRT engines are compiled for specific GPUs. Recompile for different hardware:

```bash
# On target machine
linvidia export-tensorrt --model both
```

### ONNX Export Fails

Ensure model is in eval mode and on CUDA:
```python
model.eval()
model = model.cuda()
```

## Advanced Usage

### Custom Workspace Size

```python
from linvidia.inference.converter import convert_to_tensorrt

convert_to_tensorrt(
    onnx_path='model.onnx',
    engine_path='model.engine',
    fp16_mode=True,
    max_workspace_size=2 << 30  # 2GB
)
```

### Multiple Optimization Profiles

For models with varying input sizes:
```python
# Set min, optimal, and max shapes
# TensorRT will optimize for the optimal shape
```

### Engine Inspection

```python
import tensorrt as trt

# Load engine
with open('model.engine', 'rb') as f:
    engine_data = f.read()

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)

# Inspect
print(f"Num bindings: {engine.num_bindings}")
print(f"Max batch size: {engine.max_batch_size}")

for i in range(engine.num_bindings):
    print(f"Binding {i}: {engine.get_binding_name(i)}")
    print(f"  Shape: {engine.get_binding_shape(i)}")
    print(f"  Dtype: {engine.get_binding_dtype(i)}")
```

## Best Practices

1. **Always use FP16** on RTX GPUs (2x faster, minimal accuracy loss)
2. **Warmup engines** before benchmarking (first run is slower)
3. **Recompile for target GPU** (engines are hardware-specific)
4. **Profile on real data** (synthetic benchmarks may differ)
5. **Monitor GPU utilization** (`nvidia-smi dmon`)
6. **Use CUDA streams** for async execution (enabled by default)

## Integration Checklist

- [ ] Export models to TensorRT
- [ ] Benchmark performance
- [ ] Update config to use TensorRT engines
- [ ] Verify accuracy on real data
- [ ] Profile end-to-end pipeline
- [ ] Document engine paths in config

## References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX Documentation](https://onnx.ai/)
- [Tensor Core Programming](https://docs.nvidia.com/cuda/tensor-core/)

## Next Steps

After optimizing with TensorRT:
1. Integrate engines into production pipeline
2. Compare accuracy with PyTorch models
3. Profile full system latency
4. Consider INT8 quantization for even faster inference (advanced)

---

For issues or questions, see [GitHub Issues](https://github.com/hellasleeper108/LiNvidia-Broadcast/issues)

"""
TensorRT inference engine for real-time processing

Optimized for low latency and high throughput using tensor cores
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from typing import Optional, List, Tuple
import time


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTEngine:
    """
    TensorRT inference engine wrapper

    Provides high-performance inference with minimal overhead
    """

    def __init__(
        self,
        engine_path: str,
        use_cuda_stream: bool = True
    ):
        """
        Initialize TensorRT engine

        Args:
            engine_path: Path to TensorRT engine file
            use_cuda_stream: Use CUDA stream for async execution
        """
        self.engine_path = engine_path
        self.use_cuda_stream = use_cuda_stream

        # Load engine
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

        # Performance tracking
        self.inference_count = 0
        self.total_time = 0.0

    def _load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """Load TensorRT engine from file"""
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        engine = self.runtime.deserialize_cuda_engine(engine_data)

        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        print(f"Loaded TensorRT engine from {engine_path}")
        return engine

    def _allocate_buffers(self) -> Tuple[List, List, List, cuda.Stream]:
        """Allocate GPU buffers for inputs and outputs"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream() if self.use_cuda_stream else None

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append to the appropriate list
            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape, 'name': tensor_name})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape, 'name': tensor_name})

        return inputs, outputs, bindings, stream

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference

        Args:
            input_data: Input numpy array

        Returns:
            Output numpy array
        """
        start_time = time.perf_counter()

        # Ensure input is the right shape
        if len(self.inputs) != 1:
            raise ValueError("Engine expects exactly one input")

        input_buffer = self.inputs[0]

        # Flatten and copy input to host buffer
        input_flat = input_data.flatten()
        np.copyto(input_buffer['host'], input_flat)

        # Transfer input data to GPU
        cuda.memcpy_htod_async(
            input_buffer['device'],
            input_buffer['host'],
            self.stream
        ) if self.use_cuda_stream else cuda.memcpy_htod(
            input_buffer['device'],
            input_buffer['host']
        )

        # Set tensor addresses
        for i, inp in enumerate(self.inputs):
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for i, out in enumerate(self.outputs):
            self.context.set_tensor_address(out['name'], int(out['device']))

        # Run inference
        if self.use_cuda_stream:
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            self.stream.synchronize()
        else:
            self.context.execute_v2(bindings=self.bindings)

        # Transfer predictions back to host
        output_buffer = self.outputs[0]
        cuda.memcpy_dtoh_async(
            output_buffer['host'],
            output_buffer['device'],
            self.stream
        ) if self.use_cuda_stream else cuda.memcpy_dtoh(
            output_buffer['host'],
            output_buffer['device']
        )

        if self.use_cuda_stream:
            self.stream.synchronize()

        # Reshape output
        output = output_buffer['host'].reshape(output_buffer['shape'])

        # Track performance
        end_time = time.perf_counter()
        self.inference_count += 1
        self.total_time += (end_time - start_time)

        return output

    def infer_batch(self, input_batch: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch

        Args:
            input_batch: Batch of inputs (batch_size, ...)

        Returns:
            Batch of outputs
        """
        # For now, process sequentially
        # TODO: Implement true batched inference
        results = []
        for input_data in input_batch:
            output = self.infer(input_data)
            results.append(output)

        return np.stack(results)

    def get_input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape"""
        return self.inputs[0]['shape']

    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape"""
        return self.outputs[0]['shape']

    def get_average_latency(self) -> float:
        """Get average inference latency in milliseconds"""
        if self.inference_count == 0:
            return 0.0
        return (self.total_time / self.inference_count) * 1000

    def reset_stats(self):
        """Reset performance statistics"""
        self.inference_count = 0
        self.total_time = 0.0

    def __del__(self):
        """Cleanup resources"""
        # Free GPU memory
        for inp in self.inputs:
            inp['device'].free()
        for out in self.outputs:
            out['device'].free()


class TensorRTNoiseSuppressionEngine:
    """
    Specialized TensorRT engine for noise suppression

    Handles preprocessing, inference, and postprocessing
    """

    def __init__(
        self,
        engine_path: str,
        freq_bins: int = 257,
        use_cuda_stream: bool = True
    ):
        """
        Initialize noise suppression engine

        Args:
            engine_path: Path to TensorRT engine
            freq_bins: Number of frequency bins
            use_cuda_stream: Use CUDA stream
        """
        self.engine = TensorRTEngine(engine_path, use_cuda_stream)
        self.freq_bins = freq_bins

    def suppress_noise(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray,
        strength: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Suppress noise in audio spectrum

        Args:
            magnitude: Magnitude spectrum (freq_bins,)
            phase: Phase spectrum (freq_bins,)
            strength: Suppression strength [0, 1]

        Returns:
            magnitude_out: Suppressed magnitude
            phase_out: Phase (unchanged)
        """
        # Prepare input
        mag_input = magnitude.reshape(1, -1).astype(np.float32)

        # Normalize
        mag_mean = mag_input.mean()
        mag_std = mag_input.std() + 1e-8
        mag_normalized = (mag_input - mag_mean) / mag_std

        # Inference
        gain_mask = self.engine.infer(mag_normalized)

        # Apply suppression strength
        if strength < 1.0:
            gain_mask = 1.0 - strength * (1.0 - gain_mask)

        # Apply gain mask
        mag_suppressed = magnitude * gain_mask.flatten()

        return mag_suppressed, phase

    def get_latency(self) -> float:
        """Get average inference latency in ms"""
        return self.engine.get_average_latency()

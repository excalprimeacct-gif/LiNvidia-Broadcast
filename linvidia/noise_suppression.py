"""
Real-time noise suppression pipeline

Integrates audio capture, processing, model inference, and playback
for low-latency noise suppression
"""

import numpy as np
import threading
import time
from collections import deque
from typing import Optional
from loguru import logger

from .audio import AudioCapture, AudioPlayback, AudioProcessor
from .models import NoiseSuppressionModel, create_noise_suppression_model
from .utils import Config


class RealtimeNoiseSuppression:
    """
    Real-time noise suppression system

    Captures audio, processes with neural network, and outputs clean audio
    with minimal latency.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model: Optional[NoiseSuppressionModel] = None,
        input_device: Optional[str] = None,
        output_device: Optional[str] = None
    ):
        """
        Initialize real-time noise suppression

        Args:
            config: Configuration object
            model: Pre-loaded noise suppression model
            input_device: Input device name
            output_device: Output device name
        """
        self.config = config or Config()

        # Audio processor
        self.processor = AudioProcessor(
            sample_rate=self.config.audio.sample_rate,
            frame_size=self.config.audio.frame_size,
            hop_size=self.config.audio.hop_size,
            n_fft=self.config.audio.n_fft,
            window=self.config.audio.window
        )

        # Audio capture
        self.capture = AudioCapture(
            device=input_device,
            sample_rate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
            frame_size=self.config.audio.hop_size
        )

        # Audio playback
        self.playback = AudioPlayback(
            device=output_device,
            sample_rate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
            frame_size=self.config.audio.hop_size
        )

        # Noise suppression model
        if model is None:
            logger.info("Creating noise suppression model...")
            self.model = create_noise_suppression_model(
                freq_bins=self.processor.freq_bins,
                hidden_size=self.config.model.hidden_size,
                num_layers=self.config.model.num_layers,
                device=self.config.model.device,
                use_fp16=self.config.model.use_fp16
            )
        else:
            self.model = model

        # Processing state
        self.is_running = False
        self.processing_thread = None

        # Statistics (bounded ring-buffers; pop(0) on a list is O(n))
        self.frames_processed = 0
        self.processing_times = deque(maxlen=1000)
        self.latencies = deque(maxlen=1000)

    def _process_callback(self, audio_frame: np.ndarray):
        """
        Audio processing callback

        Called for each captured audio frame
        """
        start_time = time.perf_counter()

        try:
            # Process with overlap-add
            def process_spectrum(magnitude, phase):
                # Apply noise suppression
                if self.config.noise_suppression.enable:
                    mag_out, phase_out = self.model.suppress_noise(
                        magnitude,
                        phase,
                        strength=self.config.noise_suppression.strength
                    )
                else:
                    mag_out, phase_out = magnitude, phase

                return mag_out, phase_out

            # Process frame
            output_frame = self.processor.process_frame_overlap_add(
                audio_frame,
                process_spectrum
            )

            # Write to playback
            self.playback.write(output_frame, timeout=0.01)

            # Update statistics
            end_time = time.perf_counter()
            processing_time = (end_time - start_time) * 1000  # ms

            self.frames_processed += 1
            self.processing_times.append(processing_time)

        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")

    def start(self):
        """Start real-time noise suppression"""
        if self.is_running:
            logger.warning("Noise suppression already running")
            return

        logger.info("Starting real-time noise suppression...")

        # Start playback first (to avoid initial buffer underrun)
        self.playback.start()

        # Start capture with callback
        self.capture.start(callback=self._process_callback)

        self.is_running = True

        logger.info("Real-time noise suppression started")
        logger.info(f"Input: {self.capture.get_device_info()['name']}")
        logger.info(f"Output: {self.playback.get_device_info()['name']}")
        logger.info(f"Latency target: {self.get_target_latency():.1f} ms")

    def stop(self):
        """Stop real-time noise suppression"""
        if not self.is_running:
            return

        logger.info("Stopping real-time noise suppression...")

        self.is_running = False

        # Stop capture and playback
        self.capture.stop()
        self.playback.stop()

        # Print statistics
        self._print_statistics()

        logger.info("Real-time noise suppression stopped")

    def _print_statistics(self):
        """Print performance statistics"""
        if not self.processing_times:
            return

        avg_processing = np.mean(self.processing_times)
        max_processing = np.max(self.processing_times)
        min_processing = np.min(self.processing_times)

        logger.info("=== Performance Statistics ===")
        logger.info(f"Frames processed: {self.frames_processed}")
        logger.info(f"Processing time (avg): {avg_processing:.2f} ms")
        logger.info(f"Processing time (min/max): {min_processing:.2f} / {max_processing:.2f} ms")
        logger.info(f"Buffer overruns: {self.capture.buffer_overruns}")
        logger.info(f"Buffer underruns: {self.playback.buffer_underruns}")

    def get_target_latency(self) -> float:
        """Get target system latency in milliseconds"""
        # Latency = capture buffer + processing + playback buffer
        frame_time = (self.config.audio.hop_size / self.config.audio.sample_rate) * 1000
        return frame_time * 3  # Typically 3 frame times

    def run(self, duration: Optional[float] = None):
        """
        Run noise suppression for specified duration

        Args:
            duration: Duration in seconds (None = run until stopped)
        """
        self.start()

        try:
            if duration:
                logger.info(f"Running for {duration} seconds...")
                time.sleep(duration)
            else:
                logger.info("Running (press Ctrl+C to stop)...")
                while self.is_running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


def create_noise_suppression(
    input_device: Optional[str] = None,
    output_device: Optional[str] = None,
    strength: float = 0.95,
    config_path: Optional[str] = None
) -> RealtimeNoiseSuppression:
    """
    Factory function to create noise suppression system

    Args:
        input_device: Input device name
        output_device: Output device name
        strength: Suppression strength [0, 1]
        config_path: Path to configuration file

    Returns:
        RealtimeNoiseSuppression instance
    """
    # Load configuration
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config()

    # Update suppression strength
    config.noise_suppression.strength = strength

    return RealtimeNoiseSuppression(
        config=config,
        input_device=input_device,
        output_device=output_device
    )

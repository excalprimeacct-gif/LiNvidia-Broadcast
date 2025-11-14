"""
Audio capture module for real-time audio input
Supports multiple backends: PulseAudio, PipeWire, PortAudio
"""

import numpy as np
import sounddevice as sd
from typing import Optional, Callable
from queue import Queue
import threading


class AudioCapture:
    """
    Real-time audio capture with low latency

    Captures audio from specified input device and provides
    frames for processing via callback or queue.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 48000,
        channels: int = 1,
        frame_size: int = 480,  # 10ms at 48kHz
        dtype: str = 'float32'
    ):
        """
        Initialize audio capture

        Args:
            device: Input device name (None for default)
            sample_rate: Sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            frame_size: Size of each audio frame in samples
            dtype: Data type for audio samples
        """
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = frame_size
        self.dtype = dtype

        # Resolve device
        self.device_index = self._resolve_device(device)

        # Stream and callback
        self.stream = None
        self.callback_fn = None
        self.queue = Queue()
        self.is_running = False

        # Stats
        self.frames_captured = 0
        self.buffer_overruns = 0

    def _resolve_device(self, device_name: Optional[str]) -> Optional[int]:
        """Resolve device name to device index"""
        if device_name is None:
            return None

        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if device_name.lower() in dev['name'].lower():
                if dev['max_input_channels'] > 0:
                    return idx

        raise ValueError(f"Input device '{device_name}' not found")

    def _audio_callback(self, indata, frames, time, status):
        """Internal callback for audio stream"""
        if status:
            print(f"Audio capture status: {status}")
            if status.input_overflow:
                self.buffer_overruns += 1

        # Convert to mono if needed
        if self.channels == 1 and indata.shape[1] > 1:
            audio_frame = indata[:, 0].copy()
        else:
            audio_frame = indata.copy()

        # Increment counter
        self.frames_captured += 1

        # Call user callback if set
        if self.callback_fn:
            self.callback_fn(audio_frame)
        else:
            # Otherwise, put in queue
            if not self.queue.full():
                self.queue.put(audio_frame)

    def start(self, callback: Optional[Callable] = None):
        """
        Start audio capture

        Args:
            callback: Optional callback function called for each frame
                     Signature: callback(audio_frame: np.ndarray)
        """
        if self.is_running:
            raise RuntimeError("Audio capture already running")

        self.callback_fn = callback
        self.is_running = True

        # Create input stream
        self.stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.frame_size,
            callback=self._audio_callback,
            latency='low'
        )

        self.stream.start()
        print(f"Audio capture started: {self.get_device_info()['name']}")

    def stop(self):
        """Stop audio capture"""
        if not self.is_running:
            return

        self.is_running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        print(f"Audio capture stopped. Captured {self.frames_captured} frames, "
              f"{self.buffer_overruns} overruns")

    def read(self, timeout: Optional[float] = None) -> np.ndarray:
        """
        Read next audio frame from queue (blocking)

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            audio_frame: Audio data
        """
        return self.queue.get(timeout=timeout)

    def get_device_info(self) -> dict:
        """Get information about the current device"""
        if self.device_index is None:
            return sd.query_devices(kind='input')
        return sd.query_devices(self.device_index)

    @staticmethod
    def list_devices() -> list:
        """List all available input devices"""
        devices = sd.query_devices()
        input_devices = []

        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append({
                    'index': idx,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': dev['default_samplerate']
                })

        return input_devices

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

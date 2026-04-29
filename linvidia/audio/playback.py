"""
Audio playback module for real-time audio output
Supports multiple backends: PulseAudio, PipeWire, PortAudio
"""

import numpy as np
import sounddevice as sd
from typing import Optional
from queue import Queue, Full, Empty
import threading


class AudioPlayback:
    """
    Real-time audio playback with low latency

    Plays processed audio to specified output device with
    minimal buffering for low latency.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 48000,
        channels: int = 1,
        frame_size: int = 480,  # 10ms at 48kHz
        dtype: str = 'float32',
        buffer_size: int = 3  # Number of frames to buffer
    ):
        """
        Initialize audio playback

        Args:
            device: Output device name (None for default)
            sample_rate: Sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            frame_size: Size of each audio frame in samples
            dtype: Data type for audio samples
            buffer_size: Number of frames to buffer (larger = more latency but safer)
        """
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = frame_size
        self.dtype = dtype

        # Resolve device
        self.device_index = self._resolve_device(device)

        # Stream and buffer
        self.stream = None
        self.queue = Queue(maxsize=buffer_size)
        self.is_running = False

        # Stats
        self.frames_played = 0
        self.buffer_underruns = 0

        # Silence frame for underruns
        self.silence = np.zeros((frame_size, channels), dtype=dtype)

    def _resolve_device(self, device_name: Optional[str]) -> Optional[int]:
        """Resolve device name to device index"""
        if device_name is None:
            return None

        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if device_name.lower() in dev['name'].lower():
                if dev['max_output_channels'] > 0:
                    return idx

        raise ValueError(f"Output device '{device_name}' not found")

    def _audio_callback(self, outdata, frames, time, status):
        """Internal callback for audio stream"""
        if status:
            print(f"Audio playback status: {status}")

        try:
            # Get frame from queue (no double-counting: status underflow above
            # is just a hardware warning; queue-empty here is the real underrun).
            audio_frame = self.queue.get_nowait()
        except Empty:
            outdata[:] = self.silence
            self.buffer_underruns += 1
            return

        # Reshape for output if needed
        if audio_frame.ndim == 1:
            audio_frame = audio_frame.reshape(-1, 1)

        # Match channel count expected by the stream
        if audio_frame.shape[1] != self.channels:
            if audio_frame.shape[1] == 1 and self.channels > 1:
                audio_frame = np.broadcast_to(audio_frame, (audio_frame.shape[0], self.channels))
            else:
                audio_frame = audio_frame[:, : self.channels]

        # Match block size expected by the stream
        if audio_frame.shape[0] != frames:
            if audio_frame.shape[0] < frames:
                audio_frame = np.pad(audio_frame, ((0, frames - audio_frame.shape[0]), (0, 0)))
            else:
                audio_frame = audio_frame[:frames]

        outdata[:] = audio_frame
        self.frames_played += 1

    def start(self):
        """Start audio playback"""
        if self.is_running:
            raise RuntimeError("Audio playback already running")

        self.is_running = True

        # Create output stream
        self.stream = sd.OutputStream(
            device=self.device_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.frame_size,
            callback=self._audio_callback,
            latency='low'
        )

        self.stream.start()
        print(f"Audio playback started: {self.get_device_info()['name']}")

    def stop(self):
        """Stop audio playback"""
        if not self.is_running:
            return

        self.is_running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        print(f"Audio playback stopped. Played {self.frames_played} frames, "
              f"{self.buffer_underruns} underruns")

    def write(self, audio_frame: np.ndarray, timeout: Optional[float] = None):
        """
        Write audio frame to playback queue

        Args:
            audio_frame: Audio data to play
            timeout: Maximum time to wait if queue is full
        """
        try:
            self.queue.put(audio_frame, timeout=timeout)
        except Full:
            print("Playback queue full, dropping frame")

    def get_device_info(self) -> dict:
        """Get information about the current device"""
        if self.device_index is None:
            return sd.query_devices(kind='output')
        return sd.query_devices(self.device_index)

    @staticmethod
    def list_devices() -> list:
        """List all available output devices"""
        devices = sd.query_devices()
        output_devices = []

        for idx, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                output_devices.append({
                    'index': idx,
                    'name': dev['name'],
                    'channels': dev['max_output_channels'],
                    'sample_rate': dev['default_samplerate']
                })

        return output_devices

    def get_latency(self) -> float:
        """Get current playback latency in seconds"""
        if self.stream:
            return self.stream.latency
        return 0.0

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

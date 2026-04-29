"""
Audio processing utilities for noise suppression
Handles framing, windowing, STFT, and inverse operations
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


class AudioProcessor:
    """
    Audio processor for real-time noise suppression

    Performs Short-Time Fourier Transform (STFT) and inverse STFT
    with overlapping frames for smooth reconstruction.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        frame_size: int = 480,  # 10ms at 48kHz
        hop_size: Optional[int] = None,
        n_fft: int = 512,
        window: str = 'hann'
    ):
        """
        Initialize audio processor

        Args:
            sample_rate: Audio sample rate in Hz
            frame_size: Size of each frame in samples
            hop_size: Number of samples between frames (default: frame_size // 2)
            n_fft: FFT size (should be >= frame_size)
            window: Window function ('hann', 'hamming', 'blackman')
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size or frame_size // 2
        self.n_fft = max(n_fft, frame_size)

        # Create window function
        self.window = self._create_window(window, frame_size)

        # Synthesis window for perfect reconstruction
        self.synthesis_window = self._create_synthesis_window()

        # Buffers for overlap-add
        self.input_buffer = np.zeros(self.n_fft, dtype=np.float32)
        self.output_buffer = np.zeros(self.n_fft, dtype=np.float32)

        # Frequency bins
        self.freq_bins = self.n_fft // 2 + 1

        # Lazily-cached mel filterbank, keyed by n_mels.
        self._mel_cache: dict = {}

    def _create_window(self, window_type: str, size: int) -> np.ndarray:
        """Create window function"""
        if window_type == 'hann':
            return signal.windows.hann(size, sym=False).astype(np.float32)
        elif window_type == 'hamming':
            return signal.windows.hamming(size, sym=False).astype(np.float32)
        elif window_type == 'blackman':
            return signal.windows.blackman(size, sym=False).astype(np.float32)
        else:
            raise ValueError(f"Unknown window type: {window_type}")

    def _create_synthesis_window(self) -> np.ndarray:
        """Create synthesis window for perfect reconstruction"""
        # For 50% overlap with Hann window, synthesis window is the same as analysis
        synthesis_win = self.window.copy()

        # Normalize for perfect reconstruction
        # Window squared and summed across overlaps should equal 1
        win_squared_sum = np.zeros(self.n_fft)
        for i in range(0, self.n_fft, self.hop_size):
            if i + self.frame_size <= self.n_fft:
                win_squared_sum[i:i+self.frame_size] += self.window ** 2

        # Avoid division by zero
        win_squared_sum = np.maximum(win_squared_sum, 1e-8)

        return synthesis_win / win_squared_sum[:self.frame_size]

    def stft(self, audio_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform

        Args:
            audio_frame: Input audio frame (frame_size samples)

        Returns:
            magnitude: Magnitude spectrum (freq_bins,)
            phase: Phase spectrum (freq_bins,)
        """
        # Apply window
        windowed = audio_frame[:self.frame_size] * self.window

        # Zero-pad if needed
        if len(windowed) < self.n_fft:
            windowed = np.pad(windowed, (0, self.n_fft - len(windowed)))

        # Compute FFT
        spectrum = np.fft.rfft(windowed, n=self.n_fft)

        # Extract magnitude and phase
        magnitude = np.abs(spectrum).astype(np.float32)
        phase = np.angle(spectrum).astype(np.float32)

        return magnitude, phase

    def istft(self, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Compute Inverse Short-Time Fourier Transform

        Args:
            magnitude: Magnitude spectrum (freq_bins,)
            phase: Phase spectrum (freq_bins,)

        Returns:
            audio_frame: Reconstructed audio frame (frame_size samples)
        """
        # Reconstruct complex spectrum
        spectrum = magnitude * np.exp(1j * phase)

        # Inverse FFT
        audio = np.fft.irfft(spectrum, n=self.n_fft).astype(np.float32)

        # Apply synthesis window
        audio[:self.frame_size] *= self.synthesis_window

        return audio[:self.frame_size]

    def process_frame_overlap_add(
        self,
        audio_frame: np.ndarray,
        process_fn
    ) -> np.ndarray:
        """
        Process audio frame with overlap-add for smooth reconstruction

        Args:
            audio_frame: Input audio frame
            process_fn: Function that processes (magnitude, phase) and returns (magnitude, phase)

        Returns:
            output_frame: Processed audio frame
        """
        # Shift input buffer
        self.input_buffer[:-self.hop_size] = self.input_buffer[self.hop_size:]
        self.input_buffer[-self.hop_size:] = audio_frame[:self.hop_size]

        # STFT
        magnitude, phase = self.stft(self.input_buffer[:self.frame_size])

        # Process
        mag_processed, phase_processed = process_fn(magnitude, phase)

        # ISTFT
        reconstructed = self.istft(mag_processed, phase_processed)

        # Overlap-add
        self.output_buffer[:self.frame_size] += reconstructed

        # Extract output
        output_frame = self.output_buffer[:self.hop_size].copy()

        # Shift output buffer
        self.output_buffer[:-self.hop_size] = self.output_buffer[self.hop_size:]
        self.output_buffer[-self.hop_size:] = 0

        return output_frame

    def to_mel_scale(self, magnitude: np.ndarray, n_mels: int = 64) -> np.ndarray:
        """
        Convert magnitude spectrum to mel scale

        Args:
            magnitude: Linear magnitude spectrum
            n_mels: Number of mel bands

        Returns:
            mel_magnitude: Mel-scale magnitude
        """
        mel_basis = self._mel_cache.get(n_mels)
        if mel_basis is None:
            mel_basis = self._mel_filterbank(n_mels)
            self._mel_cache[n_mels] = mel_basis
        return mel_basis @ magnitude

    def _mel_filterbank(self, n_mels: int) -> np.ndarray:
        """Create mel filterbank matrix"""
        # Mel scale conversion functions
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Create mel points
        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Convert to FFT bin numbers
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        # Create filterbank
        filterbank = np.zeros((n_mels, self.freq_bins))

        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Rising slope
            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)

            # Falling slope
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank.astype(np.float32)

    def compute_features(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Compute feature vector for noise suppression model

        Args:
            magnitude: Magnitude spectrum

        Returns:
            features: Feature vector (log magnitude, delta, delta-delta)
        """
        # Log magnitude (with small epsilon to avoid log(0))
        log_mag = np.log(magnitude + 1e-8).astype(np.float32)

        # Normalize
        log_mag = (log_mag - log_mag.mean()) / (log_mag.std() + 1e-8)

        return log_mag

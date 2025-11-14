"""
Audio augmentation techniques for noise suppression training
"""

import numpy as np
import random
from typing import Optional


class AudioAugmentation:
    """
    Audio augmentation for training robustness

    Applies various augmentations to improve model generalization
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        apply_reverb: bool = True,
        apply_eq: bool = True,
        apply_compression: bool = True,
        apply_pitch_shift: bool = False
    ):
        self.sample_rate = sample_rate
        self.apply_reverb = apply_reverb
        self.apply_eq = apply_eq
        self.apply_compression = apply_compression
        self.apply_pitch_shift = apply_pitch_shift

    def add_reverb(self, audio: np.ndarray, room_size: float = 0.5) -> np.ndarray:
        """
        Add simple reverb using comb filters

        Args:
            audio: Input audio
            room_size: Room size parameter [0, 1]

        Returns:
            Audio with reverb
        """
        # Simple comb filter delays (in samples)
        delays = [
            int(0.0297 * self.sample_rate * room_size),
            int(0.0371 * self.sample_rate * room_size),
            int(0.0411 * self.sample_rate * room_size),
            int(0.0437 * self.sample_rate * room_size),
        ]

        output = audio.copy()

        for delay in delays:
            if delay > 0 and delay < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay:] = audio[:-delay] * 0.3
                output += delayed

        return output

    def apply_eq_curve(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random EQ curve

        Args:
            audio: Input audio

        Returns:
            Equalized audio
        """
        from scipy import signal

        # Random filter parameters
        filter_type = random.choice(['lowpass', 'highpass', 'bandpass'])

        # Design filter
        if filter_type == 'lowpass':
            cutoff = random.uniform(4000, 8000)
            sos = signal.butter(4, cutoff, btype='low', fs=self.sample_rate, output='sos')
        elif filter_type == 'highpass':
            cutoff = random.uniform(100, 300)
            sos = signal.butter(4, cutoff, btype='high', fs=self.sample_rate, output='sos')
        else:  # bandpass
            low = random.uniform(200, 500)
            high = random.uniform(6000, 10000)
            sos = signal.butter(4, [low, high], btype='band', fs=self.sample_rate, output='sos')

        # Apply filter
        filtered = signal.sosfilt(sos, audio)

        return filtered.astype(np.float32)

    def dynamic_range_compression(
        self,
        audio: np.ndarray,
        threshold: float = -20.0,
        ratio: float = 4.0
    ) -> np.ndarray:
        """
        Apply dynamic range compression

        Args:
            audio: Input audio
            threshold: Threshold in dB
            ratio: Compression ratio

        Returns:
            Compressed audio
        """
        # Convert to dB
        epsilon = 1e-10
        audio_db = 20 * np.log10(np.abs(audio) + epsilon)

        # Compute gain
        gain_db = np.zeros_like(audio_db)
        mask = audio_db > threshold
        gain_db[mask] = (threshold - audio_db[mask]) * (1 - 1/ratio)

        # Apply gain
        gain_linear = 10 ** (gain_db / 20)
        compressed = audio * gain_linear

        return compressed.astype(np.float32)

    def add_noise(
        self,
        audio: np.ndarray,
        noise_level: float = 0.005
    ) -> np.ndarray:
        """
        Add random noise

        Args:
            audio: Input audio
            noise_level: Noise standard deviation

        Returns:
            Audio with added noise
        """
        noise = np.random.normal(0, noise_level, audio.shape)
        return (audio + noise).astype(np.float32)

    def time_stretch(
        self,
        audio: np.ndarray,
        rate: float = 1.0
    ) -> np.ndarray:
        """
        Time stretching (requires librosa)

        Args:
            audio: Input audio
            rate: Stretch rate (1.0 = no change, >1 = faster, <1 = slower)

        Returns:
            Time-stretched audio
        """
        try:
            import librosa
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            return stretched.astype(np.float32)
        except ImportError:
            print("Warning: librosa not installed, skipping time stretch")
            return audio

    def augment(
        self,
        audio: np.ndarray,
        augmentation_prob: float = 0.5
    ) -> np.ndarray:
        """
        Apply random augmentations

        Args:
            audio: Input audio
            augmentation_prob: Probability of applying each augmentation

        Returns:
            Augmented audio
        """
        augmented = audio.copy()

        # Reverb
        if self.apply_reverb and random.random() < augmentation_prob:
            room_size = random.uniform(0.2, 0.8)
            augmented = self.add_reverb(augmented, room_size)

        # EQ
        if self.apply_eq and random.random() < augmentation_prob:
            augmented = self.apply_eq_curve(augmented)

        # Compression
        if self.apply_compression and random.random() < augmentation_prob:
            threshold = random.uniform(-30, -10)
            ratio = random.uniform(2, 6)
            augmented = self.dynamic_range_compression(augmented, threshold, ratio)

        # Small amount of noise
        if random.random() < 0.3:
            augmented = self.add_noise(augmented, noise_level=random.uniform(0.001, 0.01))

        return augmented

    def __call__(self, audio: np.ndarray, prob: float = 0.5) -> np.ndarray:
        """
        Apply augmentations

        Args:
            audio: Input audio
            prob: Probability of applying augmentations

        Returns:
            Augmented audio
        """
        return self.augment(audio, prob)

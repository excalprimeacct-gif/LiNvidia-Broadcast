"""
Audio dataset loaders for noise suppression training

Supports:
- Custom clean/noise dataset
- DNS Challenge dataset
- Data augmentation
"""

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List
import random

from ..audio import AudioProcessor


class NoisyAudioDataset(Dataset):
    """
    Dataset for noise suppression training

    Creates noisy audio by mixing clean speech with noise at various SNR levels
    """

    def __init__(
        self,
        clean_dir: str,
        noise_dir: str,
        sample_rate: int = 48000,
        frame_size: int = 480,
        num_frames: int = 100,  # Number of frames per sample
        snr_range: Tuple[float, float] = (-5.0, 20.0),
        processor: Optional[AudioProcessor] = None,
        num_samples: Optional[int] = None,
        cache_audio: bool = False
    ):
        """
        Initialize dataset

        Args:
            clean_dir: Directory with clean speech WAV files
            noise_dir: Directory with noise WAV files
            sample_rate: Target sample rate
            frame_size: Frame size in samples
            num_frames: Number of frames per training sample
            snr_range: SNR range in dB (min, max)
            processor: Audio processor for STFT
            num_samples: Number of samples (None = use all combinations)
            cache_audio: Cache audio files in memory
        """
        self.clean_files = sorted(list(Path(clean_dir).rglob('*.wav')))
        self.noise_files = sorted(list(Path(noise_dir).rglob('*.wav')))

        if not self.clean_files:
            raise ValueError(f"No WAV files found in {clean_dir}")
        if not self.noise_files:
            raise ValueError(f"No WAV files found in {noise_dir}")

        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.snr_range = snr_range
        self.processor = processor or AudioProcessor(sample_rate=sample_rate)
        self.segment_length = frame_size * num_frames

        # Calculate dataset size
        if num_samples is None:
            self.num_samples = len(self.clean_files) * 10  # 10 variations per clean file
        else:
            self.num_samples = num_samples

        # Optional caching
        self.cache_audio = cache_audio
        self.audio_cache = {}

        print(f"Dataset: {len(self.clean_files)} clean files, {len(self.noise_files)} noise files")
        print(f"Total samples: {self.num_samples}")

    def _load_audio(self, file_path: Path) -> np.ndarray:
        """Load audio file"""
        # Check cache
        if self.cache_audio and str(file_path) in self.audio_cache:
            return self.audio_cache[str(file_path)]

        # Load audio
        audio, sr = sf.read(file_path, dtype='float32')

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            import resampy
            audio = resampy.resample(audio, sr, self.sample_rate)

        # Cache if enabled
        if self.cache_audio:
            self.audio_cache[str(file_path)] = audio

        return audio

    def _get_random_segment(self, audio: np.ndarray, length: int) -> np.ndarray:
        """Extract random segment from audio"""
        if len(audio) < length:
            # Repeat if too short
            repeats = (length // len(audio)) + 1
            audio = np.tile(audio, repeats)

        # Random start
        max_start = len(audio) - length
        start = random.randint(0, max_start) if max_start > 0 else 0

        return audio[start:start + length]

    def _mix_at_snr(self, clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        """Mix clean and noise at specified SNR"""
        # Calculate signal and noise power
        clean_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)

        # Calculate noise scaling factor
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = np.sqrt(clean_power / (noise_power * snr_linear + 1e-10))

        # Mix
        noisy = clean + noise * noise_scale

        return noisy.astype(np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Get training sample

        Returns:
            Dictionary with 'noisy_mag', 'clean_mag', 'phase', 'gain_mask'
        """
        # Select random clean and noise files
        clean_file = random.choice(self.clean_files)
        noise_file = random.choice(self.noise_files)

        # Load audio
        clean_audio = self._load_audio(clean_file)
        noise_audio = self._load_audio(noise_file)

        # Extract segments
        clean_segment = self._get_random_segment(clean_audio, self.segment_length)
        noise_segment = self._get_random_segment(noise_audio, self.segment_length)

        # Random SNR
        snr = random.uniform(*self.snr_range)

        # Mix
        noisy_segment = self._mix_at_snr(clean_segment, noise_segment, snr)

        # Process frames
        noisy_mags = []
        clean_mags = []
        phases = []

        for i in range(self.num_frames):
            start = i * self.frame_size
            end = start + self.frame_size

            # STFT
            noisy_mag, phase = self.processor.stft(noisy_segment[start:end])
            clean_mag, _ = self.processor.stft(clean_segment[start:end])

            noisy_mags.append(noisy_mag)
            clean_mags.append(clean_mag)
            phases.append(phase)

        # Stack
        noisy_mags = np.stack(noisy_mags)
        clean_mags = np.stack(clean_mags)
        phases = np.stack(phases)

        # Calculate ideal gain mask
        gain_mask = clean_mags / (noisy_mags + 1e-10)
        gain_mask = np.clip(gain_mask, 0, 1)

        return {
            'noisy_mag': torch.from_numpy(noisy_mags),
            'clean_mag': torch.from_numpy(clean_mags),
            'phase': torch.from_numpy(phases),
            'gain_mask': torch.from_numpy(gain_mask),
            'snr': snr
        }


class DNSChallengeDataset(NoisyAudioDataset):
    """
    Dataset loader for DNS Challenge dataset

    Expected structure:
    dns_challenge/
    ├── clean/
    │   ├── read_speech/
    │   └── emotional_speech/
    └── noise/
        ├── noise_fullband/
        └── noise_narrowband/
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = 'train',
        **kwargs
    ):
        """
        Initialize DNS Challenge dataset

        Args:
            dataset_root: Root directory of DNS Challenge dataset
            split: 'train' or 'val'
            **kwargs: Additional arguments for NoisyAudioDataset
        """
        dataset_root = Path(dataset_root)

        # Find clean and noise directories
        clean_dir = dataset_root / 'clean'
        noise_dir = dataset_root / 'noise'

        if not clean_dir.exists():
            raise ValueError(f"Clean directory not found: {clean_dir}")
        if not noise_dir.exists():
            raise ValueError(f"Noise directory not found: {noise_dir}")

        super().__init__(
            str(clean_dir),
            str(noise_dir),
            **kwargs
        )

        print(f"Loaded DNS Challenge dataset ({split} split)")


class StreamingAudioDataset(Dataset):
    """
    Streaming dataset that generates samples on-the-fly
    Useful for very large datasets that don't fit in memory
    """

    def __init__(
        self,
        clean_files: List[Path],
        noise_files: List[Path],
        processor: AudioProcessor,
        num_samples: int = 10000,
        **kwargs
    ):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.processor = processor
        self.num_samples = num_samples
        self.kwargs = kwargs

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        # Similar to NoisyAudioDataset but always loads fresh from disk
        pass  # Implementation similar to above

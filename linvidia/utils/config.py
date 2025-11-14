"""Configuration management"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class AudioConfig:
    """Audio configuration"""
    sample_rate: int = 48000
    channels: int = 1
    frame_size: int = 480  # 10ms at 48kHz
    hop_size: int = 240    # 5ms overlap
    n_fft: int = 512
    window: str = 'hann'


@dataclass
class ModelConfig:
    """Model configuration"""
    freq_bins: int = 257
    hidden_size: int = 256
    num_layers: int = 2
    use_fp16: bool = True
    device: str = 'cuda'


@dataclass
class InferenceConfig:
    """Inference configuration"""
    use_tensorrt: bool = True
    engine_path: Optional[str] = None
    batch_size: int = 1
    use_cuda_stream: bool = True


@dataclass
class NoiseSuppressionConfig:
    """Noise suppression configuration"""
    strength: float = 0.95
    enable: bool = True


@dataclass
class Config:
    """Main configuration"""
    audio: AudioConfig = None
    model: ModelConfig = None
    inference: InferenceConfig = None
    noise_suppression: NoiseSuppressionConfig = None

    def __post_init__(self):
        if self.audio is None:
            self.audio = AudioConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.noise_suppression is None:
            self.noise_suppression = NoiseSuppressionConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'audio': asdict(self.audio),
            'model': asdict(self.model),
            'inference': asdict(self.inference),
            'noise_suppression': asdict(self.noise_suppression)
        }

    def save(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        return cls(
            audio=AudioConfig(**config_dict.get('audio', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            inference=InferenceConfig(**config_dict.get('inference', {})),
            noise_suppression=NoiseSuppressionConfig(**config_dict.get('noise_suppression', {}))
        )

    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


def load_config(path: Optional[str] = None) -> Config:
    """
    Load configuration from file or create default

    Args:
        path: Path to config file (optional)

    Returns:
        Config object
    """
    if path and Path(path).exists():
        return Config.load(path)
    return Config()


def save_default_config(path: str = 'config/default.yaml'):
    """Save default configuration"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    config = Config()
    config.save(path)
    print(f"Default configuration saved to {path}")

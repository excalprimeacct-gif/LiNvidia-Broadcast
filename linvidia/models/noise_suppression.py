"""
Noise suppression neural network models

Implements RNN-based noise suppression optimized for tensor cores
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class NoiseSuppressionRNN(nn.Module):
    """
    RNN-based noise suppression model

    Uses GRU layers for temporal modeling and outputs a gain mask
    to apply to the input spectrum.

    Optimized for:
    - FP16 precision (tensor core acceleration)
    - Low latency (small model size)
    - Real-time inference (efficient operations)
    """

    def __init__(
        self,
        input_size: int = 257,  # FFT bins (n_fft // 2 + 1)
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize noise suppression model

        Args:
            input_size: Number of frequency bins
            hidden_size: Size of GRU hidden state
            num_layers: Number of GRU layers
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        )

        # GRU layers for temporal modeling
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2) if use_layer_norm else nn.Identity(),
            nn.Linear(hidden_size // 2, input_size),
            nn.Sigmoid()  # Gain mask in [0, 1]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input magnitude spectrum (batch, seq_len, freq_bins) or (batch, freq_bins)
            hidden: Previous hidden state (num_layers, batch, hidden_size)

        Returns:
            gain_mask: Gain mask to apply (same shape as input)
            hidden: Updated hidden state
        """
        # Handle single frame input
        squeeze_output = False
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, 1, freq_bins)
            squeeze_output = True

        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)

        # GRU
        if hidden is None:
            hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype
            )

        x, hidden = self.gru(x, hidden)  # (batch, seq_len, hidden_size)

        # Output projection (gain mask)
        gain_mask = self.output_proj(x)  # (batch, seq_len, freq_bins)

        if squeeze_output:
            gain_mask = gain_mask.squeeze(1)  # (batch, freq_bins)

        return gain_mask, hidden

    def init_hidden(self, batch_size: int = 1, device: str = 'cuda') -> torch.Tensor:
        """Initialize hidden state"""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=device, dtype=torch.float32
        )


class NoiseSuppressionModel:
    """
    High-level interface for noise suppression

    Handles preprocessing, inference, and postprocessing
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        use_fp16: bool = True
    ):
        """
        Initialize noise suppression model

        Args:
            model: PyTorch model
            device: Device to run inference on
            use_fp16: Whether to use FP16 precision (faster on tensor cores)
        """
        self.device = device
        self.use_fp16 = use_fp16

        # Move model to device
        self.model = model.to(device)
        self.model.eval()

        # Convert to FP16 if requested
        if use_fp16:
            self.model = self.model.half()

        # Hidden state for stateful inference
        self.hidden = None

    def reset_state(self):
        """Reset internal state"""
        self.hidden = None

    def suppress_noise(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray,
        strength: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Suppress noise in audio spectrum

        Args:
            magnitude: Input magnitude spectrum (freq_bins,)
            phase: Input phase spectrum (freq_bins,)
            strength: Suppression strength in [0, 1] (0=no suppression, 1=full)

        Returns:
            magnitude_out: Suppressed magnitude spectrum
            phase_out: Phase spectrum (unchanged)
        """
        # Preprocess
        mag_tensor = torch.from_numpy(magnitude).unsqueeze(0).to(self.device)

        if self.use_fp16:
            mag_tensor = mag_tensor.half()

        # Normalize (helps with stability)
        mag_mean = mag_tensor.mean()
        mag_std = mag_tensor.std() + 1e-8
        mag_normalized = (mag_tensor - mag_mean) / mag_std

        # Inference
        with torch.no_grad():
            gain_mask, self.hidden = self.model(mag_normalized, self.hidden)

        # Denormalize gain mask
        gain_mask = gain_mask.squeeze(0)

        # Apply suppression strength
        if strength < 1.0:
            gain_mask = 1.0 - strength * (1.0 - gain_mask)

        # Apply gain mask
        mag_suppressed = mag_tensor.squeeze(0) * gain_mask

        # Convert back to numpy
        magnitude_out = mag_suppressed.float().cpu().numpy()

        return magnitude_out, phase

    @torch.no_grad()
    def process_batch(
        self,
        magnitude_batch: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Process batch of magnitude spectra

        Args:
            magnitude_batch: Batch of magnitude spectra (batch, freq_bins)
            strength: Suppression strength

        Returns:
            magnitude_out: Suppressed magnitude spectra
        """
        # Convert to tensor
        mag_tensor = torch.from_numpy(magnitude_batch).to(self.device)

        if self.use_fp16:
            mag_tensor = mag_tensor.half()

        # Normalize
        mag_mean = mag_tensor.mean(dim=1, keepdim=True)
        mag_std = mag_tensor.std(dim=1, keepdim=True) + 1e-8
        mag_normalized = (mag_tensor - mag_mean) / mag_std

        # Inference
        gain_mask, _ = self.model(mag_normalized, None)

        # Apply suppression strength
        if strength < 1.0:
            gain_mask = 1.0 - strength * (1.0 - gain_mask)

        # Apply gain mask
        mag_suppressed = mag_tensor * gain_mask

        # Convert back to numpy
        return mag_suppressed.float().cpu().numpy()

    def save(self, path: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'device': self.device,
            'use_fp16': self.use_fp16
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'NoiseSuppressionModel':
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=device)

        # Create model
        model = NoiseSuppressionRNN()  # TODO: Save architecture info
        model.load_state_dict(checkpoint['model_state_dict'])

        return cls(
            model,
            device=device,
            use_fp16=checkpoint.get('use_fp16', True)
        )


def create_noise_suppression_model(
    freq_bins: int = 257,
    hidden_size: int = 256,
    num_layers: int = 2,
    device: str = 'cuda',
    use_fp16: bool = True
) -> NoiseSuppressionModel:
    """
    Factory function to create noise suppression model

    Args:
        freq_bins: Number of frequency bins
        hidden_size: GRU hidden size
        num_layers: Number of GRU layers
        device: Device to run on
        use_fp16: Use FP16 precision

    Returns:
        NoiseSuppressionModel instance
    """
    model = NoiseSuppressionRNN(
        input_size=freq_bins,
        hidden_size=hidden_size,
        num_layers=num_layers
    )

    return NoiseSuppressionModel(model, device=device, use_fp16=use_fp16)

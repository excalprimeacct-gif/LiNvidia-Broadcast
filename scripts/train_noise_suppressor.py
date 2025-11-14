#!/usr/bin/env python3
"""
Train noise suppression model

This script demonstrates how to train the noise suppression model
on a custom dataset. You'll need clean speech and noise samples.
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from linvidia.models import NoiseSuppressionRNN, NoiseSuppressionModel
from linvidia.audio import AudioProcessor


class NoisyAudioDataset(Dataset):
    """
    Dataset for noise suppression training

    Expects:
    - clean_dir: Directory with clean speech WAV files
    - noise_dir: Directory with noise WAV files
    """

    def __init__(
        self,
        clean_dir: str,
        noise_dir: str,
        processor: AudioProcessor,
        num_samples: int = 10000,
        snr_range: tuple = (-5, 20)
    ):
        self.clean_files = list(Path(clean_dir).glob('*.wav'))
        self.noise_files = list(Path(noise_dir).glob('*.wav'))
        self.processor = processor
        self.num_samples = num_samples
        self.snr_range = snr_range

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # TODO: Implement audio loading and mixing
        # For now, return random data
        freq_bins = self.processor.freq_bins
        clean_mag = np.random.rand(freq_bins).astype(np.float32)
        noisy_mag = clean_mag + np.random.rand(freq_bins).astype(np.float32) * 0.5

        return {
            'noisy': torch.from_numpy(noisy_mag),
            'clean': torch.from_numpy(clean_mag)
        }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    device: str = 'cuda',
    learning_rate: float = 0.001
):
    """
    Train noise suppression model

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on
        learning_rate: Learning rate
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            # Forward pass
            optimizer.zero_grad()
            predicted, _ = model(noisy)

            # Compute loss
            loss = criterion(predicted, clean)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                noisy = batch['noisy'].to(device)
                clean = batch['clean'].to(device)

                predicted, _ = model(noisy)
                loss = criterion(predicted, clean)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/noise_suppression_best.pth')
            print("✓ Saved best model")


def main():
    parser = argparse.ArgumentParser(description='Train noise suppression model')
    parser.add_argument('--clean-dir', required=True, help='Directory with clean speech')
    parser.add_argument('--noise-dir', required=True, help='Directory with noise samples')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create processor
    processor = AudioProcessor()

    # Create datasets
    print("Creating datasets...")
    train_dataset = NoisyAudioDataset(
        args.clean_dir,
        args.noise_dir,
        processor,
        num_samples=8000
    )
    val_dataset = NoisyAudioDataset(
        args.clean_dir,
        args.noise_dir,
        processor,
        num_samples=2000
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    print("Creating model...")
    model = NoiseSuppressionRNN(
        input_size=processor.freq_bins,
        hidden_size=256,
        num_layers=2
    )

    # Train
    print("Starting training...")
    train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        device=device,
        learning_rate=args.lr
    )

    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()

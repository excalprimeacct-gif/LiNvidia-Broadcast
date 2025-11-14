#!/usr/bin/env python3
"""
Train noise suppression model with comprehensive data augmentation and metrics

Supports:
- Custom clean/noise datasets
- DNS Challenge dataset
- Real-time validation metrics (SNR, PESQ, STOI)
- TensorBoard logging
- Model checkpointing
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from linvidia.models import NoiseSuppressionRNN, NoiseSuppressionModel
from linvidia.audio import AudioProcessor
from linvidia.datasets import NoisyAudioDataset, DNSChallengeDataset, AudioAugmentation
from linvidia.metrics import EvaluationMetrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    num_epochs: int
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

    for batch in pbar:
        # Get data
        noisy_mag = batch['noisy_mag'].to(device)  # (batch, seq_len, freq_bins)
        gain_mask_target = batch['gain_mask'].to(device)

        batch_size = noisy_mag.size(0)

        # Forward pass
        optimizer.zero_grad()

        # Process sequence
        predicted_mask, _ = model(noisy_mag)

        # Compute loss
        loss = criterion(predicted_mask, gain_mask_target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update stats
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'avg_loss': f'{total_loss/num_batches:.6f}'
        })

    return total_loss / num_batches


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    processor: AudioProcessor,
    compute_metrics: bool = True
) -> dict:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    metrics = EvaluationMetrics(sample_rate=processor.sample_rate)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")

        for batch in pbar:
            noisy_mag = batch['noisy_mag'].to(device)
            gain_mask_target = batch['gain_mask'].to(device)

            # Forward pass
            predicted_mask, _ = model(noisy_mag)

            # Compute loss
            loss = criterion(predicted_mask, gain_mask_target)

            total_loss += loss.item()
            num_batches += 1

            # Compute audio metrics on first batch
            if compute_metrics and num_batches == 1:
                # Reconstruct audio for metrics
                batch_size = noisy_mag.size(0)
                for i in range(min(batch_size, 5)):  # Evaluate 5 samples
                    # Get one sample
                    noisy_mags = noisy_mag[i].cpu().numpy()
                    clean_mags = batch['clean_mag'][i].cpu().numpy()
                    phases = batch['phase'][i].cpu().numpy()
                    pred_masks = predicted_mask[i].cpu().numpy()

                    # Reconstruct audio
                    noisy_audio = []
                    clean_audio = []
                    enhanced_audio = []

                    for frame_idx in range(len(noisy_mags)):
                        # Noisy
                        noisy_frame = processor.istft(noisy_mags[frame_idx], phases[frame_idx])
                        noisy_audio.extend(noisy_frame[:processor.hop_size])

                        # Clean
                        clean_frame = processor.istft(clean_mags[frame_idx], phases[frame_idx])
                        clean_audio.extend(clean_frame[:processor.hop_size])

                        # Enhanced
                        enhanced_mag = noisy_mags[frame_idx] * pred_masks[frame_idx]
                        enhanced_frame = processor.istft(enhanced_mag, phases[frame_idx])
                        enhanced_audio.extend(enhanced_frame[:processor.hop_size])

                    # Add to metrics
                    metrics.add_sample(
                        np.array(clean_audio),
                        np.array(noisy_audio),
                        np.array(enhanced_audio)
                    )

            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    results = {
        'loss': total_loss / num_batches
    }

    if compute_metrics:
        results.update(metrics.get_summary())

    return results


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: str,
    learning_rate: float,
    output_dir: Path,
    processor: AudioProcessor
):
    """
    Full training loop with checkpointing and metrics

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        device: Device to train on
        learning_rate: Learning rate
        output_dir: Output directory for checkpoints
        processor: Audio processor for metrics
    """
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Loss function
    criterion = nn.MSELoss()

    # Tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, num_epochs
        )
        train_losses.append(train_loss)

        # Validate
        compute_metrics = (epoch % 5 == 0)  # Compute metrics every 5 epochs
        val_results = validate_epoch(
            model, val_loader, criterion, device, processor, compute_metrics
        )
        val_loss = val_results['loss']
        val_losses.append(val_loss)

        # Print results
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")

        if compute_metrics:
            if 'snr_improvement_mean' in val_results:
                print(f"  SNR Improvement: {val_results['snr_improvement_mean']:.2f} dB")
            if 'pesq_mean' in val_results:
                print(f"  PESQ: {val_results['pesq_mean']:.2f}")
            if 'stoi_mean' in val_results:
                print(f"  STOI: {val_results['stoi_mean']:.3f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_results': val_results
        }

        # Save latest
        torch.save(checkpoint, output_dir / 'checkpoint_latest.pth')

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'checkpoint_best.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')

        print()

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train noise suppression model')
    parser.add_argument('--clean-dir', required=True, help='Directory with clean speech')
    parser.add_argument('--noise-dir', required=True, help='Directory with noise samples')
    parser.add_argument('--dns-challenge', action='store_true', help='Use DNS Challenge dataset format')
    parser.add_argument('--output-dir', default='models/training', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-frames', type=int, default=100, help='Frames per sample')
    parser.add_argument('--num-train', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--num-val', type=int, default=1000, help='Number of validation samples')
    parser.add_argument('--hidden-size', type=int, default=256, help='Model hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of GRU layers')
    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create processor
    print("Creating audio processor...")
    processor = AudioProcessor()

    # Create datasets
    print("Creating datasets...")
    if args.dns_challenge:
        train_dataset = DNSChallengeDataset(
            args.clean_dir,
            split='train',
            processor=processor,
            num_frames=args.num_frames,
            num_samples=args.num_train
        )
        val_dataset = DNSChallengeDataset(
            args.noise_dir,
            split='val',
            processor=processor,
            num_frames=args.num_frames,
            num_samples=args.num_val
        )
    else:
        train_dataset = NoisyAudioDataset(
            args.clean_dir,
            args.noise_dir,
            processor=processor,
            num_frames=args.num_frames,
            num_samples=args.num_train
        )
        val_dataset = NoisyAudioDataset(
            args.clean_dir,
            args.noise_dir,
            processor=processor,
            num_frames=args.num_frames,
            num_samples=args.num_val
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True
    )

    # Create model
    print("Creating model...")
    model = NoiseSuppressionRNN(
        input_size=processor.freq_bins,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\nStarting training...")
    train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        device=device,
        learning_rate=args.lr,
        output_dir=output_dir,
        processor=processor
    )

    # Export final model
    print("\nExporting final model...")
    model_wrapper = NoiseSuppressionModel(model, device=device, use_fp16=True)
    model_wrapper.save(output_dir / 'noise_suppression_final.pth')

    print(f"\n✓ Training complete! Models saved to {output_dir}")


if __name__ == '__main__':
    main()

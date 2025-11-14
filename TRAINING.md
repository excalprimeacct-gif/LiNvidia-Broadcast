# Training Guide for LiNvidia Broadcast

This guide explains how to train the noise suppression model on custom datasets.

## Dataset Preparation

### Required Data

You need two types of audio:
1. **Clean speech**: WAV files containing clean speech without noise
2. **Noise**: WAV files containing various types of noise (background, ambient, etc.)

### Dataset Structure

```
dataset/
├── clean/
│   ├── speech1.wav
│   ├── speech2.wav
│   └── ...
└── noise/
    ├── noise1.wav
    ├── noise2.wav
    └── ...
```

### Recommended Datasets

1. **DNS Challenge Dataset** (Recommended)
   - Download: https://github.com/microsoft/DNS-Challenge
   - Contains high-quality clean speech and diverse noise samples
   - ~500GB full dataset, smaller subsets available

2. **LibriSpeech** (Clean speech)
   - Download: https://www.openslr.org/12
   - Clean read speech in English

3. **FSD50K** (Noise samples)
   - Download: https://zenodo.org/record/4060432
   - Diverse environmental sounds

4. **Custom Recording**
   - Record your own clean speech
   - Record various noise environments you want to suppress

## Training Process

### Basic Training

```bash
python scripts/train_noise_suppressor.py \
  --clean-dir /path/to/clean \
  --noise-dir /path/to/noise \
  --epochs 50 \
  --batch-size 16 \
  --output-dir models/training
```

### DNS Challenge Dataset

```bash
python scripts/train_noise_suppressor.py \
  --clean-dir /path/to/dns_challenge \
  --dns-challenge \
  --epochs 100 \
  --batch-size 32 \
  --output-dir models/training_dns
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--clean-dir` | Required | Directory with clean speech WAV files |
| `--noise-dir` | Required | Directory with noise WAV files |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 16 | Batch size (reduce if OOM) |
| `--lr` | 0.001 | Learning rate |
| `--num-frames` | 100 | Frames per training sample |
| `--num-train` | 10000 | Number of training samples per epoch |
| `--num-val` | 1000 | Number of validation samples |
| `--hidden-size` | 256 | GRU hidden size |
| `--num-layers` | 2 | Number of GRU layers |
| `--output-dir` | models/training | Output directory |

### Advanced Configuration

For larger models with better quality:

```bash
python scripts/train_noise_suppressor.py \
  --clean-dir /path/to/clean \
  --noise-dir /path/to/noise \
  --hidden-size 512 \
  --num-layers 3 \
  --batch-size 8 \
  --epochs 100
```

For faster training on smaller datasets:

```bash
python scripts/train_noise_suppressor.py \
  --clean-dir /path/to/clean \
  --noise-dir /path/to/noise \
  --hidden-size 128 \
  --num-layers 2 \
  --batch-size 32 \
  --num-frames 50
```

## Monitoring Training

The training script outputs:

1. **Training loss**: MSE loss on training set
2. **Validation loss**: MSE loss on validation set
3. **SNR improvement**: Signal-to-Noise Ratio improvement in dB (every 5 epochs)
4. **PESQ score**: Perceptual speech quality metric (every 5 epochs)
5. **STOI score**: Speech intelligibility metric (every 5 epochs)

### Good Training Metrics

- **Validation loss**: Should decrease and stabilize
- **SNR improvement**: > 10 dB is good, > 15 dB is excellent
- **PESQ**: > 3.5 is good, > 4.0 is excellent (max 4.5)
- **STOI**: > 0.85 is good, > 0.90 is excellent (max 1.0)

## Model Checkpoints

The training script saves:

- `checkpoint_latest.pth`: Latest model
- `checkpoint_best.pth`: Best model (lowest validation loss)
- `checkpoint_epoch_N.pth`: Periodic checkpoints every 10 epochs
- `noise_suppression_final.pth`: Final exported model

## Using Trained Models

### 1. Load in Python

```python
from linvidia.models import NoiseSuppressionModel

model = NoiseSuppressionModel.load('models/training/noise_suppression_final.pth')
```

### 2. Convert to TensorRT

```bash
python -m linvidia.cli create-model \
  --output-dir models \
  --tensorrt
```

### 3. Use in Real-time

Update your config to use the trained model:

```yaml
# config/custom.yaml
inference:
  engine_path: models/noise_suppression.engine
```

Then run:

```bash
python -m linvidia.cli noise-suppression --config config/custom.yaml
```

## Data Augmentation

The training pipeline includes automatic augmentation:

- **SNR variation**: Random SNR from -5 dB to +20 dB
- **Reverb**: Simulated room acoustics (optional)
- **EQ**: Random frequency response variations (optional)
- **Dynamic compression**: Random compression settings (optional)

## Tips for Best Results

1. **Diverse data**: Use diverse noise types and speakers
2. **Balanced SNRs**: Include both low and high SNR examples
3. **Enough data**: At least 10,000 training samples recommended
4. **Regular validation**: Monitor metrics during training
5. **Early stopping**: Stop if validation loss stops improving
6. **Fine-tuning**: Start from pre-trained model if available

## Troubleshooting

### Out of Memory (OOM)

- Reduce `--batch-size`
- Reduce `--num-frames`
- Reduce `--hidden-size`
- Use gradient checkpointing (advanced)

### Poor Performance

- Train longer (more epochs)
- Use larger model (`--hidden-size 512`)
- Use more diverse training data
- Check data quality (sample rate, clipping, etc.)

### Overfitting

- Reduce model size
- Add more training data
- Increase regularization (weight decay)
- Use more aggressive augmentation

## Example: Training on DNS Challenge

```bash
# 1. Download DNS Challenge dataset
# https://github.com/microsoft/DNS-Challenge

# 2. Extract to /data/dns_challenge

# 3. Train model
python scripts/train_noise_suppressor.py \
  --clean-dir /data/dns_challenge \
  --dns-challenge \
  --epochs 100 \
  --batch-size 32 \
  --hidden-size 256 \
  --num-layers 2 \
  --output-dir models/dns_trained

# 4. Monitor training
# Training will print metrics every epoch

# 5. Use best model
cp models/dns_trained/checkpoint_best.pth models/noise_suppression_final.pth

# 6. Convert to TensorRT (optional, for better performance)
python -m linvidia.cli create-model \
  --output-dir models \
  --tensorrt
```

## Advanced: Custom Loss Functions

You can modify the training script to use custom loss functions:

```python
# In train_noise_suppressor.py
# Replace MSELoss with custom loss:

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # MSE loss
        mse_loss = self.mse(pred, target)

        # Add perceptual loss
        # ... (implement custom metrics)

        return mse_loss

criterion = PerceptualLoss()
```

## Next Steps

After training:

1. **Evaluate**: Test on real recordings
2. **Optimize**: Convert to TensorRT for speed
3. **Deploy**: Use in production with your config
4. **Iterate**: Collect failure cases and retrain

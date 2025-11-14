# LiNvidia Broadcast - Linux NVIDIA Broadcast Alternative

Real-time AI-powered audio and video effects for Linux using NVIDIA tensor cores.

## Overview

LiNvidia Broadcast is an open-source implementation of NVIDIA Broadcast features for Linux systems, leveraging NVIDIA GPUs with tensor cores for real-time AI processing.

### Current Features

#### 🎤 Audio Features
- **Noise Suppression**: Real-time audio noise removal using deep learning
  - Low latency (<10ms target)
  - Optimized for NVIDIA tensor cores via TensorRT
  - Support for PulseAudio and PipeWire
  - Adjustable suppression strength
  - Virtual microphone support

#### 📹 Video Features
- **Background Blur**: Blur background with adjustable strength
  - Bokeh-style depth-of-field effect
  - Adjustable blur strength (1-100)
  - Real-time performance (30fps @ 720p)

- **Background Removal**: Remove background with green screen effect
  - Perfect for streaming and video production
  - Clean chroma key output

- **Background Replacement**: Replace background with custom images
  - Use any image as your virtual background
  - Automatic scaling and blending
  - Professional meeting ready

- **Auto-Framing**: Intelligent automatic framing with face tracking
  - AI-powered face detection using MediaPipe
  - Multi-person tracking with temporal smoothing
  - 6 framing modes: Center, Headroom, Tight, Wide, Group, and Off
  - Smooth transitions with smart cropping
  - Professional-quality framing rules (Rule of Thirds)

- **Virtual Camera**: v4l2loopback integration
  - Works with Zoom, Teams, Discord, OBS, and more
  - System-wide camera effects

### Planned Features

- Eye contact correction
- Advanced lighting effects
- Portrait relighting

## Requirements

### Hardware
- NVIDIA GPU with tensor cores (RTX 20xx series or newer recommended)
- CUDA Compute Capability 7.0+

### Software
- Linux (tested on Ubuntu 20.04+)
- NVIDIA GPU drivers (>=470.0)
- CUDA Toolkit (>=11.0)
- TensorRT (>=8.0)
- Python 3.8+

## Architecture

### Noise Suppression Pipeline

```
Audio Input → Framing → STFT → Neural Network (TensorRT) → ISTFT → Audio Output
                ↓                       ↓
            Windowing            Tensor Core Acceleration
```

The noise suppression system uses:
1. **Audio Processing**: Short-time Fourier transform (STFT) for frequency domain processing
2. **Neural Network**: RNN-based model (similar to RNNoise) for noise classification
3. **TensorRT**: Optimized inference engine utilizing tensor cores for maximum throughput
4. **Low-latency pipeline**: Circular buffers and async processing for <10ms latency

## Installation

```bash
# Clone the repository
git clone https://github.com/hellasleeper108/LiNvidia-Broadcast.git
cd LiNvidia-Broadcast

# Install Python dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py
```

## Usage

### Noise Suppression

```bash
# Run with default settings
linvidia noise-suppression

# Specify input/output devices
linvidia noise-suppression --input "device_name" --output "device_name"

# Adjust suppression strength
linvidia noise-suppression --strength 0.95
```

### Background Effects

```bash
# Blur background (default)
linvidia background-effects --effect blur --blur-strength 50

# Remove background (green screen)
linvidia background-effects --effect remove

# Replace background with image
linvidia background-effects --effect replace --background /path/to/image.jpg

# Enable auto-framing with professional headroom mode
linvidia background-effects --effect blur --auto-frame --framing-mode headroom

# Combine effects: blur + auto-framing to virtual camera
linvidia background-effects --effect blur --blur-strength 40 \
  --auto-frame --framing-mode center --output virtual

# Group framing for multiple people
linvidia background-effects --auto-frame --framing-mode group
```

### GUI Application

```bash
# Launch full-featured GUI with both audio and video
linvidia gui --full

# Or just audio GUI
linvidia gui
```

## Development

### Project Structure

```
linvidia/
├── audio/              # Audio I/O and processing
│   ├── capture.py     # Audio input handling
│   ├── playback.py    # Audio output handling
│   └── processing.py  # STFT, windowing, framing
├── video/             # Video I/O and processing
│   ├── capture.py     # Camera input handling
│   ├── display.py     # Video output and virtual camera
│   └── processing.py  # Video effects processing
├── models/            # Neural network models
│   ├── noise_suppression.py
│   └── segmentation/  # Background segmentation models
├── tracking/          # Face detection and auto-framing
│   ├── face_detector.py   # MediaPipe/OpenCV face detection
│   ├── face_tracker.py    # Multi-object tracking
│   └── auto_frame.py      # Auto-framing logic
├── inference/         # TensorRT inference engine
│   ├── engine.py
│   └── optimizer.py
├── ui/               # User interface
│   ├── cli.py
│   ├── gui.py
│   └── gui_full.py   # Full-featured GUI with video
└── utils/            # Utilities and helpers
```

### Training Custom Models

```bash
# Train noise suppression model
python scripts/train_noise_suppressor.py --dataset path/to/dataset
```

## Technical Details

### Noise Suppression Model

The noise suppression model is based on recurrent neural networks (RNN) with:
- GRU layers for temporal modeling
- Frequency masking for noise removal
- Mixed precision (FP16) for tensor core utilization
- Optimized for real-time inference

### TensorRT Optimization

- FP16 precision for 2x throughput on tensor cores
- Layer fusion and kernel auto-tuning
- Dynamic batch size support
- CUDA graph optimization for minimal overhead

## Performance

Target metrics on RTX 3060:
- **Audio**: <8ms latency (capture to playback)
- **Video**: ~35ms total latency with all effects enabled
- **FPS**: 28-30 fps @ 720p with background effects + auto-framing
- **Quality**: PESQ >4.0, SNR improvement >15dB (audio)

## Documentation

- **[INSTALL.md](INSTALL.md)**: Detailed installation instructions and system setup
- **[VIDEO_FEATURES.md](VIDEO_FEATURES.md)**: Complete guide to video features and background effects
- **[AUTO_FRAMING.md](AUTO_FRAMING.md)**: Auto-framing modes, usage, and troubleshooting
- **[TRAINING.md](TRAINING.md)**: Training custom models and datasets

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by NVIDIA Broadcast
- RNNoise paper by Jean-Marc Valin
- Open source audio processing community

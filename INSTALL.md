# Installation Guide

This guide will help you install LiNvidia Broadcast and its dependencies.

## System Requirements

### Hardware Requirements

- **NVIDIA GPU**: RTX 20xx series or newer (requires tensor cores)
  - Compute Capability 7.0 or higher
  - At least 4GB VRAM
  - Recommended: RTX 3060 or better

### Software Requirements

- **Operating System**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.8 or newer
- **NVIDIA Drivers**: 470.0 or newer
- **CUDA**: 11.0 or newer
- **TensorRT**: 8.0 or newer (optional but recommended)
- **Audio System**: PulseAudio or PipeWire

## Quick Installation

### Automated Installation

```bash
# Clone the repository
git clone https://github.com/hellasleeper108/LiNvidia-Broadcast.git
cd LiNvidia-Broadcast

# Run installation script
./scripts/install.sh
```

The installation script will:
1. Check prerequisites
2. Create Python virtual environment
3. Install dependencies
4. Set up configuration
5. Optionally create PulseAudio virtual devices
6. Optionally create initial model

## Manual Installation

### 1. Install System Dependencies

#### Ubuntu/Debian

```bash
# NVIDIA drivers and CUDA
sudo apt update
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit

# Audio development files
sudo apt install portaudio19-dev python3-pyaudio pulseaudio

# Video dependencies
sudo apt install v4l2loopback-dkms v4l-utils

# Python development
sudo apt install python3-dev python3-pip python3-venv
```

#### Arch Linux

```bash
# NVIDIA drivers and CUDA
sudo pacman -S nvidia nvidia-utils cuda

# Audio
sudo pacman -S portaudio python-pyaudio pulseaudio

# Video
sudo pacman -S v4l2loopback-dkms v4l-utils

# Python
sudo pacman -S python python-pip
```

### 2. Install TensorRT (Optional but Recommended)

Download from: https://developer.nvidia.com/tensorrt

```bash
# Extract TensorRT
tar -xzvf TensorRT-8.x.x.tar.gz

# Install Python package
cd TensorRT-8.x.x/python
pip install tensorrt-*-cp3x-*.whl

# Install PyCUDA
pip install pycuda
```

### 3. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python Dependencies

```bash
# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install optional dependencies
pip install pesq pystoi  # Audio quality metrics
pip install PyQt6        # GUI application
```

### 5. Install LiNvidia Broadcast

```bash
# Development installation
pip install -e .

# Or regular installation
pip install .
```

### 6. Verify Installation

```bash
# Check system requirements
linvidia check-system

# List audio devices
linvidia list-devices
```

Expected output:
```
=== System Check ===

✓ CUDA available
  Device: NVIDIA GeForce RTX 3060
  CUDA version: 11.8
  Compute capability: (8, 6)
✓ Tensor cores available
✓ TensorRT available (version 8.6.0)
✓ Audio backend available
  12 input devices
  15 output devices
```

## Post-Installation

### Create Initial Model

```bash
# Create and download initial model
python scripts/download_models.py
```

This creates an untrained model that can be used immediately or fine-tuned on your data.

### Setup PulseAudio Virtual Devices (Optional)

For system-wide noise suppression:

```bash
# Create virtual devices
./scripts/pulseaudio/setup_virtual_device.sh

# To remove
./scripts/pulseaudio/cleanup_virtual_device.sh
```

### Setup Virtual Camera for Video Effects (Optional)

For system-wide video effects with Zoom, Teams, Discord, etc.:

```bash
# Load v4l2loopback module
sudo modprobe v4l2loopback video_nr=2 card_label='LiNvidia_Broadcast' exclusive_caps=1

# Make it persistent (load on boot)
echo "v4l2loopback" | sudo tee -a /etc/modules

# Configure module options
echo "options v4l2loopback video_nr=2 card_label='LiNvidia_Broadcast' exclusive_caps=1" | \
    sudo tee /etc/modprobe.d/v4l2loopback.conf

# Verify virtual camera exists
ls -l /dev/video*
```

Or use the built-in command:

```bash
linvidia setup-virtual-camera --device-id 2
```

### Create Configuration

```bash
# Create default configuration
linvidia create-config --output ~/.config/linvidia/config.yaml

# Edit configuration
nano ~/.config/linvidia/config.yaml
```

## Troubleshooting

### CUDA Not Found

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### TensorRT Not Found

```bash
# Install TensorRT via pip (if available)
pip install nvidia-tensorrt

# Or download from NVIDIA website
# https://developer.nvidia.com/tensorrt
```

### Audio Device Issues

```bash
# Check PulseAudio
pactl info

# Restart PulseAudio
pulseaudio --kill
pulseaudio --start

# Check devices
pactl list sources short
pactl list sinks short
```

### Import Errors

```bash
# Reinstall in virtual environment
source venv/bin/activate
pip install --force-reinstall -e .
```

### Permission Errors

```bash
# Add user to audio and video groups
sudo usermod -a -G audio,video $USER

# Log out and log back in
```

### Virtual Camera Not Working

```bash
# Check if v4l2loopback is loaded
lsmod | grep v4l2loopback

# Load the module
sudo modprobe v4l2loopback video_nr=2 card_label='LiNvidia_Broadcast' exclusive_caps=1

# Check available video devices
v4l2-ctl --list-devices

# Verify permissions
ls -l /dev/video*
```

### MediaPipe Installation Issues

If you encounter issues with MediaPipe (for auto-framing):

```bash
# Try installing with specific version
pip install mediapipe==0.10.9

# On some systems, you may need system dependencies
sudo apt install ffmpeg libsm6 libxext6
```

## Docker Installation (Alternative)

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    portaudio19-dev \
    pulseaudio \
    git

# Clone repository
RUN git clone https://github.com/hellasleeper108/LiNvidia-Broadcast.git
WORKDIR /LiNvidia-Broadcast

# Install Python dependencies
RUN pip3 install -r requirements.txt
RUN pip3 install -e .

CMD ["linvidia", "check-system"]
```

Build and run:

```bash
docker build -t linvidia-broadcast .
docker run --gpus all -it linvidia-broadcast
```

## Uninstallation

```bash
# Remove virtual environment
rm -rf venv

# Remove configuration
rm -rf ~/.config/linvidia

# Remove PulseAudio virtual devices
./scripts/pulseaudio/cleanup_virtual_device.sh

# Uninstall package
pip uninstall linvidia-broadcast
```

## Next Steps

After installation:

1. **Test Audio**: Run `python scripts/test_audio.py` to verify audio I/O
2. **Test Video**: Run `linvidia background-effects --effect blur` to test video
3. **Try Auto-Framing**: Run `linvidia background-effects --auto-frame --framing-mode center`
4. **Setup Virtual Camera**: Follow the virtual camera setup above for use with Zoom/Teams
5. **Train Model**: Follow [TRAINING.md](TRAINING.md) to train custom models
6. **Run GUI**: Execute `linvidia gui --full` for graphical interface
7. **Configure**: Customize settings in `~/.config/linvidia/config.yaml`
8. **Read Docs**: Check [VIDEO_FEATURES.md](VIDEO_FEATURES.md) and [AUTO_FRAMING.md](AUTO_FRAMING.md)

## Getting Help

- **Documentation**: See README.md and other docs
- **Issues**: https://github.com/hellasleeper108/LiNvidia-Broadcast/issues
- **Discussions**: https://github.com/hellasleeper108/LiNvidia-Broadcast/discussions

#!/bin/bash
#
# LiNvidia Broadcast Installation Script
#
# This script installs LiNvidia Broadcast and its dependencies
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}→${NC} $1"
}

# Check if running as root (we don't want this)
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    exit 1
fi

echo "======================================"
echo "  LiNvidia Broadcast Installation"
echo "======================================"
echo

# Check prerequisites
print_info "Checking prerequisites..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Python $PYTHON_VERSION found"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU driver found"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1
else
    print_error "NVIDIA GPU driver not found"
    echo "Please install NVIDIA drivers before continuing"
    exit 1
fi

# Check audio system
if command -v pactl &> /dev/null; then
    print_success "PulseAudio found"
elif command -v pipewire &> /dev/null; then
    print_success "PipeWire found"
else
    print_error "No supported audio system found (PulseAudio or PipeWire required)"
    exit 1
fi

echo

# Install Python package
print_info "Installing LiNvidia Broadcast Python package..."

# Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Created virtual environment"
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip > /dev/null 2>&1

# Install package in development mode
pip install -e . || {
    print_error "Failed to install Python package"
    exit 1
}

print_success "Python package installed"

# Install optional dependencies
print_info "Installing optional dependencies..."

# Try to install PESQ and STOI
pip install pesq pystoi 2>/dev/null && print_success "Installed audio metrics packages" || \
    print_info "Could not install audio metrics (optional)"

echo

# Setup configuration
print_info "Setting up configuration..."

CONFIG_DIR="$HOME/.config/linvidia"
mkdir -p "$CONFIG_DIR"

if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    cp config/default.yaml "$CONFIG_DIR/config.yaml"
    print_success "Created default configuration"
else
    print_info "Configuration already exists, skipping"
fi

echo

# Setup PulseAudio virtual devices (optional)
read -p "Setup PulseAudio virtual devices for system-wide noise suppression? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Setting up PulseAudio virtual devices..."
    ./scripts/pulseaudio/setup_virtual_device.sh
    print_success "PulseAudio virtual devices created"

    # Create cleanup script in user's bin
    mkdir -p "$HOME/.local/bin"
    cp scripts/pulseaudio/setup_virtual_device.sh "$HOME/.local/bin/linvidia-setup-pulseaudio"
    cp scripts/pulseaudio/cleanup_virtual_device.sh "$HOME/.local/bin/linvidia-cleanup-pulseaudio"
    chmod +x "$HOME/.local/bin/linvidia-setup-pulseaudio"
    chmod +x "$HOME/.local/bin/linvidia-cleanup-pulseaudio"
    print_success "Installed PulseAudio management scripts to ~/.local/bin"
fi

echo

# Create models directory
print_info "Creating models directory..."
mkdir -p models
print_success "Models directory created"

echo

# Offer to download/create initial model
read -p "Create initial noise suppression model? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Creating initial model (this may take a moment)..."
    python3 scripts/download_models.py --output-dir models || {
        print_error "Model creation failed (you may need to install TensorRT)"
        print_info "You can still use the PyTorch model"
    }
fi

echo

# Installation complete
echo "======================================"
print_success "Installation complete!"
echo "======================================"
echo
echo "Quick start:"
echo "  1. List audio devices:"
echo "     python -m linvidia.cli list-devices"
echo
echo "  2. Check system requirements:"
echo "     python -m linvidia.cli check-system"
echo
echo "  3. Run noise suppression:"
echo "     python -m linvidia.cli noise-suppression"
echo
echo "  4. For system-wide integration (if you set up PulseAudio):"
echo "     a. Run: python -m linvidia.cli noise-suppression --output linvidia_sink"
echo "     b. In your apps, select 'LiNvidia_Broadcast_Microphone' as input"
echo
echo "Documentation: https://github.com/hellasleeper108/LiNvidia-Broadcast"
echo

# Offer to run test
read -p "Run audio passthrough test? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 scripts/test_audio.py --duration 5
fi

echo
print_success "All done! Enjoy LiNvidia Broadcast!"

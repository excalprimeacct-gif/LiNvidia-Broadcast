#!/usr/bin/env bash
# LiNvidia Broadcast - native installer for CachyOS / Arch Linux
#
# This script:
#   1. Installs system packages from the official repos.
#   2. Builds & installs the linvidia-broadcast pacman package via makepkg.
#   3. Installs a desktop launcher and icon so the GUI shows up in your
#      application menu (KDE / GNOME / Hyprland-rofi all pick it up).
#   4. Optionally loads the v4l2loopback kernel module for the virtual camera.
#
# Run as a regular user (NOT root). It will use sudo when needed.

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
info()  { echo -e "${BLUE}[..]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
fail()  { echo -e "${RED}[fail]${NC}  $*"; exit 1; }

if [[ $EUID -eq 0 ]]; then
    fail "Do not run this script as root. It will call sudo when needed."
fi

if ! command -v pacman >/dev/null 2>&1; then
    fail "This script targets CachyOS / Arch Linux (pacman not found)."
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================"
echo "  LiNvidia Broadcast - CachyOS installer"
echo "========================================"
echo "Repo: $REPO_ROOT"
echo

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
info "Installing system packages with pacman..."

PACMAN_PKGS=(
    base-devel
    git
    python
    python-pip
    python-numpy
    python-scipy
    python-yaml
    python-click
    python-pyqt6
    python-sounddevice
    python-pyaudio
    python-opencv
    portaudio
    pulseaudio
    pipewire
    v4l-utils
    v4l2loopback-dkms
    hicolor-icon-theme
)

# pytorch-cuda is in CachyOS/Arch extra repos; skip silently if unavailable.
if pacman -Si python-pytorch-cuda >/dev/null 2>&1; then
    PACMAN_PKGS+=(python-pytorch-cuda python-torchvision)
elif pacman -Si python-pytorch >/dev/null 2>&1; then
    warn "python-pytorch-cuda not found, falling back to CPU python-pytorch."
    PACMAN_PKGS+=(python-pytorch python-torchvision)
fi

sudo pacman -S --needed --noconfirm "${PACMAN_PKGS[@]}"
ok "System packages installed"

# Optional NVIDIA stack
if command -v nvidia-smi >/dev/null 2>&1; then
    ok "NVIDIA driver detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    warn "nvidia-smi not found. CPU fallback will be used. For GPU acceleration:"
    warn "    sudo pacman -S nvidia-dkms nvidia-utils cuda"
fi

# ---------------------------------------------------------------------------
# 2. Build & install pacman package
# ---------------------------------------------------------------------------
info "Building linvidia-broadcast pacman package via makepkg..."

PKGBUILD_DIR="$REPO_ROOT/packaging/cachyos"
[[ -f "$PKGBUILD_DIR/PKGBUILD" ]] || fail "PKGBUILD not found at $PKGBUILD_DIR"

pushd "$PKGBUILD_DIR" >/dev/null
# Clean previous builds
rm -rf src pkg ./*.pkg.tar.* ./*.zst 2>/dev/null || true

# -s: install missing build deps, -i: install built package, -f: force rebuild
makepkg -sif --noconfirm
popd >/dev/null
ok "Pacman package installed"

# ---------------------------------------------------------------------------
# 3. Refresh desktop & icon caches so the launcher shows up immediately
# ---------------------------------------------------------------------------
info "Refreshing desktop database..."
sudo update-desktop-database -q /usr/share/applications 2>/dev/null || true
sudo gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor 2>/dev/null || true
ok "Desktop cache refreshed"

# ---------------------------------------------------------------------------
# 4. Virtual camera kernel module
# ---------------------------------------------------------------------------
if ! lsmod | grep -q v4l2loopback; then
    info "Loading v4l2loopback kernel module (virtual camera)..."
    sudo modprobe v4l2loopback video_nr=10 card_label=LiNvidia_Broadcast exclusive_caps=1 \
        || warn "Could not load v4l2loopback. Reboot may be required after dkms build."
else
    ok "v4l2loopback already loaded"
fi

# ---------------------------------------------------------------------------
# 5. Per-user config seed
# ---------------------------------------------------------------------------
USER_CFG="$HOME/.config/linvidia/config.yaml"
if [[ ! -f "$USER_CFG" ]] && [[ -f /usr/share/linvidia-broadcast/default.yaml ]]; then
    install -Dm644 /usr/share/linvidia-broadcast/default.yaml "$USER_CFG"
    ok "Created user config at $USER_CFG"
fi

echo
echo "========================================"
ok "Installation complete!"
echo "========================================"
echo
echo "Launch the GUI:"
echo "    linvidia-gui              # full GUI (audio + video)"
echo "    linvidia-gui-simple       # audio-only GUI"
echo "    linvidia --help           # CLI"
echo
echo "Or open the application launcher and search for 'LiNvidia Broadcast'."
echo
echo "Uninstall:"
echo "    sudo pacman -R linvidia-broadcast"
echo

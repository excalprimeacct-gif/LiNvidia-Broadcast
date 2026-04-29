#!/usr/bin/env bash
# LiNvidia Broadcast - native installer for CachyOS / Arch Linux
#
# This script:
#   1. Installs system packages from the official repos (skipping any that
#      aren't packaged so the script never aborts on a missing target).
#   2. Pulls AUR-only packages (sounddevice, pyaudio, v4l2loopback-dkms,
#      pytorch-cuda) via paru/yay if you have one. If you don't, those
#      packages are installed via pip into a per-user virtualenv instead.
#   3. Pip-installs anything left over from requirements.txt into the same
#      venv (PEP 668: Arch marks the system Python as "externally managed",
#      so we use a venv to stay polite).
#   4. Builds & installs the linvidia-broadcast pacman package via makepkg.
#   5. Installs a desktop launcher and icon so the GUI shows up in your
#      application menu.
#   6. Loads the v4l2loopback kernel module for the virtual camera.
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

VENV_DIR="${LINVIDIA_VENV:-$HOME/.local/share/linvidia-broadcast/venv}"

echo "========================================"
echo "  LiNvidia Broadcast - CachyOS installer"
echo "========================================"
echo "Repo: $REPO_ROOT"
echo "Venv: $VENV_DIR"
echo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Returns 0 if the package exists in any synced pacman repository.
pkg_in_repo() {
    pacman -Si "$1" >/dev/null 2>&1
}

# Returns 0 if pacman has the package installed locally.
pkg_installed() {
    pacman -Qi "$1" >/dev/null 2>&1
}

detect_aur_helper() {
    for h in paru yay pikaur trizen; do
        if command -v "$h" >/dev/null 2>&1; then
            echo "$h"
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# 1. Refresh package databases
# ---------------------------------------------------------------------------
info "Refreshing pacman databases..."
sudo pacman -Sy --noconfirm >/dev/null
ok "Databases refreshed"

# ---------------------------------------------------------------------------
# 2. System packages from official repos
# ---------------------------------------------------------------------------
# Wishlist of every package we'd like from pacman. Anything missing locally
# is filtered out; anything that's not in the user's repos at all gets a
# warning and falls through to pip later.
WISHLIST=(
    base-devel
    git
    python
    python-pip
    python-virtualenv
    python-numpy
    python-scipy
    python-yaml
    python-click
    python-pyqt6
    python-opencv
    python-pytorch
    python-pytorch-cuda
    python-torchvision
    python-sounddevice
    python-pyaudio
    python-mediapipe
    python-loguru
    python-pyyaml
    python-tqdm
    python-psutil
    python-colorama
    python-dotenv
    portaudio
    pipewire
    pipewire-pulse
    pulseaudio-alsa
    v4l-utils
    v4l2loopback-dkms
    hicolor-icon-theme
)

PACMAN_INSTALL=()
MISSING_FROM_REPOS=()

info "Resolving available system packages..."
for p in "${WISHLIST[@]}"; do
    if pkg_installed "$p"; then
        continue
    fi
    if pkg_in_repo "$p"; then
        PACMAN_INSTALL+=("$p")
    else
        MISSING_FROM_REPOS+=("$p")
    fi
done

if (( ${#PACMAN_INSTALL[@]} > 0 )); then
    info "Installing: ${PACMAN_INSTALL[*]}"
    sudo pacman -S --needed --noconfirm "${PACMAN_INSTALL[@]}"
    ok "Pacman packages installed"
else
    ok "All available pacman packages already present"
fi

# ---------------------------------------------------------------------------
# 3. AUR packages (only what we still need)
# ---------------------------------------------------------------------------
# We try AUR first for anything missing from the official repos. If no AUR
# helper exists, we'll silently fall through to pip.
AUR_CANDIDATES=()
for p in "${MISSING_FROM_REPOS[@]}"; do
    case "$p" in
        python-sounddevice|python-pyaudio|v4l2loopback-dkms|python-pytorch-cuda|python-mediapipe|python-loguru|python-tqdm)
            AUR_CANDIDATES+=("$p")
            ;;
    esac
done

if (( ${#AUR_CANDIDATES[@]} > 0 )); then
    if AUR_HELPER="$(detect_aur_helper)"; then
        info "Installing from AUR via $AUR_HELPER: ${AUR_CANDIDATES[*]}"
        "$AUR_HELPER" -S --needed --noconfirm "${AUR_CANDIDATES[@]}" || \
            warn "AUR helper failed for some packages; will retry via pip."
    else
        warn "No AUR helper (paru/yay) found. Skipping AUR step and using pip instead."
    fi
fi

# ---------------------------------------------------------------------------
# 4. Per-user virtualenv for the remaining Python deps
# ---------------------------------------------------------------------------
# Arch's system Python is PEP 668 "externally managed", so we put any pip
# packages we still need into a dedicated venv. The pacman package's
# wrapper scripts will respect this if it's set up.
info "Creating per-user virtualenv at $VENV_DIR..."
mkdir -p "$(dirname "$VENV_DIR")"
if [[ ! -d "$VENV_DIR" ]]; then
    python -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null
ok "Virtualenv ready"

# Install what's listed in requirements.txt; --upgrade only if missing.
info "Installing remaining Python dependencies via pip..."
PIP_REQS=(
    numpy
    scipy
    pyyaml
    click
    sounddevice
    pyaudio
    opencv-python
    loguru
    psutil
    colorama
    tqdm
    python-dotenv
    PyQt6
)
# Heavy / GPU bits — best effort, never fatal.
PIP_OPTIONAL=(
    torch
    torchvision
    torchaudio
    mediapipe
    onnx
    onnxruntime-gpu
    pesq
    pystoi
    librosa
    resampy
    soundfile
    pulsectl
)

python -m pip install --upgrade "${PIP_REQS[@]}" || \
    warn "Some required pip packages failed to install."

for p in "${PIP_OPTIONAL[@]}"; do
    python -m pip install --upgrade "$p" 2>/dev/null && ok "pip: $p" || \
        warn "pip skipped (optional): $p"
done

deactivate
ok "Python dependencies installed"

# ---------------------------------------------------------------------------
# 5. NVIDIA driver detection
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    ok "NVIDIA driver detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    warn "nvidia-smi not found. CPU fallback will be used. For GPU acceleration:"
    warn "    sudo pacman -S nvidia-dkms nvidia-utils cuda"
fi

# ---------------------------------------------------------------------------
# 6. Build & install pacman package
# ---------------------------------------------------------------------------
info "Building linvidia-broadcast pacman package via makepkg..."

PKGBUILD_DIR="$REPO_ROOT/packaging/cachyos"
[[ -f "$PKGBUILD_DIR/PKGBUILD" ]] || fail "PKGBUILD not found at $PKGBUILD_DIR"

pushd "$PKGBUILD_DIR" >/dev/null
rm -rf src pkg ./*.pkg.tar.* ./*.zst 2>/dev/null || true
# -s: install missing build deps, -i: install built package, -f: force rebuild
makepkg -sif --noconfirm
popd >/dev/null
ok "Pacman package installed"

# Wire the pacman-installed app to the venv we built so its imports resolve.
# /usr/bin/linvidia-gui is created by the wheel; we shim the venv onto its
# PYTHONPATH via /etc/profile.d so every shell session picks it up.
SHIM=/etc/profile.d/linvidia-broadcast.sh
sudo install -Dm644 /dev/stdin "$SHIM" <<EOF
# Added by linvidia-broadcast installer
if [ -d "$VENV_DIR/lib" ]; then
    for _site in "$VENV_DIR"/lib/python*/site-packages; do
        case ":\${PYTHONPATH:-}:" in
            *":\$_site:"*) ;;
            *) PYTHONPATH="\${PYTHONPATH:+\$PYTHONPATH:}\$_site" ;;
        esac
    done
    export PYTHONPATH
    unset _site
fi
EOF
ok "Wired venv site-packages onto PYTHONPATH (login shells)."

# ---------------------------------------------------------------------------
# 7. Refresh desktop & icon caches
# ---------------------------------------------------------------------------
info "Refreshing desktop database..."
sudo update-desktop-database -q /usr/share/applications 2>/dev/null || true
sudo gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor 2>/dev/null || true
ok "Desktop cache refreshed"

# ---------------------------------------------------------------------------
# 8. Virtual camera kernel module
# ---------------------------------------------------------------------------
if ! lsmod | grep -q v4l2loopback; then
    info "Loading v4l2loopback kernel module (virtual camera)..."
    sudo modprobe v4l2loopback video_nr=10 card_label=LiNvidia_Broadcast exclusive_caps=1 \
        || warn "Could not load v4l2loopback. A reboot may be required after dkms build."
else
    ok "v4l2loopback already loaded"
fi

# ---------------------------------------------------------------------------
# 9. Per-user config seed
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
echo "If imports fail, open a NEW shell (so $SHIM is sourced) or run:"
echo "    source $VENV_DIR/bin/activate && linvidia-gui"
echo
echo "Uninstall:"
echo "    sudo pacman -R linvidia-broadcast"
echo "    sudo rm -f $SHIM"
echo "    rm -rf $VENV_DIR"
echo

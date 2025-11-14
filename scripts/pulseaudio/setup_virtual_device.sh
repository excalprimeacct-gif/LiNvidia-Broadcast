#!/bin/bash
#
# Setup PulseAudio virtual devices for LiNvidia Broadcast
#
# This script creates:
# 1. Virtual microphone sink (where processed audio goes)
# 2. Virtual microphone source (what applications see)
# 3. Null sink for routing
#

set -e

SINK_NAME="linvidia_sink"
SOURCE_NAME="linvidia_source"
MONITOR_NAME="${SINK_NAME}.monitor"

echo "=== LiNvidia Broadcast PulseAudio Setup ==="
echo

# Check if PulseAudio is running
if ! pgrep -x "pulseaudio" > /dev/null; then
    echo "Error: PulseAudio is not running"
    exit 1
fi

echo "Creating virtual devices..."

# Create null sink (this will have a monitor source)
pactl load-module module-null-sink \
    sink_name="${SINK_NAME}" \
    sink_properties=device.description="LiNvidia_Broadcast_Sink" \
    rate=48000 \
    channels=1

# Create virtual source (remap from the null sink's monitor)
pactl load-module module-remap-source \
    master="${MONITOR_NAME}" \
    source_name="${SOURCE_NAME}" \
    source_properties=device.description="LiNvidia_Broadcast_Microphone" \
    channels=1

echo
echo "✓ Virtual devices created successfully!"
echo
echo "Devices:"
echo "  Sink:   ${SINK_NAME}"
echo "  Source: ${SOURCE_NAME}"
echo
echo "Usage:"
echo "  1. Run LiNvidia Broadcast with:"
echo "     python -m linvidia.cli noise-suppression \\"
echo "       --input \"your_real_mic\" \\"
echo "       --output \"${SINK_NAME}\""
echo
echo "  2. In your applications (Zoom, Discord, etc), select:"
echo "     Microphone: ${SOURCE_NAME}"
echo
echo "To remove virtual devices, run: ./cleanup_virtual_device.sh"
echo

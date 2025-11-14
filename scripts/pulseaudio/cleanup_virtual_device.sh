#!/bin/bash
#
# Cleanup PulseAudio virtual devices created for LiNvidia Broadcast
#

set -e

SINK_NAME="linvidia_sink"
SOURCE_NAME="linvidia_source"

echo "=== LiNvidia Broadcast PulseAudio Cleanup ==="
echo

# Find and unload modules
echo "Removing virtual devices..."

# Get module IDs
SINK_MODULE=$(pactl list modules short | grep "module-null-sink.*${SINK_NAME}" | awk '{print $1}' || true)
SOURCE_MODULE=$(pactl list modules short | grep "module-remap-source.*${SOURCE_NAME}" | awk '{print $1}' || true)

# Unload modules
if [ -n "$SOURCE_MODULE" ]; then
    pactl unload-module "$SOURCE_MODULE"
    echo "✓ Removed virtual source"
fi

if [ -n "$SINK_MODULE" ]; then
    pactl unload-module "$SINK_MODULE"
    echo "✓ Removed virtual sink"
fi

echo
echo "Cleanup complete!"
echo

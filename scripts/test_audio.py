#!/usr/bin/env python3
"""
Test audio capture and playback

Simple script to test that audio I/O is working correctly
"""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from linvidia.audio import AudioCapture, AudioPlayback


def test_audio_passthrough(duration: float = 5.0):
    """
    Test audio passthrough (capture -> playback)

    Args:
        duration: Test duration in seconds
    """
    print("=== Audio Passthrough Test ===\n")

    # List devices
    print("Input devices:")
    for dev in AudioCapture.list_devices():
        print(f"  [{dev['index']}] {dev['name']}")

    print("\nOutput devices:")
    for dev in AudioPlayback.list_devices():
        print(f"  [{dev['index']}] {dev['name']}")

    # Create capture and playback
    print(f"\nRunning passthrough test for {duration} seconds...")
    print("Speak into your microphone - you should hear yourself back\n")

    capture = AudioCapture()
    playback = AudioPlayback()

    try:
        # Start
        playback.start()
        capture.start()

        # Passthrough
        start_time = time.time()
        while time.time() - start_time < duration:
            audio_frame = capture.read(timeout=1.0)
            playback.write(audio_frame)

        # Stop
        capture.stop()
        playback.stop()

        print("\n✓ Test completed successfully!")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        capture.stop()
        playback.stop()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test audio I/O')
    parser.add_argument('--duration', '-d', type=float, default=5.0, help='Test duration in seconds')
    args = parser.parse_args()

    test_audio_passthrough(args.duration)

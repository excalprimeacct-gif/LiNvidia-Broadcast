"""
Video display module for showing processed video

Supports OpenCV display and V4L2 virtual camera output
"""

import numpy as np
import cv2
from typing import Optional
import subprocess
import os


class VideoDisplay:
    """
    Display processed video frames

    Supports both GUI display and virtual camera output
    """

    def __init__(
        self,
        output_type: str = 'window',
        window_name: str = 'LiNvidia Broadcast',
        virtual_device: Optional[str] = None,
        width: int = 1280,
        height: int = 720
    ):
        """
        Initialize video display

        Args:
            output_type: 'window' for GUI, 'virtual' for v4l2loopback
            window_name: Window name for GUI display
            virtual_device: Path to virtual device (e.g., '/dev/video2')
            width: Frame width
            height: Frame height
        """
        self.output_type = output_type
        self.window_name = window_name
        self.virtual_device = virtual_device
        self.width = width
        self.height = height

        self.is_running = False
        self.virtual_writer = None
        self.frames_displayed = 0

    def start(self):
        """Start display"""
        if self.is_running:
            return

        if self.output_type == 'window':
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)
        elif self.output_type == 'virtual':
            self._start_virtual_camera()

        self.is_running = True
        print(f"Video display started: {self.output_type}")

    def _start_virtual_camera(self):
        """Start v4l2loopback virtual camera"""
        if not self.virtual_device:
            raise ValueError("Virtual device path required")

        if not os.path.exists(self.virtual_device):
            raise RuntimeError(
                f"Virtual device {self.virtual_device} not found. "
                "Please load v4l2loopback module first."
            )

        # Use ffmpeg to write to virtual camera
        # OpenCV VideoWriter with V4L2 backend
        fourcc = cv2.VideoWriter_fourcc(*'YUY2')
        self.virtual_writer = cv2.VideoWriter(
            self.virtual_device,
            fourcc,
            30,
            (self.width, self.height),
            True
        )

        if not self.virtual_writer.isOpened():
            raise RuntimeError(f"Failed to open virtual camera {self.virtual_device}")

    def show(self, frame: np.ndarray):
        """
        Display frame

        Args:
            frame: Frame as numpy array (H, W, 3) RGB
        """
        if not self.is_running:
            return

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.output_type == 'window':
            cv2.imshow(self.window_name, frame_bgr)
            cv2.waitKey(1)
        elif self.output_type == 'virtual' and self.virtual_writer:
            # Resize if needed
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

            self.virtual_writer.write(frame_bgr)

        self.frames_displayed += 1

    def stop(self):
        """Stop display"""
        if not self.is_running:
            return

        self.is_running = False

        if self.output_type == 'window':
            cv2.destroyWindow(self.window_name)
        elif self.virtual_writer:
            self.virtual_writer.release()

        print(f"Video display stopped: {self.frames_displayed} frames displayed")

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


def check_v4l2loopback() -> bool:
    """Check if v4l2loopback module is loaded"""
    try:
        result = subprocess.run(
            ['lsmod'],
            capture_output=True,
            text=True
        )
        return 'v4l2loopback' in result.stdout
    except:
        return False


def setup_virtual_camera(device_id: int = 2) -> str:
    """
    Setup v4l2loopback virtual camera

    Args:
        device_id: Device ID to create (e.g., 2 for /dev/video2)

    Returns:
        Path to virtual device
    """
    device_path = f'/dev/video{device_id}'

    # Check if already exists
    if os.path.exists(device_path):
        print(f"Virtual camera already exists: {device_path}")
        return device_path

    # Load module
    try:
        subprocess.run(
            ['sudo', 'modprobe', 'v4l2loopback',
             f'video_nr={device_id}',
             'card_label=LiNvidia_Broadcast',
             'exclusive_caps=1'],
            check=True
        )
        print(f"Virtual camera created: {device_path}")
        return device_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to create virtual camera. "
            f"Install v4l2loopback-dkms and try again."
        ) from e

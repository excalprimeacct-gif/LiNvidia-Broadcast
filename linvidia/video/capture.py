"""
Video capture module for webcam input

Supports various backends: OpenCV, V4L2
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from queue import Queue
import threading
import time


class VideoCapture:
    """
    Real-time video capture from webcam

    Captures video frames for processing with background effects
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: str = 'auto'
    ):
        """
        Initialize video capture

        Args:
            device_id: Camera device ID (0 for default webcam)
            width: Frame width
            height: Frame height
            fps: Target frames per second
            backend: Backend to use ('auto', 'v4l2', 'opencv')
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend

        self.capture = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)
        self.capture_thread = None

        # Stats
        self.frames_captured = 0
        self.dropped_frames = 0
        self.start_time = None

    def start(self):
        """Start video capture"""
        if self.is_running:
            raise RuntimeError("Video capture already running")

        # Open camera
        if self.backend == 'v4l2':
            self.capture = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        else:
            self.capture = cv2.VideoCapture(self.device_id)

        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open camera {self.device_id}")

        # Set properties
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        # Get actual properties
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))

        print(f"Video capture started: {actual_width}x{actual_height} @ {actual_fps} fps")

        # Start capture thread
        self.is_running = True
        self.start_time = time.time()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        """Capture loop running in background thread"""
        while self.is_running:
            ret, frame = self.capture.read()

            if not ret:
                print("Failed to read frame")
                continue

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.frames_captured += 1

            # Try to put frame in queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame)
            except:
                self.dropped_frames += 1

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Read next frame

        Args:
            timeout: Timeout in seconds

        Returns:
            Frame as numpy array (H, W, 3) RGB, or None if timeout
        """
        if not self.is_running:
            return None

        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None

    def stop(self):
        """Stop video capture"""
        if not self.is_running:
            return

        self.is_running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        if self.capture:
            self.capture.release()

        # Print stats
        if self.start_time:
            duration = time.time() - self.start_time
            actual_fps = self.frames_captured / duration if duration > 0 else 0
            print(f"Video capture stopped: {self.frames_captured} frames captured "
                  f"({actual_fps:.1f} fps), {self.dropped_frames} dropped")

    def get_properties(self) -> dict:
        """Get camera properties"""
        if not self.capture:
            return {}

        return {
            'width': int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.capture.get(cv2.CAP_PROP_FPS)),
            'backend': self.capture.getBackendName()
        }

    @staticmethod
    def list_devices() -> list:
        """List available video devices"""
        devices = []

        # Try first 10 device IDs
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append({
                    'id': i,
                    'name': f'Camera {i}',
                    'backend': cap.getBackendName()
                })
                cap.release()

        return devices

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

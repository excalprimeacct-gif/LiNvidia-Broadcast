"""Video processing modules for background effects"""

from .capture import VideoCapture
from .display import VideoDisplay
from .processing import VideoProcessor, BackgroundEffect

__all__ = ["VideoCapture", "VideoDisplay", "VideoProcessor", "BackgroundEffect"]

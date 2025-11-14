"""
Auto-framing logic with smart cropping and zooming

Automatically keeps subjects centered and properly framed
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

from .face_tracker import TrackedFace, FaceTracker


class FramingMode(Enum):
    """Auto-framing modes"""
    OFF = "off"
    CENTER = "center"  # Keep face centered
    HEADROOM = "headroom"  # Professional headroom (1/3 rule)
    TIGHT = "tight"  # Tight framing (close-up)
    WIDE = "wide"  # Wide framing (show more context)
    GROUP = "group"  # Frame multiple people


@dataclass
class FramingParameters:
    """Parameters for auto-framing"""
    target_face_height: float = 0.6  # Face should occupy 60% of frame height
    headroom: float = 0.15  # 15% space above head
    min_zoom: float = 1.0  # Minimum zoom level
    max_zoom: float = 3.0  # Maximum zoom level
    smoothing_factor: float = 0.1  # Smoothing for pan/zoom (lower = smoother)
    padding: float = 0.1  # Padding around face (10%)


class AutoFramer:
    """
    Automatic framing system

    Crops and zooms video to keep subjects properly framed
    """

    def __init__(
        self,
        tracker: FaceTracker,
        output_size: Tuple[int, int] = (1280, 720),
        mode: FramingMode = FramingMode.CENTER,
        params: Optional[FramingParameters] = None
    ):
        """
        Initialize auto-framer

        Args:
            tracker: Face tracker instance
            output_size: Output frame size (width, height)
            mode: Framing mode
            params: Framing parameters
        """
        self.tracker = tracker
        self.output_size = output_size
        self.mode = mode
        self.params = params or FramingParameters()

        # State for smooth transitions
        self.current_crop = None  # (x, y, width, height)
        self.current_zoom = 1.0

    def set_mode(self, mode: FramingMode):
        """Set framing mode"""
        self.mode = mode

    def _calculate_target_crop(
        self,
        image_size: Tuple[int, int],
        faces: List[TrackedFace]
    ) -> Tuple[int, int, int, int]:
        """
        Calculate target crop region

        Args:
            image_size: Input image size (width, height)
            faces: List of tracked faces

        Returns:
            Crop region (x, y, width, height)
        """
        img_width, img_height = image_size

        if not faces or self.mode == FramingMode.OFF:
            # No faces or framing disabled - return full frame
            return (0, 0, img_width, img_height)

        if self.mode == FramingMode.GROUP or len(faces) > 1:
            # Frame all faces
            return self._calculate_group_crop(image_size, faces)
        else:
            # Frame primary face
            primary_face = faces[0]
            return self._calculate_single_face_crop(image_size, primary_face)

    def _calculate_single_face_crop(
        self,
        image_size: Tuple[int, int],
        face: TrackedFace
    ) -> Tuple[int, int, int, int]:
        """Calculate crop for single face"""
        img_width, img_height = image_size
        face_bbox = face.get_smoothed_bbox()
        fx, fy, fw, fh = face_bbox

        # Calculate face center
        face_center_x = fx + fw // 2
        face_center_y = fy + fh // 2

        # Calculate desired crop size based on mode
        if self.mode == FramingMode.TIGHT:
            # Tight framing - face occupies 80% of frame
            target_height = int(fh / 0.8)
        elif self.mode == FramingMode.WIDE:
            # Wide framing - face occupies 40% of frame
            target_height = int(fh / 0.4)
        elif self.mode == FramingMode.HEADROOM:
            # Professional headroom - face at 1/3 from top
            target_height = int(fh / self.params.target_face_height)
        else:  # CENTER
            # Standard framing - face occupies 60% of frame
            target_height = int(fh / self.params.target_face_height)

        # Maintain aspect ratio
        aspect_ratio = self.output_size[0] / self.output_size[1]
        target_width = int(target_height * aspect_ratio)

        # Apply zoom limits
        zoom_factor = img_height / target_height
        zoom_factor = np.clip(zoom_factor, self.params.min_zoom, self.params.max_zoom)

        # Recalculate dimensions with zoom limit
        target_height = int(img_height / zoom_factor)
        target_width = int(target_height * aspect_ratio)

        # Calculate crop position based on mode
        if self.mode == FramingMode.HEADROOM:
            # Position face at 1/3 from top
            crop_y = max(0, fy - int(target_height * (1/3)))
            crop_x = max(0, face_center_x - target_width // 2)
        else:  # CENTER, TIGHT, WIDE
            # Center on face
            crop_x = max(0, face_center_x - target_width // 2)
            crop_y = max(0, face_center_y - target_height // 2)

        # Ensure crop stays within image bounds
        if crop_x + target_width > img_width:
            crop_x = img_width - target_width
        if crop_y + target_height > img_height:
            crop_y = img_height - target_height

        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)

        return (crop_x, crop_y, target_width, target_height)

    def _calculate_group_crop(
        self,
        image_size: Tuple[int, int],
        faces: List[TrackedFace]
    ) -> Tuple[int, int, int, int]:
        """Calculate crop to include all faces"""
        img_width, img_height = image_size

        # Find bounding box that contains all faces
        all_bboxes = [face.get_smoothed_bbox() for face in faces]

        min_x = min(bbox[0] for bbox in all_bboxes)
        min_y = min(bbox[1] for bbox in all_bboxes)
        max_x = max(bbox[0] + bbox[2] for bbox in all_bboxes)
        max_y = max(bbox[1] + bbox[3] for bbox in all_bboxes)

        # Add padding
        padding_x = int((max_x - min_x) * self.params.padding)
        padding_y = int((max_y - min_y) * self.params.padding)

        min_x = max(0, min_x - padding_x)
        min_y = max(0, min_y - padding_y)
        max_x = min(img_width, max_x + padding_x)
        max_y = min(img_height, max_y + padding_y)

        width = max_x - min_x
        height = max_y - min_y

        # Adjust to maintain output aspect ratio
        aspect_ratio = self.output_size[0] / self.output_size[1]
        current_ratio = width / height

        if current_ratio > aspect_ratio:
            # Too wide - adjust height
            new_height = int(width / aspect_ratio)
            y_adjust = (new_height - height) // 2
            min_y = max(0, min_y - y_adjust)
            height = min(new_height, img_height - min_y)
        else:
            # Too tall - adjust width
            new_width = int(height * aspect_ratio)
            x_adjust = (new_width - width) // 2
            min_x = max(0, min_x - x_adjust)
            width = min(new_width, img_width - min_x)

        return (min_x, min_y, width, height)

    def _smooth_crop(
        self,
        target_crop: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Apply temporal smoothing to crop region"""
        if self.current_crop is None:
            self.current_crop = target_crop
            return target_crop

        # Exponential moving average
        alpha = self.params.smoothing_factor

        smoothed = tuple(
            int(alpha * target + (1 - alpha) * current)
            for target, current in zip(target_crop, self.current_crop)
        )

        self.current_crop = smoothed
        return smoothed

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        """
        Process frame with auto-framing

        Args:
            image: Input image (H, W, 3) RGB

        Returns:
            Cropped and resized image
        """
        h, w = image.shape[:2]

        # Update face tracking
        tracked_faces = self.tracker.update(image)

        # Calculate target crop
        target_crop = self._calculate_target_crop((w, h), tracked_faces)

        # Apply smoothing
        smooth_crop = self._smooth_crop(target_crop)

        # Extract crop
        x, y, crop_w, crop_h = smooth_crop
        x, y = max(0, x), max(0, y)
        crop_w = min(crop_w, w - x)
        crop_h = min(crop_h, h - y)

        cropped = image[y:y+crop_h, x:x+crop_w]

        # Resize to output size
        if cropped.size > 0:
            result = cv2.resize(cropped, self.output_size, interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback if crop failed
            result = cv2.resize(image, self.output_size, interpolation=cv2.INTER_LINEAR)

        return result

    def get_framing_info(self) -> dict:
        """Get current framing information"""
        return {
            'mode': self.mode.value,
            'current_crop': self.current_crop,
            'zoom_level': self.current_zoom,
            'tracked_faces': len(self.tracker.tracks)
        }

    def reset(self):
        """Reset framing state"""
        self.current_crop = None
        self.current_zoom = 1.0
        self.tracker.reset()


def create_auto_framer(
    output_size: Tuple[int, int] = (1280, 720),
    mode: FramingMode = FramingMode.CENTER,
    detector_backend: str = 'auto'
) -> AutoFramer:
    """
    Factory function to create auto-framer

    Args:
        output_size: Output frame size
        mode: Framing mode
        detector_backend: Face detector backend

    Returns:
        AutoFramer instance
    """
    from .face_detector import create_face_detector

    # Create face detector and tracker
    detector = create_face_detector(backend=detector_backend)
    tracker = FaceTracker(detector)

    return AutoFramer(tracker, output_size, mode)

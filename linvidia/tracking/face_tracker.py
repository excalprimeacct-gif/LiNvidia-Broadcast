"""
Face tracking with temporal smoothing

Tracks faces across frames and provides smooth, stable tracking
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import time

from .face_detector import Face, FaceDetector


@dataclass
class TrackedFace:
    """Face being tracked across frames"""
    face: Face
    track_id: int
    age: int = 0  # Number of frames tracked
    last_seen: float = field(default_factory=time.time)
    history: List[Tuple[int, int, int, int]] = field(default_factory=list)  # Bbox history
    velocity: Tuple[float, float] = (0.0, 0.0)  # Movement velocity

    def update(self, face: Face):
        """Update tracked face with new detection"""
        # Update velocity
        old_center = (self.face.bbox[0] + self.face.bbox[2] // 2,
                     self.face.bbox[1] + self.face.bbox[3] // 2)
        new_center = face.center

        self.velocity = (
            new_center[0] - old_center[0],
            new_center[1] - old_center[1]
        )

        # Update face
        self.face = face
        self.age += 1
        self.last_seen = time.time()

        # Add to history
        self.history.append(face.bbox)
        if len(self.history) > 30:  # Keep last 30 frames (1 second at 30fps)
            self.history.pop(0)

    def get_smoothed_bbox(self, alpha: float = 0.3) -> Tuple[int, int, int, int]:
        """
        Get smoothed bounding box using exponential moving average

        Args:
            alpha: Smoothing factor (0=no smoothing, 1=no history)

        Returns:
            Smoothed bbox (x, y, w, h)
        """
        if not self.history:
            return self.face.bbox

        # Exponential moving average
        current = np.array(self.face.bbox, dtype=float)

        if len(self.history) > 1:
            # Average recent history
            recent = np.array(self.history[-5:], dtype=float)
            avg_history = recent.mean(axis=0)

            # Blend current with history
            smoothed = alpha * current + (1 - alpha) * avg_history
        else:
            smoothed = current

        return tuple(smoothed.astype(int))

    def predict_position(self) -> Tuple[int, int, int, int]:
        """Predict next position based on velocity"""
        x, y, w, h = self.face.bbox
        vx, vy = self.velocity

        # Simple linear prediction
        pred_x = int(x + vx)
        pred_y = int(y + vy)

        return (pred_x, pred_y, w, h)


class FaceTracker:
    """
    Multi-object face tracker

    Tracks multiple faces across frames with ID assignment
    """

    def __init__(
        self,
        detector: FaceDetector,
        max_age: int = 30,  # Max frames without detection before removing track
        iou_threshold: float = 0.3,  # IoU threshold for matching
        smoothing_alpha: float = 0.3  # Smoothing factor
    ):
        """
        Initialize face tracker

        Args:
            detector: Face detector to use
            max_age: Maximum frames to keep track without detection
            iou_threshold: IoU threshold for matching detections to tracks
            smoothing_alpha: Temporal smoothing factor
        """
        self.detector = detector
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.smoothing_alpha = smoothing_alpha

        self.tracks: List[TrackedFace] = []
        self.next_track_id = 0

    def _iou(self, bbox1: Tuple[int, int, int, int],
             bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU)

        Args:
            bbox1: First bounding box (x, y, w, h)
            bbox2: Second bounding box (x, y, w, h)

        Returns:
            IoU score [0, 1]
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, image: np.ndarray) -> List[TrackedFace]:
        """
        Update tracks with new frame

        Args:
            image: Input image (H, W, 3) RGB

        Returns:
            List of currently tracked faces
        """
        # Detect faces
        detections = self.detector.detect(image)

        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()

        # Match based on IoU
        for i, track in enumerate(self.tracks):
            best_iou = 0.0
            best_detection = None
            best_idx = -1

            # Use predicted position for better matching
            pred_bbox = track.predict_position()

            for j, detection in enumerate(detections):
                if j in matched_detections:
                    continue

                iou = self._iou(pred_bbox, detection.bbox)

                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_detection = detection
                    best_idx = j

            if best_detection is not None:
                # Update track
                track.update(best_detection)
                matched_tracks.add(i)
                matched_detections.add(best_idx)

        # Create new tracks for unmatched detections
        for j, detection in enumerate(detections):
            if j not in matched_detections:
                new_track = TrackedFace(
                    face=detection,
                    track_id=self.next_track_id
                )
                self.tracks.append(new_track)
                self.next_track_id += 1

        # Remove old tracks
        current_time = time.time()
        self.tracks = [
            track for track in self.tracks
            if (current_time - track.last_seen) < (self.max_age / 30.0)  # Assuming 30fps
        ]

        return self.tracks

    def get_primary_face(self) -> Optional[TrackedFace]:
        """
        Get primary face (largest, most stable)

        Returns:
            Primary tracked face or None
        """
        if not self.tracks:
            return None

        # Sort by age (stability) and size
        sorted_tracks = sorted(
            self.tracks,
            key=lambda t: (t.age, t.face.area),
            reverse=True
        )

        return sorted_tracks[0]

    def get_smoothed_faces(self) -> List[Tuple[TrackedFace, Tuple[int, int, int, int]]]:
        """
        Get all faces with smoothed bounding boxes

        Returns:
            List of (TrackedFace, smoothed_bbox) tuples
        """
        return [
            (track, track.get_smoothed_bbox(self.smoothing_alpha))
            for track in self.tracks
        ]

    def reset(self):
        """Reset all tracks"""
        self.tracks = []
        self.next_track_id = 0

    def draw_tracks(self, image: np.ndarray, show_ids: bool = True) -> np.ndarray:
        """
        Draw tracked faces on image

        Args:
            image: Input image
            show_ids: Show track IDs

        Returns:
            Image with drawn tracks
        """
        import cv2

        result = image.copy()

        for track, smoothed_bbox in self.get_smoothed_faces():
            x, y, w, h = smoothed_bbox

            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            # Draw landmarks if available
            if track.face.landmarks:
                for name, (lx, ly) in track.face.landmarks.items():
                    cv2.circle(result, (lx, ly), 3, (255, 0, 0), -1)

            # Draw track ID
            if show_ids:
                text = f"ID: {track.track_id} ({track.age})"
                cv2.putText(
                    result, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        return result

"""
Face detection module with multiple backends

Supports MediaPipe (fast, GPU-accelerated) and OpenCV (CPU fallback)
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Face:
    """Detected face information"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None  # e.g., {'left_eye': (x, y), ...}

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of face"""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    @property
    def area(self) -> int:
        """Get area of face bounding box"""
        _, _, w, h = self.bbox
        return w * h


class FaceDetector(ABC):
    """Abstract base class for face detectors"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Face]:
        """
        Detect faces in image

        Args:
            image: Input image (H, W, 3) RGB

        Returns:
            List of detected faces
        """
        pass


class MediaPipeFaceDetector(FaceDetector):
    """
    MediaPipe face detector

    Fast and accurate, works on CPU and GPU
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0  # 0 for short-range (2m), 1 for full-range (5m)
    ):
        """
        Initialize MediaPipe face detector

        Args:
            min_detection_confidence: Minimum confidence threshold
            model_selection: 0 for short-range, 1 for full-range detection
        """
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                min_detection_confidence=min_detection_confidence,
                model_selection=model_selection
            )
            self.available = True
        except ImportError:
            print("Warning: MediaPipe not installed. Install with: pip install mediapipe")
            self.available = False

    def detect(self, image: np.ndarray) -> List[Face]:
        """Detect faces using MediaPipe"""
        if not self.available:
            return []

        # MediaPipe expects RGB
        results = self.detector.process(image)

        faces = []
        if results.detections:
            h, w = image.shape[:2]

            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Ensure bbox is within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)

                # Get confidence
                confidence = detection.score[0] if detection.score else 0.0

                # Get landmarks
                landmarks = {}
                if detection.location_data.relative_keypoints:
                    keypoint_names = ['right_eye', 'left_eye', 'nose_tip',
                                    'mouth_center', 'right_ear', 'left_ear']
                    for i, keypoint in enumerate(detection.location_data.relative_keypoints):
                        if i < len(keypoint_names):
                            landmarks[keypoint_names[i]] = (
                                int(keypoint.x * w),
                                int(keypoint.y * h)
                            )

                faces.append(Face(
                    bbox=(x, y, width, height),
                    confidence=confidence,
                    landmarks=landmarks if landmarks else None
                ))

        return faces

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'detector'):
            self.detector.close()


class OpenCVFaceDetector(FaceDetector):
    """
    OpenCV DNN face detector (CPU/GPU)

    Fallback option when MediaPipe is not available
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        use_gpu: bool = False
    ):
        """
        Initialize OpenCV face detector

        Args:
            confidence_threshold: Minimum confidence threshold
            use_gpu: Use GPU acceleration if available
        """
        self.confidence_threshold = confidence_threshold

        # Load pre-trained model (Caffe-based)
        # Download from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
        try:
            prototxt = cv2.data.haarcascades + "deploy.prototxt"
            model = cv2.data.haarcascades + "res10_300x300_ssd_iter_140000.caffemodel"

            # Try to use DNN module
            self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

            if use_gpu:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            self.use_dnn = True
        except:
            # Fallback to Haar Cascade
            self.cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.use_dnn = False
            print("Using Haar Cascade fallback for face detection")

    def detect(self, image: np.ndarray) -> List[Face]:
        """Detect faces using OpenCV"""
        if self.use_dnn:
            return self._detect_dnn(image)
        else:
            return self._detect_haar(image)

    def _detect_dnn(self, image: np.ndarray) -> List[Face]:
        """Detect using DNN"""
        h, w = image.shape[:2]

        # Prepare blob
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), (104.0, 177.0, 123.0)
        )

        # Inference
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                # Get bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                # Convert to x, y, width, height
                x = max(0, x1)
                y = max(0, y1)
                width = min(x2 - x1, w - x)
                height = min(y2 - y1, h - y)

                faces.append(Face(
                    bbox=(x, y, width, height),
                    confidence=float(confidence),
                    landmarks=None
                ))

        return faces

    def _detect_haar(self, image: np.ndarray) -> List[Face]:
        """Detect using Haar Cascade"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect faces
        detections = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        faces = []
        for (x, y, w, h) in detections:
            faces.append(Face(
                bbox=(x, y, w, h),
                confidence=1.0,  # Haar doesn't provide confidence
                landmarks=None
            ))

        return faces


def create_face_detector(
    backend: str = 'auto',
    **kwargs
) -> FaceDetector:
    """
    Factory function to create face detector

    Args:
        backend: 'mediapipe', 'opencv', or 'auto'
        **kwargs: Additional arguments for detector

    Returns:
        FaceDetector instance
    """
    if backend == 'auto':
        # Try MediaPipe first, fallback to OpenCV
        detector = MediaPipeFaceDetector(**kwargs)
        if detector.available:
            return detector
        else:
            return OpenCVFaceDetector(**kwargs)
    elif backend == 'mediapipe':
        return MediaPipeFaceDetector(**kwargs)
    elif backend == 'opencv':
        return OpenCVFaceDetector(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

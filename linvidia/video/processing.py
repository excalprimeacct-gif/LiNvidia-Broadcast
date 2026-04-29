"""
Video processing for background effects

Handles background blur, removal, and replacement
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from enum import Enum


class BackgroundEffect(Enum):
    """Background effect types"""
    NONE = "none"
    BLUR = "blur"
    REMOVE = "remove"
    REPLACE = "replace"


class VideoProcessor:
    """
    Video processor for background effects

    Applies blur, removal, or replacement based on segmentation mask
    """

    def __init__(
        self,
        effect: BackgroundEffect = BackgroundEffect.BLUR,
        blur_strength: int = 25,
        background_image: Optional[np.ndarray] = None,
        edge_smoothing: bool = True
    ):
        """
        Initialize video processor

        Args:
            effect: Background effect to apply
            blur_strength: Blur kernel size (must be odd)
            background_image: Background replacement image (H, W, 3) RGB
            edge_smoothing: Apply edge smoothing to mask
        """
        self.effect = effect
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self.background_image = background_image
        self.edge_smoothing = edge_smoothing

        # Stats
        self.frames_processed = 0

    def set_effect(self, effect: BackgroundEffect):
        """Set background effect"""
        self.effect = effect

    def set_blur_strength(self, strength: int):
        """
        Set blur strength

        Args:
            strength: Blur strength (1-100, will be converted to kernel size)
        """
        # Map 1-100 to kernel sizes 1-51
        kernel_size = int(strength / 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.blur_strength = max(1, min(kernel_size, 51))

    def set_background_image(self, image: np.ndarray):
        """
        Set background replacement image

        Args:
            image: Background image (H, W, 3) RGB
        """
        self.background_image = image

    def apply_blur(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply background blur

        Args:
            image: Input image (H, W, 3) RGB
            mask: Person mask (H, W) in range [0, 1]

        Returns:
            Processed image with blurred background
        """
        # Blur the entire image
        blurred = cv2.GaussianBlur(image, (self.blur_strength, self.blur_strength), 0)

        # Expand mask to 3 channels (broadcast, no copy)
        mask_3ch = mask[..., np.newaxis]

        # Blend: person (mask=1) from original, background (mask=0) from blurred.
        # Clip before uint8 cast to avoid wraparound from float rounding.
        result = np.clip(image * mask_3ch + blurred * (1.0 - mask_3ch), 0, 255).astype(np.uint8)

        return result

    def apply_removal(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Remove background (replace with solid color)

        Args:
            image: Input image (H, W, 3) RGB
            mask: Person mask (H, W) in range [0, 1]
            background_color: Color for background (R, G, B)

        Returns:
            Processed image with removed background
        """
        # Create background with solid color
        background = np.full_like(image, background_color, dtype=np.uint8)

        mask_3ch = mask[..., np.newaxis]
        result = np.clip(image * mask_3ch + background * (1.0 - mask_3ch), 0, 255).astype(np.uint8)
        return result

    def apply_replacement(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Replace background with custom image

        Args:
            image: Input image (H, W, 3) RGB
            mask: Person mask (H, W) in range [0, 1]
            background: Background image (H, W, 3) RGB, or None to use self.background_image

        Returns:
            Processed image with replaced background
        """
        if background is None:
            background = self.background_image

        if background is None:
            # Fallback to green screen
            return self.apply_removal(image, mask)

        # Resize background to match image size if needed
        if background.shape[:2] != image.shape[:2]:
            background = cv2.resize(background, (image.shape[1], image.shape[0]))

        mask_3ch = mask[..., np.newaxis]
        result = np.clip(image * mask_3ch + background * (1.0 - mask_3ch), 0, 255).astype(np.uint8)
        return result

    def smooth_mask(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Smooth mask edges for better blending

        Args:
            mask: Input mask (H, W)
            kernel_size: Gaussian kernel size

        Returns:
            Smoothed mask
        """
        if kernel_size % 2 == 0:
            kernel_size += 1

        smoothed = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        return smoothed

    def process_frame(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Process frame with selected effect

        Args:
            image: Input image (H, W, 3) RGB
            mask: Person mask (H, W) in range [0, 1]

        Returns:
            Processed image
        """
        # Smooth mask edges if enabled
        if self.edge_smoothing:
            mask = self.smooth_mask(mask)

        # Apply effect
        if self.effect == BackgroundEffect.NONE:
            result = image
        elif self.effect == BackgroundEffect.BLUR:
            result = self.apply_blur(image, mask)
        elif self.effect == BackgroundEffect.REMOVE:
            result = self.apply_removal(image, mask)
        elif self.effect == BackgroundEffect.REPLACE:
            result = self.apply_replacement(image, mask)
        else:
            result = image

        self.frames_processed += 1
        return result


class AdvancedVideoProcessor(VideoProcessor):
    """
    Advanced video processor with additional features

    Includes temporal smoothing, color correction, etc.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Temporal smoothing
        self.use_temporal_smoothing = True
        self.prev_mask = None
        self.temporal_alpha = 0.7  # Weight for current frame

    def temporal_smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing to mask

        Reduces flickering by blending with previous frame

        Args:
            mask: Current frame mask

        Returns:
            Temporally smoothed mask
        """
        if not self.use_temporal_smoothing:
            return mask

        if self.prev_mask is None:
            self.prev_mask = mask
            return mask

        # Blend with previous mask
        smoothed = self.temporal_alpha * mask + (1 - self.temporal_alpha) * self.prev_mask

        self.prev_mask = smoothed
        return smoothed

    def process_frame(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Process frame with temporal smoothing"""

        # Temporal smoothing
        mask = self.temporal_smooth_mask(mask)

        # Call parent implementation
        return super().process_frame(image, mask)

    def reset(self):
        """Reset temporal state"""
        self.prev_mask = None


def apply_bokeh_blur(
    image: np.ndarray,
    mask: np.ndarray,
    blur_amount: int = 25,
    focus_point: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Apply bokeh-style background blur

    Creates depth-of-field effect with circular blur

    Args:
        image: Input image (H, W, 3) RGB
        mask: Person mask (H, W) in range [0, 1]
        blur_amount: Blur strength
        focus_point: Optional focus point (y, x)

    Returns:
        Image with bokeh blur
    """
    # Create depth map from mask
    # Person (mask=1) is in focus, background (mask=0) is out of focus
    depth_map = mask

    # Apply variable blur based on depth
    # For simplicity, use two-pass approach
    h, w = image.shape[:2]

    # Heavily blur background
    kernel_size = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Blend based on mask
    mask_3ch = mask[..., np.newaxis]
    result = np.clip(image * mask_3ch + blurred * (1.0 - mask_3ch), 0, 255).astype(np.uint8)

    return result

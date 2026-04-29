"""
Real-time background effects pipeline

Integrates video capture, segmentation, and effects processing
"""

import numpy as np
import time
from typing import Optional
from loguru import logger

from typing import Union

from .video import VideoCapture, VideoDisplay, VideoProcessor, BackgroundEffect
from .models.segmentation import create_segmentation_model, BackgroundSegmentationModel
from .tracking import create_auto_framer, FramingMode
from .utils import Config


_FRAMING_MODE_MAP = {
    'off': FramingMode.OFF,
    'center': FramingMode.CENTER,
    'headroom': FramingMode.HEADROOM,
    'tight': FramingMode.TIGHT,
    'wide': FramingMode.WIDE,
    'group': FramingMode.GROUP,
}


def _coerce_framing_mode(mode: Union[str, FramingMode]) -> FramingMode:
    if isinstance(mode, FramingMode):
        return mode
    return _FRAMING_MODE_MAP.get(str(mode).lower(), FramingMode.CENTER)


class RealtimeBackgroundEffects:
    """
    Real-time background effects system

    Captures video, segments person, and applies background effects
    with minimal latency
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        segmentation_model: Optional[BackgroundSegmentationModel] = None,
        camera_id: int = 0,
        output_type: str = 'window',
        virtual_device: Optional[str] = None,
        effect: BackgroundEffect = BackgroundEffect.BLUR,
        blur_strength: int = 25,
        enable_auto_frame: bool = False,
        framing_mode: Union[FramingMode, str] = FramingMode.CENTER,
        use_tensorrt: bool = False,
        tensorrt_engine_path: Optional[str] = None
    ):
        """
        Initialize background effects

        Args:
            config: Configuration object
            segmentation_model: Pre-loaded segmentation model
            camera_id: Camera device ID
            output_type: 'window' or 'virtual'
            virtual_device: Path to v4l2loopback device
            effect: Background effect to apply
            blur_strength: Blur strength (1-100)
            enable_auto_frame: Enable auto-framing
            framing_mode: Auto-framing mode
            use_tensorrt: Use TensorRT for 2-3x faster inference
            tensorrt_engine_path: Path to TensorRT engine (required if use_tensorrt=True)
        """
        self.config = config or Config()
        self.use_tensorrt = use_tensorrt

        # Video capture
        self.capture = VideoCapture(
            device_id=camera_id,
            width=1280,
            height=720,
            fps=30
        )

        # Video display
        self.display = VideoDisplay(
            output_type=output_type,
            virtual_device=virtual_device,
            width=1280,
            height=720
        )

        # Segmentation model
        if segmentation_model is None:
            if use_tensorrt:
                logger.info("Loading TensorRT segmentation engine...")
                if not tensorrt_engine_path:
                    logger.warning("TensorRT enabled but no engine path provided, falling back to PyTorch")
                    use_tensorrt = False
                    self.use_tensorrt = False

            if use_tensorrt:
                self.segmentation_model = create_segmentation_model(
                    use_tensorrt=True,
                    tensorrt_engine_path=tensorrt_engine_path,
                    input_size=(256, 256)
                )
                logger.info("✓ TensorRT segmentation engine loaded")
            else:
                logger.info("Loading PyTorch segmentation model...")
                self.segmentation_model = create_segmentation_model(
                    model_type='mobilenet',
                    device='cuda',
                    use_fp16=True,
                    input_size=(256, 256)
                )
                logger.info("✓ PyTorch segmentation model loaded")
        else:
            self.segmentation_model = segmentation_model

        # Video processor
        self.processor = VideoProcessor(
            effect=effect,
            blur_strength=blur_strength,
            edge_smoothing=True
        )

        # Auto-framing (optional)
        self.enable_auto_frame = enable_auto_frame
        self.auto_framer = None
        if enable_auto_frame:
            logger.info("Initializing auto-framing...")
            self.auto_framer = create_auto_framer(
                output_size=(1280, 720),
                mode=_coerce_framing_mode(framing_mode)
            )

        # State
        self.is_running = False

        # Stats
        self.frames_processed = 0
        self.total_time = 0.0
        self.segmentation_times = []
        self.processing_times = []
        self.framing_times = []

        # Latest timing values for display
        self.last_seg_time = 0.0
        self.last_proc_time = 0.0

    def set_effect(self, effect: BackgroundEffect):
        """Set background effect"""
        self.processor.set_effect(effect)
        logger.info(f"Effect changed to: {effect.value}")

    def set_blur_strength(self, strength: int):
        """Set blur strength (1-100)"""
        self.processor.set_blur_strength(strength)
        logger.info(f"Blur strength set to: {strength}")

    def set_background_image(self, image_path: str):
        """
        Set background replacement image

        Args:
            image_path: Path to background image
        """
        import cv2
        bg_image = cv2.imread(image_path)
        if bg_image is None:
            logger.error(f"Failed to load background image: {image_path}")
            return

        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
        self.processor.set_background_image(bg_image)
        logger.info(f"Background image loaded: {image_path}")

    def set_framing_mode(self, mode: Union[FramingMode, str]):
        """Set auto-framing mode"""
        if self.auto_framer:
            mode = _coerce_framing_mode(mode)
            self.auto_framer.set_mode(mode)
            logger.info(f"Framing mode changed to: {mode.value}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process single frame

        Args:
            frame: Input frame (H, W, 3) RGB

        Returns:
            Processed frame
        """
        # Auto-framing (before effects)
        if self.enable_auto_frame and self.auto_framer:
            frame_start = time.perf_counter()
            frame = self.auto_framer.process_frame(frame)
            frame_time = (time.perf_counter() - frame_start) * 1000
            self.framing_times.append(frame_time)

        # Segmentation
        seg_start = time.perf_counter()
        mask = self.segmentation_model.segment(frame, smooth=True)
        seg_time = (time.perf_counter() - seg_start) * 1000
        self.segmentation_times.append(seg_time)
        self.last_seg_time = seg_time

        # Apply effect
        proc_start = time.perf_counter()
        result = self.processor.process_frame(frame, mask)
        proc_time = (time.perf_counter() - proc_start) * 1000
        self.processing_times.append(proc_time)
        self.last_proc_time = proc_time

        # Keep only last 100 samples
        if len(self.segmentation_times) > 100:
            self.segmentation_times.pop(0)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        if len(self.framing_times) > 100:
            self.framing_times.pop(0)

        return result

    def run(self, duration: Optional[float] = None, show_fps: bool = True):
        """
        Run background effects

        Args:
            duration: Duration in seconds (None = run forever)
            show_fps: Show FPS overlay
        """
        self.start()

        try:
            start_time = time.time()
            frame_count = 0
            fps_update_interval = 1.0
            last_fps_update = start_time
            current_fps = 0.0

            while self.is_running:
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break

                # Read frame
                frame = self.capture.read(timeout=1.0)
                if frame is None:
                    continue

                # Process frame
                frame_start = time.perf_counter()
                processed_frame = self.process_frame(frame)
                frame_time = (time.perf_counter() - frame_start) * 1000

                # Add FPS overlay
                if show_fps:
                    import cv2
                    # Update FPS every second
                    frame_count += 1
                    elapsed = time.time() - last_fps_update
                    if elapsed >= fps_update_interval:
                        current_fps = frame_count / elapsed
                        frame_count = 0
                        last_fps_update = time.time()

                    # Draw FPS
                    text = f"FPS: {current_fps:.1f} | Seg: {self.last_seg_time:.1f}ms | Proc: {self.last_proc_time:.1f}ms"
                    cv2.putText(
                        processed_frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

                # Display
                self.display.show(processed_frame)

                self.frames_processed += 1
                self.total_time += frame_time

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def start(self):
        """Start background effects"""
        if self.is_running:
            return

        logger.info("Starting background effects...")

        self.capture.start()
        self.display.start()

        self.is_running = True

        logger.info("Background effects started")
        logger.info(f"Camera: {self.capture.get_properties()}")
        logger.info(f"Effect: {self.processor.effect.value}")

    def stop(self):
        """Stop background effects"""
        if not self.is_running:
            return

        logger.info("Stopping background effects...")

        self.is_running = False

        self.capture.stop()
        self.display.stop()

        # Print statistics
        self._print_statistics()

        logger.info("Background effects stopped")

    def _print_statistics(self):
        """Print performance statistics"""
        if self.frames_processed == 0:
            return

        avg_total = self.total_time / self.frames_processed
        avg_seg = np.mean(self.segmentation_times) if self.segmentation_times else 0
        avg_proc = np.mean(self.processing_times) if self.processing_times else 0
        avg_frame = np.mean(self.framing_times) if self.framing_times else 0
        fps = 1000 / avg_total if avg_total > 0 else 0

        logger.info("=== Performance Statistics ===")
        logger.info(f"Frames processed: {self.frames_processed}")
        logger.info(f"Average FPS: {fps:.1f}")
        logger.info(f"Total latency: {avg_total:.2f} ms")
        if self.enable_auto_frame:
            logger.info(f"  Auto-framing: {avg_frame:.2f} ms")
        logger.info(f"  Segmentation: {avg_seg:.2f} ms")
        logger.info(f"  Processing: {avg_proc:.2f} ms")

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


def create_background_effects(
    camera_id: int = 0,
    output_type: str = 'window',
    virtual_device: Optional[str] = None,
    effect: str = 'blur',
    blur_strength: int = 25,
    background_image: Optional[str] = None,
    enable_auto_frame: bool = False,
    framing_mode: str = 'center',
    use_tensorrt: bool = False,
    tensorrt_engine_path: Optional[str] = None
) -> RealtimeBackgroundEffects:
    """
    Factory function to create background effects system

    Args:
        camera_id: Camera device ID
        output_type: 'window' or 'virtual'
        virtual_device: Path to virtual device
        effect: 'none', 'blur', 'remove', 'replace'
        blur_strength: Blur strength (1-100)
        background_image: Path to background replacement image
        enable_auto_frame: Enable auto-framing
        framing_mode: 'off', 'center', 'headroom', 'tight', 'wide', 'group'
        use_tensorrt: Use TensorRT for 2-3x faster inference
        tensorrt_engine_path: Path to TensorRT engine

    Returns:
        RealtimeBackgroundEffects instance
    """
    # Convert effect string to enum
    effect_map = {
        'none': BackgroundEffect.NONE,
        'blur': BackgroundEffect.BLUR,
        'remove': BackgroundEffect.REMOVE,
        'replace': BackgroundEffect.REPLACE
    }
    effect_enum = effect_map.get(effect.lower(), BackgroundEffect.BLUR)

    # Convert framing mode string to enum
    framing_map = {
        'off': FramingMode.OFF,
        'center': FramingMode.CENTER,
        'headroom': FramingMode.HEADROOM,
        'tight': FramingMode.TIGHT,
        'wide': FramingMode.WIDE,
        'group': FramingMode.GROUP
    }
    framing_enum = framing_map.get(framing_mode.lower(), FramingMode.CENTER)

    # Create system
    bg_effects = RealtimeBackgroundEffects(
        camera_id=camera_id,
        output_type=output_type,
        virtual_device=virtual_device,
        effect=effect_enum,
        blur_strength=blur_strength,
        enable_auto_frame=enable_auto_frame,
        framing_mode=framing_enum,
        use_tensorrt=use_tensorrt,
        tensorrt_engine_path=tensorrt_engine_path
    )

    # Load background image if provided
    if background_image:
        bg_effects.set_background_image(background_image)

    return bg_effects

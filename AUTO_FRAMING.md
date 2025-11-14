# Auto-Framing and Face Tracking

LiNvidia Broadcast now includes intelligent auto-framing that automatically keeps you centered and properly framed!

## Features

### Face Detection & Tracking

- **MediaPipe Integration**: Fast, accurate face detection using Google's MediaPipe
- **Multi-Face Support**: Track multiple faces simultaneously
- **Temporal Smoothing**: Smooth tracking without jitter
- **Facial Landmarks**: Detect eyes, nose, mouth for precise tracking
- **Automatic Fallback**: Falls back to OpenCV if MediaPipe unavailable

### Auto-Framing Modes

1. **Center** (Default)
   - Keeps your face centered in frame
   - Face occupies ~60% of frame height
   - Best for general video calls

2. **Headroom**
   - Professional framing with 1/3 headroom rule
   - Face positioned at 1/3 from top
   - Perfect for presentations and professional meetings

3. **Tight**
   - Close-up framing
   - Face occupies ~80% of frame
   - Great for talking head content

4. **Wide**
   - More context around you
   - Face occupies ~40% of frame
   - Shows more of your environment

5. **Group**
   - Automatically frames all detected faces
   - Adjusts zoom to include everyone
   - Perfect for multi-person calls

6. **Off**
   - Disables auto-framing
   - Uses full camera view

## Quick Start

### Enable Auto-Framing

```bash
# Basic auto-framing with default (center) mode
linvidia background-effects --auto-frame

# Combine with blur and auto-framing
linvidia background-effects \
  --effect blur \
  --auto-frame \
  --framing-mode center
```

### Framing Modes

```bash
# Professional headroom framing
linvidia background-effects \
  --auto-frame \
  --framing-mode headroom

# Tight close-up
linvidia background-effects \
  --auto-frame \
  --framing-mode tight

# Wide framing with context
linvidia background-effects \
  --auto-frame \
  --framing-mode wide

# Group framing for multiple people
linvidia background-effects \
  --auto-frame \
  --framing-mode group
```

### Complete Example

```bash
# Professional setup: blur background + headroom framing + virtual camera
linvidia background-effects \
  --effect blur \
  --blur-strength 40 \
  --auto-frame \
  --framing-mode headroom \
  --output virtual \
  --virtual-device /dev/video2
```

## Technical Details

### Face Detection

**MediaPipe Face Detection** (Preferred)
- GPU-accelerated inference
- ~5-10ms latency
- Supports facial landmarks
- Confidence scores
- Two models: short-range (2m) and full-range (5m)

**OpenCV DNN** (Fallback)
- Caffe-based detector
- CPU/GPU support
- ~15ms latency
- Reliable fallback option

### Face Tracking

**Features:**
- **Track IDs**: Each face gets unique ID
- **Temporal Smoothing**: Exponential moving average for stable bounding boxes
- **Prediction**: Linear motion prediction for better tracking
- **IoU Matching**: Intersection over Union for associating detections

**Parameters:**
- `max_age`: 30 frames (1 second @ 30fps)
- `iou_threshold`: 0.3 (30% overlap required)
- `smoothing_alpha`: 0.3 (lower = smoother)

### Auto-Framing Algorithm

```
1. Detect faces in frame
2. Track faces across time
3. Calculate target crop region based on mode:
   - Single face: Center/position based on mode
   - Multiple faces: Bounding box containing all
4. Apply temporal smoothing to crop region
5. Crop and resize to output resolution
6. Apply background effects (if enabled)
```

**Zoom Limits:**
- Minimum: 1.0x (no zoom)
- Maximum: 3.0x (prevents excessive cropping)

**Smoothing:**
- Exponential moving average (α = 0.1)
- Prevents jerky movements
- Smooth transitions when you move

## Performance

| Component | Latency | Notes |
|-----------|---------|-------|
| **Face Detection** | 5-10ms | MediaPipe GPU |
| **Face Tracking** | <1ms | Lightweight |
| **Crop Calculation** | <1ms | Fast math |
| **Crop & Resize** | 3-5ms | GPU accelerated |
| **Total** | **~10-15ms** | Added to pipeline |

**Total Pipeline with Auto-Framing:**
- Auto-framing: ~15ms
- Segmentation: ~15ms
- Effects: ~5ms
- **Total: ~35ms** (28-30 FPS)

## Use Cases

### Video Conferences

```bash
# Professional meeting setup
linvidia background-effects \
  --effect blur \
  --auto-frame \
  --framing-mode headroom \
  --output virtual
```

- Automatically stays centered when you move
- Professional headroom framing
- Blurred background for privacy

### Content Creation

```bash
# Tight framing for vlogs
linvidia background-effects \
  --auto-frame \
  --framing-mode tight \
  --output window
```

- Close-up framing
- Keeps you in frame while moving
- Perfect for talking head videos

### Presentations

```bash
# Wide framing to show gestures
linvidia background-effects \
  --auto-frame \
  --framing-mode wide \
  --effect replace \
  --background ~/backgrounds/office.jpg
```

- Shows hand gestures and body language
- Professional background
- Auto-centers when you move

### Group Calls

```bash
# Frame multiple people
linvidia background-effects \
  --auto-frame \
  --framing-mode group \
  --effect blur
```

- Automatically includes everyone
- Adjusts zoom dynamically
- Great for family/team calls

## Advanced Features

### Custom Framing Parameters

Edit `linvidia/tracking/auto_frame.py` to customize:

```python
@dataclass
class FramingParameters:
    target_face_height: float = 0.6  # 60% of frame
    headroom: float = 0.15  # 15% above head
    min_zoom: float = 1.0
    max_zoom: float = 3.0
    smoothing_factor: float = 0.1  # Lower = smoother
    padding: float = 0.1  # Padding around faces
```

### Combining Features

```bash
# The ultimate setup: everything enabled
linvidia background-effects \
  --effect blur \
  --blur-strength 50 \
  --auto-frame \
  --framing-mode headroom \
  --output virtual \
  --virtual-device /dev/video2

# Then use in Zoom/Teams by selecting "LiNvidia_Broadcast" camera
```

## Troubleshooting

### MediaPipe Not Working

```bash
# Install MediaPipe
pip install mediapipe

# If still issues, check import
python -c "import mediapipe; print('OK')"
```

### Face Not Detected

1. **Lighting**: Ensure good lighting on your face
2. **Distance**: Stay within 0.5-2m from camera
3. **Angle**: Face the camera directly
4. **Model Selection**: MediaPipe has two models:
   - Model 0: Short-range (within 2m) - default
   - Model 1: Full-range (within 5m)

### Jittery Tracking

Increase smoothing by editing `FramingParameters`:
```python
smoothing_factor: float = 0.05  # Lower = smoother
```

### Face Goes Off-Screen

Increase padding:
```python
padding: float = 0.2  # More padding
```

Or disable maximum zoom:
```python
max_zoom: float = 2.0  # Less aggressive zoom
```

### Slow Performance

1. **Disable auto-framing**: Remove `--auto-frame` flag
2. **Use OpenCV**: MediaPipe might be slow on some systems
3. **Reduce resolution**: Edit VideoCapture to use 640x480

## API Usage

### Python API

```python
from linvidia.tracking import create_auto_framer, FramingMode
from linvidia.video import VideoCapture

# Create auto-framer
framer = create_auto_framer(
    output_size=(1280, 720),
    mode=FramingMode.HEADROOM
)

# Capture and process
capture = VideoCapture(device_id=0)
capture.start()

frame = capture.read()
framed = framer.process_frame(frame)

# Change mode on the fly
framer.set_mode(FramingMode.TIGHT)

# Get framing info
info = framer.get_framing_info()
print(f"Tracked faces: {info['tracked_faces']}")
```

### Face Detection Only

```python
from linvidia.tracking import create_face_detector

# Create detector
detector = create_face_detector(backend='mediapipe')

# Detect faces
faces = detector.detect(image)

for face in faces:
    x, y, w, h = face.bbox
    confidence = face.confidence
    landmarks = face.landmarks  # Dict of facial points
```

### Face Tracking Only

```python
from linvidia.tracking import FaceTracker, create_face_detector

# Create tracker
detector = create_face_detector()
tracker = FaceTracker(detector)

# Track across frames
for frame in video_frames:
    tracked_faces = tracker.update(frame)

    primary = tracker.get_primary_face()
    if primary:
        print(f"Primary face ID: {primary.track_id}")
        print(f"Age: {primary.age} frames")
        print(f"Bbox: {primary.get_smoothed_bbox()}")
```

## Integration Examples

### With Background Effects

```python
from linvidia.background_effects import create_background_effects

# Create with auto-framing enabled
bg_effects = create_background_effects(
    camera_id=0,
    effect='blur',
    blur_strength=40,
    enable_auto_frame=True,
    framing_mode='headroom',
    output_type='virtual'
)

bg_effects.run()
```

### Custom Processing

```python
from linvidia.tracking import create_auto_framer
from linvidia.video import VideoCapture, VideoDisplay

framer = create_auto_framer()
capture = VideoCapture()
display = VideoDisplay()

capture.start()
display.start()

while True:
    frame = capture.read()

    # Auto-frame
    framed = framer.process_frame(frame)

    # Your custom processing here
    processed = your_custom_function(framed)

    display.show(processed)
```

## Configuration

Add to `config/default.yaml`:

```yaml
auto_framing:
  enable: true
  mode: headroom  # off, center, headroom, tight, wide, group
  target_face_height: 0.6
  smoothing_factor: 0.1
  min_zoom: 1.0
  max_zoom: 3.0
  detector_backend: auto  # auto, mediapipe, opencv
```

## Comparison: Modes

| Mode | Face Size | Position | Use Case |
|------|-----------|----------|----------|
| **Center** | 60% | Centered | General video calls |
| **Headroom** | 60% | 1/3 from top | Professional meetings |
| **Tight** | 80% | Centered | Close-up content |
| **Wide** | 40% | Centered | Show environment |
| **Group** | Adaptive | All faces | Multi-person |
| **Off** | N/A | Full frame | Disable framing |

## Future Enhancements

Planned features:
- **Eye gaze tracking**: Look at camera even when looking at screen
- **Gesture recognition**: Control settings with hand gestures
- **Smart reframing**: Predictive reframing based on movement patterns
- **Multi-camera**: Track across multiple camera views
- **Custom zones**: Define safe zones and preferred positions

## Dependencies

```
mediapipe>=0.10.0  # Preferred
opencv-python>=4.8.0  # Fallback
numpy>=1.21.0
```

## References

- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection.html)
- [Rule of Thirds in Video Composition](https://en.wikipedia.org/wiki/Rule_of_thirds)
- [Professional Video Framing Guidelines](https://www.videomaker.com/article/c10/15408-the-basics-of-video-framing)

## Support

Having issues with auto-framing?

1. Check MediaPipe installation: `pip install mediapipe`
2. Try different framing modes
3. Adjust lighting conditions
4. Check performance with `--verbose` flag
5. Report issues on GitHub

Enjoy professional-looking video calls with automatic framing! 🎬

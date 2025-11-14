# Video Background Effects

LiNvidia Broadcast now includes real-time video background effects powered by AI!

## Features

### Background Effects

1. **Background Blur** - Blur background while keeping you in focus
   - Adjustable blur strength (1-100)
   - Bokeh-style depth-of-field effect
   - Real-time performance

2. **Background Removal** - Remove background with green screen effect
   - Perfect for streaming and video production
   - Clean chroma key output

3. **Background Replacement** - Replace background with custom image
   - Use any image as your background
   - Automatic scaling and blending
   - Great for professional meetings

## Quick Start

### 1. Window Display (Preview)

```bash
# Blur background
linvidia background-effects --effect blur --blur-strength 50

# Remove background (green screen)
linvidia background-effects --effect remove

# Replace background
linvidia background-effects --effect replace --background /path/to/image.jpg
```

### 2. Virtual Camera (For Apps)

Setup virtual camera once:

```bash
# Install v4l2loopback
sudo apt install v4l2loopback-dkms

# Create virtual camera
sudo modprobe v4l2loopback video_nr=2 card_label='LiNvidia_Broadcast' exclusive_caps=1

# Or use built-in command
linvidia setup-virtual-camera --device-id 2
```

Use virtual camera:

```bash
# Run with virtual camera output
linvidia background-effects --effect blur --output virtual --virtual-device /dev/video2
```

Then in Zoom, Teams, Discord, etc., select **"LiNvidia_Broadcast"** as your camera!

### 3. GUI Application

```bash
# Launch full-featured GUI with video controls
linvidia gui --full
```

The GUI provides:
- Camera selection
- Effect selection (None, Blur, Remove, Replace)
- Blur strength slider
- Background image selection
- Output mode (Window/Virtual Camera)
- Real-time preview

## Usage Examples

### Blur Background for Video Calls

```bash
# Setup virtual camera (one time)
linvidia setup-virtual-camera

# Run blur effect
linvidia background-effects \
  --effect blur \
  --blur-strength 30 \
  --output virtual \
  --virtual-device /dev/video2
```

Select "LiNvidia_Broadcast" in your video app!

### Professional Virtual Background

```bash
# Use your own background image
linvidia background-effects \
  --effect replace \
  --background ~/backgrounds/office.jpg \
  --output virtual
```

### Green Screen for Streaming

```bash
# Remove background completely
linvidia background-effects \
  --effect remove \
  --output virtual

# Use in OBS with chroma key filter
```

## CLI Commands

### background-effects

Run real-time background effects:

```bash
linvidia background-effects [OPTIONS]
```

Options:
- `--camera, -c`: Camera device ID (default: 0)
- `--effect, -e`: Effect type (none/blur/remove/replace)
- `--blur-strength, -b`: Blur strength 1-100 (default: 25)
- `--background`: Background image path (for replace)
- `--output, -o`: Output type (window/virtual)
- `--virtual-device`: Virtual camera path (default: /dev/video2)
- `--duration, -d`: Run duration in seconds
- `--verbose, -v`: Verbose output

### video-devices

Manage video devices:

```bash
# List available cameras
linvidia video-devices --list-cameras

# Check v4l2loopback status
linvidia video-devices --check-v4l2

# Both
linvidia video-devices
```

### setup-virtual-camera

Setup v4l2loopback virtual camera:

```bash
linvidia setup-virtual-camera --device-id 2
```

## Technical Details

### Architecture

```
Camera → Capture → Segmentation → Effect Processing → Output
                      ↓                     ↓
                 AI Model            Blur/Remove/Replace
                 (Tensor Cores)      (GPU Accelerated)
```

### Performance

| Component | Latency | Notes |
|-----------|---------|-------|
| Capture | ~10ms | 1280x720 @ 30fps |
| Segmentation | ~15ms | MobileNetV3 FP16 |
| Processing | ~5ms | GPU-accelerated |
| **Total** | **~30ms** | Acceptable for video calls |

### Segmentation Model

- **Architecture**: MobileNetV3-based U-Net
- **Input**: 256x256 RGB image
- **Output**: Person segmentation mask
- **Precision**: FP16 (tensor core optimized)
- **Pretrained**: ImageNet weights
- **Inference**: PyTorch with optional TensorRT

### Effects Processing

1. **Blur**: Gaussian blur with variable kernel size
2. **Remove**: Alpha compositing with solid color
3. **Replace**: Image blending with custom background
4. **Temporal Smoothing**: Reduces mask flickering

## Integration with Apps

### Zoom

1. Run: `linvidia background-effects --output virtual`
2. In Zoom: Settings → Video → Camera → Select "LiNvidia_Broadcast"
3. Done!

### Microsoft Teams

1. Run: `linvidia background-effects --output virtual`
2. In Teams: Settings → Devices → Camera → Select "LiNvidia_Broadcast"
3. Done!

### OBS Studio

1. Run: `linvidia background-effects --effect remove --output virtual`
2. In OBS: Add Video Capture Device → Select "LiNvidia_Broadcast"
3. Add Chroma Key filter for green screen effect
4. Perfect!

### Discord

1. Run: `linvidia background-effects --output virtual`
2. In Discord: User Settings → Voice & Video → Camera → Select "LiNvidia_Broadcast"
3. Done!

## Advanced Usage

### Custom Blur Strength During Runtime

In GUI: Use the blur strength slider while running

Via CLI: Restart with new value:
```bash
linvidia background-effects --effect blur --blur-strength 75
```

### Multiple Background Images

Create a script to cycle through backgrounds:

```bash
#!/bin/bash
BACKGROUNDS=(~/backgrounds/*.jpg)

for bg in "${BACKGROUNDS[@]}"; do
    linvidia background-effects \
        --effect replace \
        --background "$bg" \
        --duration 30 \
        --output virtual
done
```

### High Quality Mode

For better quality at cost of performance:

```python
# Use ResNet-based segmentation (slower but better)
# Edit linvidia/background_effects.py:
segmentation_model = create_segmentation_model(
    model_type='resnet',  # Instead of 'mobilenet'
    input_size=(512, 512)  # Larger input
)
```

## Troubleshooting

### Virtual Camera Not Working

```bash
# Check if module is loaded
lsmod | grep v4l2loopback

# Load module
sudo modprobe v4l2loopback video_nr=2 card_label='LiNvidia_Broadcast'

# Make persistent (add to /etc/modules)
echo "v4l2loopback" | sudo tee -a /etc/modules

# Configure module
echo "options v4l2loopback video_nr=2 card_label='LiNvidia_Broadcast' exclusive_caps=1" | \
    sudo tee /etc/modprobe.d/v4l2loopback.conf
```

### Low FPS / High Latency

1. **Reduce resolution**: Edit `VideoCapture` to use 640x480
2. **Increase blur strength mapping**: Smaller kernels = faster
3. **Disable temporal smoothing**: Edit `AdvancedVideoProcessor`
4. **Use MobileNet** instead of ResNet (default)

### Segmentation Not Accurate

1. **Better lighting**: Ensure good lighting on face
2. **Solid background**: Avoid complex backgrounds
3. **Train custom model**: Fine-tune on your specific environment
4. **Adjust smoothing**: Increase edge smoothing kernel

### App Doesn't See Virtual Camera

1. **Check permissions**: Add user to `video` group
   ```bash
   sudo usermod -a -G video $USER
   # Log out and back in
   ```

2. **Check device**: Ensure `/dev/video2` exists
   ```bash
   ls -l /dev/video*
   ```

3. **Restart app**: Close and reopen the video app

## Performance Tips

1. **Close other GPU apps**: Free up VRAM
2. **Use FP16**: Enabled by default for tensor cores
3. **Lower camera resolution**: 720p is usually sufficient
4. **Reduce blur strength**: Smaller values = faster processing
5. **Disable temporal smoothing**: If you don't mind flickering

## System Requirements

### Minimum
- NVIDIA GPU with CUDA support
- 2GB VRAM
- 4-core CPU
- 8GB RAM

### Recommended
- NVIDIA RTX GPU (tensor cores)
- 4GB+ VRAM
- 6+ core CPU
- 16GB RAM

### Software
- CUDA 11.0+
- Python 3.8+
- OpenCV 4.5+
- PyTorch 2.0+
- v4l2loopback (for virtual camera)

## Next Steps

- **Train custom model**: Fine-tune on your data for better accuracy
- **Add more effects**: Implement custom image filters
- **Optimize for your GPU**: Profile and optimize for your specific hardware
- **Combine with audio**: Use both noise suppression and background effects!

## Examples Gallery

### Before/After

- Original → Blurred background
- Original → Removed background
- Original → Custom office background

### Use Cases

- **Remote work**: Professional appearance from home
- **Streaming**: Clean background for content creation
- **Privacy**: Hide messy room or sensitive information
- **Branding**: Use company logo/background

## Contributing

Want to improve the video features? Check out:

- `linvidia/video/` - Video capture and display
- `linvidia/models/segmentation/` - Segmentation models
- `linvidia/video/processing.py` - Effect implementations
- `linvidia/background_effects.py` - Main pipeline

Pull requests welcome!

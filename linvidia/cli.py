"""
Command-line interface for LiNvidia Broadcast
"""

import click
import sys
from pathlib import Path
from typing import Optional

from .noise_suppression import create_noise_suppression
from .audio import AudioCapture, AudioPlayback
from .utils import setup_logger, Config, save_default_config
from .models import create_noise_suppression_model
from .inference.converter import pytorch_to_tensorrt


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """LiNvidia Broadcast - AI-powered audio effects for Linux"""
    pass


@cli.command()
@click.option('--input', '-i', default=None, help='Input device name')
@click.option('--output', '-o', default=None, help='Output device name')
@click.option('--strength', '-s', default=0.95, type=float, help='Suppression strength (0-1)')
@click.option('--duration', '-d', default=None, type=float, help='Duration in seconds')
@click.option('--config', '-c', default=None, help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def noise_suppression(
    input: Optional[str],
    output: Optional[str],
    strength: float,
    duration: Optional[float],
    config: Optional[str],
    verbose: bool
):
    """Run real-time noise suppression"""

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(level=log_level)

    try:
        # Create noise suppression system
        ns = create_noise_suppression(
            input_device=input,
            output_device=output,
            strength=strength,
            config_path=config
        )

        # Run
        ns.run(duration=duration)

    except KeyboardInterrupt:
        click.echo("\nStopped by user")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            raise
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', is_flag=True, help='List input devices')
@click.option('--output', '-o', is_flag=True, help='List output devices')
def list_devices(input: bool, output: bool):
    """List available audio devices"""

    if not input and not output:
        input = output = True

    if input:
        click.echo("\n=== Input Devices ===")
        devices = AudioCapture.list_devices()
        for dev in devices:
            click.echo(f"[{dev['index']}] {dev['name']}")
            click.echo(f"    Channels: {dev['channels']}, Sample Rate: {dev['sample_rate']:.0f} Hz")

    if output:
        click.echo("\n=== Output Devices ===")
        devices = AudioPlayback.list_devices()
        for dev in devices:
            click.echo(f"[{dev['index']}] {dev['name']}")
            click.echo(f"    Channels: {dev['channels']}, Sample Rate: {dev['sample_rate']:.0f} Hz")


@cli.command()
@click.option('--output', '-o', default='config/default.yaml', help='Output path')
def create_config(output: str):
    """Create default configuration file"""
    try:
        save_default_config(output)
        click.echo(f"Configuration saved to {output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output-dir', '-o', default='models', help='Output directory')
@click.option('--freq-bins', default=257, type=int, help='Number of frequency bins')
@click.option('--hidden-size', default=256, type=int, help='Hidden size')
@click.option('--num-layers', default=2, type=int, help='Number of layers')
@click.option('--tensorrt', is_flag=True, help='Convert to TensorRT')
def create_model(
    output_dir: str,
    freq_bins: int,
    hidden_size: int,
    num_layers: int,
    tensorrt: bool
):
    """Create and export noise suppression model"""

    setup_logger(level="INFO")
    click.echo("Creating noise suppression model...")

    try:
        # Create model
        model_wrapper = create_noise_suppression_model(
            freq_bins=freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            device='cuda',
            use_fp16=True
        )

        # Save PyTorch model
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_path = f"{output_dir}/noise_suppression.pth"
        model_wrapper.save(model_path)
        click.echo(f"PyTorch model saved to {model_path}")

        # Convert to TensorRT if requested
        if tensorrt:
            click.echo("Converting to TensorRT...")
            onnx_path, engine_path = pytorch_to_tensorrt(
                model_wrapper.model,
                input_shape=(freq_bins,),
                output_dir=output_dir,
                model_name='noise_suppression',
                fp16_mode=True,
                max_batch_size=1
            )
            click.echo(f"ONNX model: {onnx_path}")
            click.echo(f"TensorRT engine: {engine_path}")

        click.echo("Done!")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def show_config(config_path: str):
    """Show configuration file contents"""
    try:
        config = Config.load(config_path)
        click.echo(config.to_dict())
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--full', is_flag=True, help='Launch full-featured GUI with audio and video')
def gui(full: bool):
    """Launch GUI application"""
    try:
        if full:
            from .ui.gui_full import launch_full_gui
            sys.exit(launch_full_gui())
        else:
            from .ui.gui import launch_gui
            sys.exit(launch_gui())
    except ImportError:
        click.echo("Error: GUI dependencies not installed", err=True)
        click.echo("Install with: pip install PyQt6")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--camera', '-c', default=0, type=int, help='Camera device ID')
@click.option('--effect', '-e', default='blur', type=click.Choice(['none', 'blur', 'remove', 'replace']), help='Background effect')
@click.option('--blur-strength', '-b', default=25, type=int, help='Blur strength (1-100)')
@click.option('--background', default=None, help='Background image path (for replace effect)')
@click.option('--auto-frame', '-a', is_flag=True, help='Enable auto-framing')
@click.option('--framing-mode', '-f', default='center', type=click.Choice(['off', 'center', 'headroom', 'tight', 'wide', 'group']), help='Auto-framing mode')
@click.option('--tensorrt', is_flag=True, help='Use TensorRT for 2-3x faster segmentation')
@click.option('--tensorrt-engine', default=None, help='Path to TensorRT engine (default: models/tensorrt/segmentation.engine)')
@click.option('--output', '-o', default='window', type=click.Choice(['window', 'virtual']), help='Output type')
@click.option('--virtual-device', default='/dev/video2', help='Virtual camera device path')
@click.option('--duration', '-d', default=None, type=float, help='Duration in seconds')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def background_effects(
    camera: int,
    effect: str,
    blur_strength: int,
    background: Optional[str],
    auto_frame: bool,
    framing_mode: str,
    tensorrt: bool,
    tensorrt_engine: Optional[str],
    output: str,
    virtual_device: str,
    duration: Optional[float],
    verbose: bool
):
    """Run real-time background effects with optional auto-framing"""
    from .utils import setup_logger
    from .background_effects import create_background_effects
    from pathlib import Path

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(level=log_level)

    # Handle TensorRT engine path
    if tensorrt:
        if tensorrt_engine is None:
            tensorrt_engine = 'models/tensorrt/segmentation.engine'

        if not Path(tensorrt_engine).exists():
            click.echo(f"Error: TensorRT engine not found: {tensorrt_engine}", err=True)
            click.echo("\nExport models first with:")
            click.echo("  linvidia export-tensorrt --model segmentation")
            sys.exit(1)

        click.echo(f"Using TensorRT engine: {tensorrt_engine}")

    try:
        # Create background effects system
        bg_effects = create_background_effects(
            camera_id=camera,
            output_type=output,
            virtual_device=virtual_device if output == 'virtual' else None,
            effect=effect,
            blur_strength=blur_strength,
            background_image=background,
            enable_auto_frame=auto_frame,
            framing_mode=framing_mode,
            use_tensorrt=tensorrt,
            tensorrt_engine_path=tensorrt_engine if tensorrt else None
        )

        # Run
        bg_effects.run(duration=duration, show_fps=True)

    except KeyboardInterrupt:
        click.echo("\nStopped by user")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            raise
        sys.exit(1)


@cli.command()
@click.option('--list-cameras', is_flag=True, help='List available cameras')
@click.option('--check-v4l2', is_flag=True, help='Check v4l2loopback status')
def video_devices(list_cameras: bool, check_v4l2: bool):
    """Manage video devices"""
    from .video import VideoCapture
    from .video.display import check_v4l2loopback

    if list_cameras or (not list_cameras and not check_v4l2):
        click.echo("\n=== Available Cameras ===")
        devices = VideoCapture.list_devices()
        for dev in devices:
            click.echo(f"[{dev['id']}] {dev['name']} ({dev['backend']})")

    if check_v4l2:
        click.echo("\n=== v4l2loopback Status ===")
        if check_v4l2loopback():
            click.echo("✓ v4l2loopback module loaded")
            import os
            virtual_devices = [f"/dev/video{i}" for i in range(10) if os.path.exists(f"/dev/video{i}")]
            click.echo(f"Virtual devices: {', '.join(virtual_devices)}")
        else:
            click.echo("✗ v4l2loopback module not loaded")
            click.echo("To install:")
            click.echo("  sudo apt install v4l2loopback-dkms")
            click.echo("  sudo modprobe v4l2loopback video_nr=2 card_label='LiNvidia_Broadcast'")


@cli.command()
@click.option('--device-id', default=2, type=int, help='Virtual device ID')
def setup_virtual_camera(device_id: int):
    """Setup v4l2loopback virtual camera"""
    from .video.display import setup_virtual_camera

    try:
        device_path = setup_virtual_camera(device_id)
        click.echo(f"✓ Virtual camera created: {device_path}")
        click.echo("\nUsage:")
        click.echo(f"  linvidia background-effects --output virtual --virtual-device {device_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', '-m', type=click.Choice(['noise', 'segmentation', 'both']), required=True,
              help='Model to export')
@click.option('--output-dir', '-o', default='./models/tensorrt', help='Output directory')
@click.option('--checkpoint', '-c', default=None, help='Model checkpoint path')
@click.option('--no-fp16', is_flag=True, help='Disable FP16 precision')
def export_tensorrt(model: str, output_dir: str, checkpoint: Optional[str], no_fp16: bool):
    """Export models to TensorRT for optimized inference"""
    import torch
    from .models.noise_suppression import NoiseSuppressionRNN
    from .models.segmentation import MobileNetV3Segmentation

    if not torch.cuda.is_available():
        click.echo("Error: CUDA is not available", err=True)
        sys.exit(1)

    fp16_mode = not no_fp16
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    click.echo(f"\nGPU: {torch.cuda.get_device_name(0)}")
    click.echo(f"CUDA: {torch.version.cuda}")
    click.echo(f"FP16: {fp16_mode}\n")

    try:
        if model in ['noise', 'both']:
            click.echo("Exporting Noise Suppression model...")
            ns_model = NoiseSuppressionRNN(257, 256, 2).cuda().eval()

            if checkpoint:
                click.echo(f"Loading checkpoint: {checkpoint}")
                ckpt = torch.load(checkpoint)
                ns_model.load_state_dict(ckpt['model_state_dict'])

            onnx_path, engine_path = pytorch_to_tensorrt(
                model=ns_model,
                input_shape=(257,),
                output_dir=output_dir,
                model_name='noise_suppression',
                fp16_mode=fp16_mode
            )
            click.echo(f"✓ Noise suppression exported to {engine_path}\n")

        if model in ['segmentation', 'both']:
            click.echo("Exporting Background Segmentation model...")
            seg_model = MobileNetV3Segmentation((256, 256), pretrained=True).cuda().eval()

            if checkpoint:
                ckpt = torch.load(checkpoint)
                seg_model.load_state_dict(ckpt['model_state_dict'])

            onnx_path, engine_path = pytorch_to_tensorrt(
                model=seg_model,
                input_shape=(3, 256, 256),
                output_dir=output_dir,
                model_name='segmentation',
                fp16_mode=fp16_mode
            )
            click.echo(f"✓ Segmentation exported to {engine_path}\n")

        click.echo(f"✓ Export complete! Engines saved in: {output_dir}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--engine', '-e', required=True, help='Path to TensorRT engine')
@click.option('--model-type', '-m', type=click.Choice(['noise', 'segmentation']), required=True,
              help='Model type')
@click.option('--warmup', default=10, help='Warmup runs')
@click.option('--test-runs', default=100, help='Test runs')
@click.option('--compare-pytorch', is_flag=True, help='Compare with PyTorch')
def benchmark_tensorrt(engine: str, model_type: str, warmup: int, test_runs: int, compare_pytorch: bool):
    """Benchmark TensorRT engine performance"""
    import torch
    import numpy as np
    import time
    from .inference.engine import TensorRTEngine

    if not Path(engine).exists():
        click.echo(f"Error: Engine not found: {engine}", err=True)
        click.echo("\nExport models first with: linvidia export-tensorrt --model both")
        sys.exit(1)

    click.echo(f"\nGPU: {torch.cuda.get_device_name(0)}")
    click.echo(f"Engine: {engine}\n")

    try:
        # Determine input shape
        if model_type == 'noise':
            input_shape = (257,)
        else:
            input_shape = (3, 256, 256)

        # Load and benchmark
        trt_engine = TensorRTEngine(engine, use_cuda_stream=True)
        input_data = np.random.randn(1, *input_shape).astype(np.float32)

        # Warmup
        click.echo(f"Warming up ({warmup} runs)...")
        for _ in range(warmup):
            _ = trt_engine.infer(input_data)

        trt_engine.reset_stats()

        # Benchmark
        click.echo(f"Benchmarking ({test_runs} runs)...")
        latencies = []
        for _ in range(test_runs):
            start = time.perf_counter()
            _ = trt_engine.infer(input_data)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        # Results
        latencies = np.array(latencies)
        click.echo("\n=== TensorRT Results ===")
        click.echo(f"Average:  {np.mean(latencies):.3f} ms")
        click.echo(f"Min:      {np.min(latencies):.3f} ms")
        click.echo(f"Max:      {np.max(latencies):.3f} ms")
        click.echo(f"P95:      {np.percentile(latencies, 95):.3f} ms")
        click.echo(f"FPS:      {1000/np.mean(latencies):.1f}")

        if compare_pytorch:
            click.echo("\n=== PyTorch Comparison ===")
            if model_type == 'noise':
                from .models.noise_suppression import NoiseSuppressionRNN
                model = NoiseSuppressionRNN(257, 256, 2).cuda().eval()
                pt_input = torch.randn(1, 257, device='cuda')

                with torch.no_grad():
                    for _ in range(warmup):
                        _ = model(pt_input)

                    pt_latencies = []
                    for _ in range(test_runs):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        _ = model(pt_input)
                        torch.cuda.synchronize()
                        end = time.perf_counter()
                        pt_latencies.append((end - start) * 1000)

                pt_avg = np.mean(pt_latencies)
                click.echo(f"PyTorch:  {pt_avg:.3f} ms")
                click.echo(f"Speedup:  {pt_avg/np.mean(latencies):.2f}x")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def check_system():
    """Check system requirements"""
    import torch

    click.echo("=== System Check ===\n")

    # Check CUDA
    if torch.cuda.is_available():
        click.echo("✓ CUDA available")
        click.echo(f"  Device: {torch.cuda.get_device_name(0)}")
        click.echo(f"  CUDA version: {torch.version.cuda}")
        click.echo(f"  Compute capability: {torch.cuda.get_device_capability(0)}")

        # Check tensor cores
        major, minor = torch.cuda.get_device_capability(0)
        has_tensor_cores = major >= 7  # Volta (7.0) and newer
        if has_tensor_cores:
            click.echo("✓ Tensor cores available")
        else:
            click.echo("✗ Tensor cores not available (need compute capability >= 7.0)")
    else:
        click.echo("✗ CUDA not available")

    # Check TensorRT
    try:
        import tensorrt as trt
        click.echo(f"✓ TensorRT available (version {trt.__version__})")
    except ImportError:
        click.echo("✗ TensorRT not available")

    # Check audio
    try:
        import sounddevice
        # Verify sounddevice is importable
        _ = sounddevice
        click.echo("✓ Audio backend available")
        click.echo(f"  {len(AudioCapture.list_devices())} input devices")
        click.echo(f"  {len(AudioPlayback.list_devices())} output devices")
    except Exception as e:
        click.echo(f"✗ Audio backend error: {e}")


if __name__ == '__main__':
    cli()

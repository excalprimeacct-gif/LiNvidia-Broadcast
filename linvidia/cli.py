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
def check_system():
    """Check system requirements"""
    import torch

    click.echo("=== System Check ===\n")

    # Check CUDA
    if torch.cuda.is_available():
        click.echo(f"✓ CUDA available")
        click.echo(f"  Device: {torch.cuda.get_device_name(0)}")
        click.echo(f"  CUDA version: {torch.version.cuda}")
        click.echo(f"  Compute capability: {torch.cuda.get_device_capability(0)}")

        # Check tensor cores
        major, minor = torch.cuda.get_device_capability(0)
        has_tensor_cores = major >= 7  # Volta (7.0) and newer
        if has_tensor_cores:
            click.echo(f"✓ Tensor cores available")
        else:
            click.echo(f"✗ Tensor cores not available (need compute capability >= 7.0)")
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
        import sounddevice as sd
        click.echo(f"✓ Audio backend available")
        click.echo(f"  {len(AudioCapture.list_devices())} input devices")
        click.echo(f"  {len(AudioPlayback.list_devices())} output devices")
    except Exception as e:
        click.echo(f"✗ Audio backend error: {e}")


if __name__ == '__main__':
    cli()

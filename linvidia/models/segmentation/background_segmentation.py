"""
Background segmentation models for person detection

Optimized for real-time inference on NVIDIA GPUs with tensor cores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import torchvision.models as models


class MobileNetV3Segmentation(nn.Module):
    """
    Lightweight segmentation model based on MobileNetV3

    Fast and efficient for real-time background segmentation
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        pretrained: bool = True
    ):
        """
        Initialize segmentation model

        Args:
            input_size: Input image size (height, width)
            pretrained: Use pretrained ImageNet weights
        """
        super().__init__()

        self.input_size = input_size

        # Backbone: MobileNetV3-Small
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        self.features = mobilenet.features

        # Decoder
        self.decoder = nn.Sequential(
            # Upsample from 8x8 to 16x16
            nn.ConvTranspose2d(576, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 16x16 to 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 32x32 to 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64x64 to 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 128x128 to 256x256
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Final conv
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input image (batch, 3, H, W)

        Returns:
            Segmentation mask (batch, 1, H, W)
        """
        # Encode
        features = self.features(x)

        # Decode
        mask = self.decoder(features)

        return mask


class ResNetSegmentation(nn.Module):
    """
    ResNet-based segmentation model

    Higher quality but slower than MobileNet
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        pretrained: bool = True
    ):
        super().__init__()

        self.input_size = input_size

        # Backbone: ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # Decoder with skip connections
        self.up4 = self._make_upconv(512, 256)
        self.up3 = self._make_upconv(256 + 256, 128)
        self.up2 = self._make_upconv(128 + 128, 64)
        self.up1 = self._make_upconv(64 + 64, 32)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _make_upconv(self, in_channels: int, out_channels: int):
        """Create upsampling block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x1)

        x2 = self.layer1(x1)  # 64
        x3 = self.layer2(x2)  # 128
        x4 = self.layer3(x3)  # 256
        x5 = self.layer4(x4)  # 512

        # Decoder with skip connections
        d4 = self.up4(x5)
        d4 = torch.cat([d4, x4], dim=1)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3], dim=1)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)

        d1 = self.up1(d2)

        # Final
        mask = self.final(d1)

        return mask


class BackgroundSegmentationModel:
    """
    High-level interface for background segmentation

    Handles preprocessing, inference, and postprocessing
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        use_fp16: bool = True,
        input_size: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize segmentation model

        Args:
            model: PyTorch segmentation model
            device: Device to run on
            use_fp16: Use FP16 precision
            input_size: Model input size
        """
        self.device = device
        self.use_fp16 = use_fp16
        self.input_size = input_size

        self.model = model.to(device)
        self.model.eval()

        if use_fp16:
            self.model = self.model.half()

        # Mean and std for normalization (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        if use_fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model

        Args:
            image: Input image (H, W, 3) RGB in range [0, 255]

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.float() / 255.0
        image_tensor = image_tensor.to(self.device)

        if self.use_fp16:
            image_tensor = image_tensor.half()

        # Resize
        if image.shape[:2] != self.input_size:
            image_tensor = F.interpolate(
                image_tensor,
                size=self.input_size,
                mode='bilinear',
                align_corners=False
            )

        # Normalize
        image_tensor = (image_tensor - self.mean) / self.std

        return image_tensor

    def postprocess(
        self,
        mask: torch.Tensor,
        original_size: Tuple[int, int],
        smooth: bool = True,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Postprocess mask

        Args:
            mask: Model output (1, 1, H, W)
            original_size: Original image size (height, width)
            smooth: Apply smoothing
            threshold: Threshold for binary mask

        Returns:
            Mask as numpy array (H, W) in range [0, 1]
        """
        # Resize to original size
        if mask.shape[2:] != original_size:
            mask = F.interpolate(
                mask,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )

        # Convert to numpy
        mask = mask.squeeze().cpu().float().numpy()

        # Smooth edges
        if smooth:
            import cv2
            kernel_size = max(3, int(original_size[0] * 0.01))
            if kernel_size % 2 == 0:
                kernel_size += 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

        return mask

    @torch.no_grad()
    def segment(
        self,
        image: np.ndarray,
        smooth: bool = True,
        return_binary: bool = False,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Segment person from background

        Args:
            image: Input image (H, W, 3) RGB
            smooth: Apply edge smoothing
            return_binary: Return binary mask instead of soft mask
            threshold: Threshold for binary mask

        Returns:
            Segmentation mask (H, W) in range [0, 1]
        """
        original_size = image.shape[:2]

        # Preprocess
        input_tensor = self.preprocess(image)

        # Inference
        mask = self.model(input_tensor)

        # Postprocess
        mask = self.postprocess(mask, original_size, smooth)

        # Binarize if requested
        if return_binary:
            mask = (mask > threshold).astype(np.float32)

        return mask

    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'device': self.device,
            'use_fp16': self.use_fp16,
            'input_size': self.input_size
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'BackgroundSegmentationModel':
        """Load model"""
        checkpoint = torch.load(path, map_location=device)

        # Create model (need to know architecture)
        # For now, assume MobileNetV3
        model = MobileNetV3Segmentation(
            input_size=checkpoint.get('input_size', (256, 256))
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        return cls(
            model,
            device=device,
            use_fp16=checkpoint.get('use_fp16', True),
            input_size=checkpoint.get('input_size', (256, 256))
        )


def create_segmentation_model(
    model_type: str = 'mobilenet',
    device: str = 'cuda',
    use_fp16: bool = True,
    input_size: Tuple[int, int] = (256, 256),
    pretrained: bool = True,
    use_tensorrt: bool = False,
    tensorrt_engine_path: Optional[str] = None
) -> BackgroundSegmentationModel:
    """
    Factory function to create segmentation model

    Args:
        model_type: 'mobilenet' or 'resnet'
        device: Device to run on
        use_fp16: Use FP16 precision
        input_size: Input size
        pretrained: Use pretrained weights
        use_tensorrt: Use TensorRT for inference (2-3x faster)
        tensorrt_engine_path: Path to TensorRT engine (required if use_tensorrt=True)

    Returns:
        BackgroundSegmentationModel or TensorRTSegmentationEngine instance
    """
    if use_tensorrt:
        if not tensorrt_engine_path:
            raise ValueError("tensorrt_engine_path is required when use_tensorrt=True")

        from ...inference import TensorRTSegmentationEngine
        return TensorRTSegmentationEngine(
            engine_path=tensorrt_engine_path,
            input_size=input_size,
            use_cuda_stream=True
        )

    # PyTorch model
    if model_type == 'mobilenet':
        model = MobileNetV3Segmentation(input_size, pretrained)
    elif model_type == 'resnet':
        model = ResNetSegmentation(input_size, pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return BackgroundSegmentationModel(model, device, use_fp16, input_size)

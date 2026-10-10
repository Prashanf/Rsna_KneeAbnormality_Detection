"""
Registry and factory for 2D MRI slice feature encoders.
"""

from typing import Dict, Optional, Type
import torch
import torch.nn as nn

from src.encoders.base import BaseSliceEncoder, get_device
from src.encoders.resnet import ResNet50Encoder
from src.encoders.dino import DINOv2Encoder


class DummyMockEncoder(BaseSliceEncoder):
    """Fast linear projection encoder used for smoke testing without network downloads."""

    def __init__(self, feature_dim: int = 64, img_size: int = 224, device: str = "auto"):
        super().__init__(name="dummy_mock", feature_dim=feature_dim, img_size=img_size, device=device)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(3, feature_dim)
        self.to(self.target_device)
        self.eval()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, 3, H, W] -> pool to [B, 3, 1, 1] -> flatten [B, 3] -> project to [B, feature_dim]
        pooled = self.pool(x).flatten(1)
        return self.proj(pooled)


ENCODER_REGISTRY = {
    "resnet50": ResNet50Encoder,
    "dinov2": lambda **kwargs: DINOv2Encoder(variant="dinov2_small", **kwargs),
    "dinov2_small": lambda **kwargs: DINOv2Encoder(variant="dinov2_small", **kwargs),
    "dinov2_base": lambda **kwargs: DINOv2Encoder(variant="dinov2_base", **kwargs),
    "dinov3": lambda **kwargs: DINOv2Encoder(variant="dinov3_small", **kwargs),
    "dino_v3": lambda **kwargs: DINOv2Encoder(variant="dinov3_small", **kwargs),
    "dinov3_small": lambda **kwargs: DINOv2Encoder(variant="dinov3_small", **kwargs),
    "dinov3_base": lambda **kwargs: DINOv2Encoder(variant="dinov3_base", **kwargs),
    "dummy_mock": DummyMockEncoder,
}


def get_encoder(
    name: str = "resnet50",
    pretrained: bool = True,
    img_size: int = 224,
    device: str = "auto",
    checkpoint_path: Optional[str] = None,
) -> BaseSliceEncoder:
    """
    Factory function to instantiate registered slice encoders.

    Supported names:
    - 'resnet50' (2048-dim)
    - 'dinov2' / 'dinov2_small' (384-dim)
    - 'dinov2_base' (768-dim)
    - 'dinov3' / 'dino_v3' / 'dinov3_small' (384-dim)
    - 'dinov3_base' (768-dim)
    - 'dummy_mock' (64-dim, lightweight offline testing)
    """
    name_clean = name.lower()
    if name_clean not in ENCODER_REGISTRY:
        raise ValueError(
            f"Encoder '{name}' not found in registry. Available options: {list(ENCODER_REGISTRY.keys())}"
        )

    constructor = ENCODER_REGISTRY[name_clean]
    if name_clean == "dummy_mock":
        return constructor(img_size=img_size, device=device)

    return constructor(
        pretrained=pretrained,
        img_size=img_size,
        device=device,
        checkpoint_path=checkpoint_path,
    )

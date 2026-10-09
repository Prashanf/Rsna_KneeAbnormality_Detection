"""
ResNet50 backbone implementation for 2D MRI slice feature extraction.
"""

from typing import Optional
import timm
import torch
import torch.nn as nn

from src.encoders.base import BaseSliceEncoder


class ResNet50Encoder(BaseSliceEncoder):
    """
    ResNet50 encoder (ImageNet pretrained via timm).
    Outputs 2048-dimensional global-average-pooled feature representations.
    """

    def __init__(
        self,
        pretrained: bool = True,
        img_size: int = 224,
        device: str = "auto",
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(name="resnet50", feature_dim=2048, img_size=img_size, device=device)

        # num_classes=0 removes final FC classification layer and applies global pooling
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=pretrained and (checkpoint_path is None),
            num_classes=0,
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=False)

        self.to(self.target_device)
        self.eval()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

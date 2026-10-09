"""
DINOv2 vision transformer encoder for 2D MRI slice feature extraction.
"""

from typing import Optional
import timm
import torch
import torch.nn as nn

from src.encoders.base import BaseSliceEncoder


class DINOv2Encoder(BaseSliceEncoder):
    """
    DINOv2 Vision Transformer encoder (pretrained via timm).
    Supports small (384-dim) and base (768-dim) architectures.
    Outputs the global [CLS] token representation.
    """

    MODEL_CONFIGS = {
        "dinov2_small": ("vit_small_patch14_dinov2.lvd142m", 384),
        "dinov2_base": ("vit_base_patch14_dinov2.lvd142m", 768),
    }

    def __init__(
        self,
        variant: str = "dinov2_small",
        pretrained: bool = True,
        img_size: int = 224,
        device: str = "auto",
        checkpoint_path: Optional[str] = None,
    ):
        if variant not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown DINOv2 variant '{variant}'. Options: {list(self.MODEL_CONFIGS.keys())}")

        timm_name, feat_dim = self.MODEL_CONFIGS[variant]
        super().__init__(name=variant, feature_dim=feat_dim, img_size=img_size, device=device)

        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained and (checkpoint_path is None),
            num_classes=0,
            img_size=img_size,
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=False)

        self.to(self.target_device)
        self.eval()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

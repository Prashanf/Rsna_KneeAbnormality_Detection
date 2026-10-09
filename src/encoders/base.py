"""
Base class and interface for 2D slice visual encoders.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T


def get_device(requested_device: str = "auto") -> torch.device:
    """Resolves target compute device (cuda, mps, or cpu)."""
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(requested_device)


class BaseSliceEncoder(nn.Module, ABC):
    """
    Abstract base class for slice-level visual feature extractors.
    All encoders take batched images [B, 3, H, W] and output [B, feature_dim].
    """

    def __init__(self, name: str, feature_dim: int, img_size: int = 224, device: str = "auto"):
        super().__init__()
        self.name = name
        self.feature_dim = feature_dim
        self.img_size = img_size
        self.target_device = get_device(device)

        # Standard ImageNet normalization: (x - mean) / std
        self.normalize_transform = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts pooled [B, feature_dim] features from normalized tensor."""
        pass

    @torch.no_grad()
    def encode(self, batch: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Takes input images with shape [B, 3, H, W] in range [0, 1].
        Applies ImageNet normalization and extracts [B, feature_dim] embeddings.

        Returns:
            np.ndarray: shape [B, feature_dim], dtype float32.
        """
        self.eval()

        if isinstance(batch, np.ndarray):
            tensor_batch = torch.from_numpy(batch).float()
        else:
            tensor_batch = batch.float()

        # Ensure on target device
        tensor_batch = tensor_batch.to(self.target_device)

        # Apply ImageNet normalization
        normalized = self.normalize_transform(tensor_batch)

        # Extract features
        features = self.forward_features(normalized)

        # Check for NaNs
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError(f"Encoder '{self.name}' produced NaN or Inf feature values.")

        return features.detach().cpu().numpy().astype(np.float32)

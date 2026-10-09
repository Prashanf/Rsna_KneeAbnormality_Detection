"""
Cross-series fusion transformer for Knee MRI studies.
Attends across variable series representations in a study to output a unified study-level vector.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.series_encoder import GatedAttentionPooling


class FusionEncoder(nn.Module):
    """
    Cross-series fusion transformer.
    Takes a variable number of series vectors per study [B, max_series, d_model]
    with a boolean mask [B, max_series], applies multi-head self-attention,
    and pools into a single study-level representation [B, d_model].
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # Attention pooling across series
        self.pooler = GatedAttentionPooling(in_features=d_model)

    def forward(
        self,
        series_vectors: torch.Tensor,
        series_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            series_vectors: [B, num_series, d_model]
            series_mask:    [B, num_series] boolean mask (True for valid, False for padded/missing)
        Returns:
            study_vector:   [B, d_model]
            series_weights: [B, num_series, 1]
        """
        # Self-attention across series (PyTorch src_key_padding_mask: True = ignore)
        key_padding_mask = ~series_mask if series_mask is not None else None
        x = self.transformer(series_vectors, src_key_padding_mask=key_padding_mask)
        x = self.layer_norm(x)

        # Gated attention pooling across series
        study_vector, series_weights = self.pooler(x, mask=series_mask)
        return study_vector, series_weights

"""
Classification heads, end-to-end model assembly, and multilabel loss computation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_audit import TARGET_LABELS
from src.model.fusion import FusionEncoder
from src.model.series_encoder import SeriesEncoder, SeriesMetadataEmbedding


class FindingHeads(nn.Module):
    """
    12 independent binary classification heads mapping study representation [B, d_model]
    to independent logits [B, 12].
    """

    def __init__(self, in_features: int = 128, num_classes: int = 12, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.shared_trunk = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 12 independent linear projection heads (one logit per finding)
        self.heads = nn.ModuleList([nn.Linear(in_features, 1) for _ in range(num_classes)])

    def forward(self, study_vector: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            study_vector: [B, in_features]
        Returns:
            logits: [B, 12]
        """
        trunk = self.shared_trunk(study_vector)
        logits = [head(trunk) for head in self.heads]  # list of [B, 1]
        return torch.cat(logits, dim=-1)  # [B, 12]


class KneeMRIClassifier(nn.Module):
    """
    End-to-end hierarchical model for Knee MRI multilabel classification:
    1. SeriesMetadataEmbedding: embeds plane, fluid sensitivity, and fat saturation.
    2. SeriesEncoder: shared slice-level Transformer + Attention-MIL pooling.
    3. FusionEncoder: cross-series Transformer + study-level Attention-MIL pooling.
    4. FindingHeads: shared trunk + 12 independent binary heads.
    """

    def __init__(
        self,
        slice_dim: int = 2048,
        d_model: int = 128,
        n_heads: int = 4,
        series_layers: int = 2,
        fusion_layers: int = 2,
        dropout: float = 0.1,
        meta_dim: int = 32,
        num_classes: int = 12,
        max_slices: int = 512,
    ):
        super().__init__()
        self.slice_dim = slice_dim
        self.d_model = d_model
        self.num_classes = num_classes

        # 1. Metadata embedding
        self.meta_embedder = SeriesMetadataEmbedding(embed_dim=meta_dim)

        # 2. Shared per-series encoder
        self.series_encoder = SeriesEncoder(
            in_dim=slice_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=series_layers,
            dropout=dropout,
            meta_dim=meta_dim,
            max_slices=max_slices,
        )

        # 3. Cross-series fusion
        self.fusion_encoder = FusionEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=fusion_layers,
            dropout=dropout,
        )

        # 4. Independent classification heads
        self.heads = FindingHeads(in_features=d_model, num_classes=num_classes, dropout=dropout)

    def forward(
        self,
        slice_embeddings: torch.Tensor,
        slice_masks: Optional[torch.Tensor] = None,
        series_masks: Optional[torch.Tensor] = None,
        series_meta: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inputs:
            slice_embeddings: [B, max_series, max_slices, slice_dim]
            slice_masks:      [B, max_series, max_slices] (bool, True for valid slice)
            series_masks:     [B, max_series] (bool, True for valid series)
            series_meta:      [B, max_series, 3] (plane_idx, fluid_idx, fat_idx)
        Returns:
            dict containing:
                'logits': [B, 12]
                'probabilities': [B, 12]
                'study_vector': [B, d_model]
                'series_vectors': [B, max_series, d_model]
                'series_attention': [B, max_series, 1]
        """
        B, S, L, D = slice_embeddings.shape

        # Default masks if not provided
        if slice_masks is None:
            slice_masks = torch.ones((B, S, L), dtype=torch.bool, device=slice_embeddings.device)
        if series_masks is None:
            series_masks = torch.ones((B, S), dtype=torch.bool, device=slice_embeddings.device)

        # 1. Compute metadata vectors for each series: [B * S, meta_dim]
        meta_vecs = None
        if series_meta is not None:
            plane_idx = series_meta[..., 0].view(-1)
            fluid_idx = series_meta[..., 1].view(-1)
            fat_idx = series_meta[..., 2].view(-1)
            meta_vecs = self.meta_embedder(plane_idx, fluid_idx, fat_idx)

        # 2. Parallel per-series encoding: reshape [B, S, L, D] -> [B * S, L, D]
        flat_slices = slice_embeddings.view(B * S, L, D)
        flat_slice_masks = slice_masks.view(B * S, L)

        series_vecs_flat, slice_attns_flat = self.series_encoder(
            slice_embeddings=flat_slices,
            slice_mask=flat_slice_masks,
            series_meta_vec=meta_vecs,
        )

        # Reshape back to study dimension: [B, S, d_model]
        series_vectors = series_vecs_flat.view(B, S, self.d_model)

        # Zero out invalid series vectors before fusion
        series_vectors = series_vectors.masked_fill(~series_masks.unsqueeze(-1), 0.0)

        # 3. Cross-series fusion
        study_vector, series_attns = self.fusion_encoder(
            series_vectors=series_vectors,
            series_mask=series_masks,
        )

        # 4. 12 Classification Heads
        logits = self.heads(study_vector)
        probabilities = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "study_vector": study_vector,
            "series_vectors": series_vectors,
            "series_attention": series_attns,
        }


def compute_multilabel_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes BCEWithLogitsLoss across the 12 binary labels,
    optionally weighted by per-class pos_weight to handle imbalance.
    """
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(logits, targets.float())

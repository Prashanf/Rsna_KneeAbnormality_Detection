"""
Per-series transformer encoder with metadata conditioning and Gated Attention-MIL pooling.
Transforms variable-length sequence of slice embeddings [num_slices, D] into
a single series representation vector [d_model].
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesMetadataEmbedding(nn.Module):
    """
    Embeds categorical series attributes:
    - Anatomical Plane: Sagittal (0), Coronal (1), Axial (2), Unknown (3)
    - Fluid Sensitive: 0 or 1
    - Fat Suppression: 0 or 1
    Projects and outputs a unified metadata vector of dimension embed_dim.
    """

    PLANE_MAP = {"sagittal": 0, "coronal": 1, "axial": 2, "unknown": 3}

    def __init__(self, embed_dim: int = 32, num_planes: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.plane_emb = nn.Embedding(num_planes, embed_dim // 2)
        # Fluid sensitive and fat suppression: 2 categories each (0 or 1)
        self.fluid_emb = nn.Embedding(2, embed_dim // 4)
        self.fat_emb = nn.Embedding(2, embed_dim // 4)
        self.proj = nn.Linear(embed_dim // 2 + (embed_dim // 4) * 2, embed_dim)

    def forward(self, plane_idx: torch.Tensor, fluid_idx: torch.Tensor, fat_idx: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            plane_idx: [B, S] or [B] (int)
            fluid_idx: [B, S] or [B] (int)
            fat_idx:   [B, S] or [B] (int)
        Returns:
            torch.Tensor: [B, S, embed_dim] or [B, embed_dim]
        """
        p = self.plane_emb(plane_idx.clamp(0, 3))
        fl = self.fluid_emb(fluid_idx.clamp(0, 1))
        fa = self.fat_emb(fat_idx.clamp(0, 1))
        concat = torch.cat([p, fl, fa], dim=-1)
        return self.proj(concat)


class GatedAttentionPooling(nn.Module):
    """
    Gated Attention Multiple Instance Learning (Attention-MIL) pooling
    as proposed by Ilse et al. (2018).
    Computes learned attention weights a_i over slices:
        a_i = softmax(w^T * (tanh(V * h_i) * sigmoid(U * h_i)))
    """

    def __init__(self, in_features: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or max(in_features // 2, 32)
        self.V = nn.Linear(in_features, hidden_dim)
        self.U = nn.Linear(in_features, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(
        self, h: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            h: [N, num_slices, in_features]
            mask: [N, num_slices] boolean mask (True = valid slice, False = padded)
        Returns:
            pooled: [N, in_features]
            weights: [N, num_slices, 1]
        """
        # Gated representation: tanh(V h) * sigmoid(U h)
        gated = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        # Unnormalized attention logits: [N, num_slices, 1]
        scores = self.w(gated)

        if mask is not None:
            # Mask out padding tokens with large negative value
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)

        # Softmax over slice dimension
        weights = F.softmax(scores, dim=1)

        # Handle degenerate all-padding case safely
        if mask is not None:
            weights = weights.masked_fill(~mask.unsqueeze(-1), 0.0)
            norm = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            weights = weights / norm

        # Weighted sum: [N, in_features]
        pooled = (weights * h).sum(dim=1)
        return pooled, weights


class SeriesEncoder(nn.Module):
    """
    Per-series Transformer Encoder with slice positional encodings,
    metadata conditioning, and Attention-MIL pooling.
    Shared across all series within a study.
    """

    def __init__(
        self,
        in_dim: int = 2048,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        meta_dim: int = 32,
        max_slices: int = 512,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model

        # Project input slice embeddings (e.g. 2048 from ResNet) to d_model
        self.input_proj = nn.Linear(in_dim, d_model)

        # Metadata projection to d_model
        self.meta_proj = nn.Linear(meta_dim, d_model)

        # Learnable slice positional embeddings
        self.pos_emb = nn.Embedding(max_slices, d_model)

        # Transformer encoder layers
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

        # Gated attention pooling to collapse slice dimension
        self.pooler = GatedAttentionPooling(in_features=d_model)

    def forward(
        self,
        slice_embeddings: torch.Tensor,
        slice_mask: Optional[torch.Tensor] = None,
        series_meta_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            slice_embeddings: [N, num_slices, in_dim]
            slice_mask: [N, num_slices] (bool, True for valid, False for padding)
            series_meta_vec: [N, meta_dim] (optional metadata representation)
        Returns:
            series_vec: [N, d_model]
            attn_weights: [N, num_slices, 1]
        """
        N, num_slices, _ = slice_embeddings.shape

        # 1. Project slice features to d_model
        x = self.input_proj(slice_embeddings)  # [N, num_slices, d_model]

        # 2. Add slice positional embeddings (anatomical sequence index)
        positions = torch.arange(num_slices, device=slice_embeddings.device).clamp(0, self.pos_emb.num_embeddings - 1)
        x = x + self.pos_emb(positions).unsqueeze(0)

        # 3. Add broadcast series metadata vector if provided
        if series_meta_vec is not None:
            meta_proj = self.meta_proj(series_meta_vec).unsqueeze(1)  # [N, 1, d_model]
            x = x + meta_proj

        # 4. Self-attention across slices (PyTorch src_key_padding_mask: True = padded/ignore)
        key_padding_mask = ~slice_mask if slice_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.layer_norm(x)

        # 5. Gated attention-MIL pooling -> [N, d_model]
        series_vec, attn_weights = self.pooler(x, mask=slice_mask)
        return series_vec, attn_weights

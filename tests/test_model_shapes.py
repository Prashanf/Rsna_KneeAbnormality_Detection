"""
Unit and shape tests for Phase 4 model architecture.
Covers variable series counts, variable slice counts, 1-slice series,
120+ slice long-tail series, missing series masks, and loss computation.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import StudyDataset, collate_study_batch
from src.model.heads import KneeMRIClassifier, compute_multilabel_bce_loss
from src.model.series_encoder import SeriesEncoder, SeriesMetadataEmbedding
from src.model.fusion import FusionEncoder


def test_series_metadata_embedding():
    """Verify metadata embedding shapes and values."""
    embedder = SeriesMetadataEmbedding(embed_dim=32)
    plane = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    fluid = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.long)
    fat = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.long)

    out = embedder(plane, fluid, fat)
    assert out.shape == (2, 3, 32)
    assert not torch.isnan(out).any()


def test_series_encoder_attention_pooling():
    """Verify SeriesEncoder consumes variable slices and pools to [B, d_model]."""
    encoder = SeriesEncoder(in_dim=64, d_model=128, n_heads=4, n_layers=2, meta_dim=32)
    B, num_slices, D = 4, 30, 64

    slices = torch.randn(B, num_slices, D)
    mask = torch.ones(B, num_slices, dtype=torch.bool)
    mask[0, 15:] = False  # 15 slices
    mask[1, 1:] = False   # 1 slice edge case
    mask[2, 29:] = False  # 29 slices
    # mask[3] has full 30 slices

    meta = torch.randn(B, 32)

    series_vec, attn_weights = encoder(slices, slice_mask=mask, series_meta_vec=meta)

    assert series_vec.shape == (B, 128)
    assert attn_weights.shape == (B, num_slices, 1)
    assert not torch.isnan(series_vec).any()

    # Verify padded slices receive 0 attention weight
    assert attn_weights[1, 5, 0].item() == 0.0


def test_cross_series_fusion_with_masked_series():
    """Verify FusionEncoder handles variable series counts per study."""
    fusion = FusionEncoder(d_model=128, n_heads=4, n_layers=2)
    B, max_series, d_model = 3, 6, 128

    series_vecs = torch.randn(B, max_series, d_model)
    series_mask = torch.ones(B, max_series, dtype=torch.bool)
    series_mask[0, 2:] = False  # Study 0 has only 2 series
    series_mask[1, 4:] = False  # Study 1 has 4 series
    # Study 2 has full 6 series

    study_vec, series_weights = fusion(series_vecs, series_mask=series_mask)

    assert study_vec.shape == (B, 128)
    assert series_weights.shape == (B, max_series, 1)
    assert not torch.isnan(study_vec).any()

    # Masked series should have 0 attention weight
    assert series_weights[0, 4, 0].item() == 0.0


def test_end_to_end_model_shapes_and_edge_cases():
    """
    Comprehensive Phase 4 Done-When verification:
    Tests variable series count, 1-slice series, 120-slice long tail,
    and missing series mask in a single batch.
    """
    model = KneeMRIClassifier(
        slice_dim=64,
        d_model=128,
        n_heads=4,
        series_layers=2,
        fusion_layers=2,
        meta_dim=32,
        num_classes=12,
        max_slices=256,
    )

    B = 2
    max_series = 5
    max_slices = 130
    slice_dim = 64

    slice_embeddings = torch.randn(B, max_series, max_slices, slice_dim)
    slice_masks = torch.zeros(B, max_series, max_slices, dtype=torch.bool)
    series_masks = torch.zeros(B, max_series, dtype=torch.bool)
    series_meta = torch.zeros(B, max_series, 3, dtype=torch.long)

    # Study 0:
    # - Series 0: 25 slices
    # - Series 1: 1 slice (edge case: single slice)
    # - Series 2: 40 slices
    # - Series 3 & 4: MISSING / MASKED OUT
    series_masks[0, :3] = True
    slice_masks[0, 0, :25] = True
    slice_masks[0, 1, :1] = True    # 1 slice edge case
    slice_masks[0, 2, :40] = True

    # Study 1:
    # - Series 0: 30 slices
    # - Series 1: 125 slices (edge case: long tail)
    # - Series 2: 15 slices
    # - Series 3: 20 slices
    # - Series 4: 10 slices
    series_masks[1, :] = True
    slice_masks[1, 0, :30] = True
    slice_masks[1, 1, :125] = True  # long-tail tail case
    slice_masks[1, 2, :15] = True
    slice_masks[1, 3, :20] = True
    slice_masks[1, 4, :10] = True

    # Forward pass
    output = model(
        slice_embeddings=slice_embeddings,
        slice_masks=slice_masks,
        series_masks=series_masks,
        series_meta=series_meta,
    )

    logits = output["logits"]
    probs = output["probabilities"]
    study_vec = output["study_vector"]

    # Verify output shapes
    assert logits.shape == (B, 12)
    assert probs.shape == (B, 12)
    assert study_vec.shape == (B, 128)

    # Verify no NaNs or Infs
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Verify backward pass gradients compute cleanly
    dummy_targets = torch.randint(0, 2, (B, 12)).float()
    pos_weight = torch.ones(12) * 2.0
    loss = compute_multilabel_bce_loss(logits, dummy_targets, pos_weight=pos_weight)

    loss.backward()

    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_dataset_and_collate_integration():
    """Verify StudyDataset loading and collate_study_batch tensor alignment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        emb_dir = tmp_p / "embeddings"
        emb_dir.mkdir(parents=True)

        # Create mock embeddings on disk
        for study_id in ["study_a", "study_b"]:
            s_dir = emb_dir / study_id
            s_dir.mkdir(parents=True)
            # study_a has 2 series, study_b has 1 series
            np.save(str(s_dir / "series_1.npy"), np.random.randn(10, 64).astype(np.float32))
            if study_id == "study_a":
                np.save(str(s_dir / "series_2.npy"), np.random.randn(25, 64).astype(np.float32))

        labels_df = pd.DataFrame([
            {"StudyInstanceUID": "study_a", "ACL": 1, "MCL": 0, "Effusion": 1},
            {"StudyInstanceUID": "study_b", "ACL": 0, "MCL": 1, "Effusion": 0},
        ])
        series_df = pd.DataFrame([
            {"StudyInstanceUID": "study_a", "SeriesInstanceUID": "series_1", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1, "Fat_Suppression": 1},
            {"StudyInstanceUID": "study_a", "SeriesInstanceUID": "series_2", "Anatomical_Plane": "Coronal", "Fluid_Sensitive": 0, "Fat_Suppression": 0},
            {"StudyInstanceUID": "study_b", "SeriesInstanceUID": "series_1", "Anatomical_Plane": "Axial", "Fluid_Sensitive": 1, "Fat_Suppression": 0},
        ])

        dataset = StudyDataset(
            labels_df_or_path=labels_df,
            embeddings_dir=emb_dir,
            series_df_or_path=series_df,
        )

        assert len(dataset) == 2
        batch = [dataset[0], dataset[1]]
        collated = collate_study_batch(batch)

        assert collated["slice_embeddings"].shape == (2, 2, 25, 64)
        assert collated["slice_masks"].shape == (2, 2, 25)
        assert collated["series_masks"].shape == (2, 2)
        assert collated["labels"].shape == (2, 12)
        # study_b only has 1 series, so series_masks[1, 1] should be False
        assert collated["series_masks"][1, 1].item() is False
        assert collated["series_masks"][0, 1].item() is True

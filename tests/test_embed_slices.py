"""
Unit tests for Phase 3: Slice Embedding Pipeline (src/embed_slices.py & src/encoders/).
Tests encoder registry, feature dimensions, anatomical slice embedding,
idempotent disk caching, and manifest creation.
"""

import json
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

from src.encoders import get_encoder
from src.embed_slices import embed_all_series, embed_series
from tests.test_dicom_io import create_synthetic_dicom


def test_encoder_registry_and_dummy_encoding():
    """Verify registry creation and feature extraction with dummy mock encoder."""
    encoder = get_encoder("dummy_mock", img_size=224, device="cpu")
    assert encoder.name == "dummy_mock"
    assert encoder.feature_dim == 64

    # Test random batch [B, 3, 224, 224]
    fake_batch = np.random.uniform(0.0, 1.0, size=(4, 3, 224, 224)).astype(np.float32)
    features = encoder.encode(fake_batch)

    assert isinstance(features, np.ndarray)
    assert features.shape == (4, 64)
    assert features.dtype == np.float32
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_resnet50_architecture_and_forward():
    """Verify ResNet50 backbone setup without predownloaded weights."""
    encoder = get_encoder("resnet50", pretrained=False, device="cpu")
    assert encoder.feature_dim == 2048
    assert encoder.img_size == 224

    # Run forward pass on small 2-slice batch
    fake_batch = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    feats = encoder.encode(fake_batch)
    assert feats.shape == (2, 2048)


def test_embed_single_series_with_anatomical_slices():
    """Verify embed_series processes slices in order and saves .npy atomically."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        series_dir = Path(tmp_dir) / "series_01"
        series_dir.mkdir(parents=True)

        # Create 5 synthetic slices with varying z-coordinates out of filename order
        slice_specs = [
            ("slice_c.dcm", 3, 30.0),
            ("slice_a.dcm", 1, 10.0),
            ("slice_e.dcm", 5, 50.0),
            ("slice_b.dcm", 2, 20.0),
            ("slice_d.dcm", 4, 40.0),
        ]
        for name, inst_num, z_pos in slice_specs:
            create_synthetic_dicom(
                series_dir / name,
                instance_number=inst_num,
                z_position=z_pos,
            )

        encoder = get_encoder("dummy_mock", device="cpu")
        out_file = Path(tmp_dir) / "embeddings" / "series_01.npy"

        res = embed_series(
            series_dir=series_dir,
            encoder=encoder,
            output_path=out_file,
            batch_size=2,
            overwrite=False,
        )

        assert res is not None
        assert res.shape == (5, 64)
        assert out_file.exists()

        # Check loaded file
        loaded = np.load(str(out_file))
        np.testing.assert_allclose(res, loaded)


def test_embed_all_series_resumability_and_manifest():
    """Verify batch embedding skips already processed series and creates manifest.json."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        raw_dir = base_dir / "raw"
        processed_dir = base_dir / "processed"
        raw_dir.mkdir(parents=True)
        processed_dir.mkdir(parents=True)

        # Create 2 mock studies, each with 1 series and 2 slices
        train_series_rows = []
        train_series_dir = raw_dir / "train_series"

        for s_idx in [1, 2]:
            study_uid = f"study_00{s_idx}"
            series_uid = f"series_00{s_idx}"
            train_series_rows.append(
                {
                    "StudyInstanceUID": study_uid,
                    "SeriesInstanceUID": series_uid,
                    "Anatomical_Plane": "Sagittal",
                    "Fluid_Sensitive": 1,
                    "Fat_Suppression": 0,
                }
            )
            s_dir = train_series_dir / study_uid / series_uid
            s_dir.mkdir(parents=True)
            create_synthetic_dicom(s_dir / "1.dcm", instance_number=1, z_position=1.0)
            create_synthetic_dicom(s_dir / "2.dcm", instance_number=2, z_position=2.0)

        pd.DataFrame(train_series_rows).to_csv(raw_dir / "train_series.csv", index=False)

        # First run: embed all
        manifest1 = embed_all_series(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            encoder_name="dummy_mock",
            batch_size=2,
            device="cpu",
        )
        assert manifest1["embedded_count"] == 2
        assert manifest1["skipped_count"] == 0
        assert manifest1["failed_count"] == 0

        # Check manifest file on disk
        manifest_path = processed_dir / "embeddings" / "dummy_mock" / "manifest.json"
        assert manifest_path.exists()

        # Second run: must skip all already-embedded series
        manifest2 = embed_all_series(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            encoder_name="dummy_mock",
            batch_size=2,
            device="cpu",
        )
        assert manifest2["embedded_count"] == 0
        assert manifest2["skipped_count"] == 2
        assert manifest2["failed_count"] == 0

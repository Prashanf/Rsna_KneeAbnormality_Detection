"""
Unit and integration tests for Phase 5 training pipeline and evaluation metrics.
Tests metric calculation, stratified splitting, pos_weight computation,
and checkpoint save/load/resumption.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit import TARGET_LABELS
from src.evaluate import calculate_multilabel_metrics
from src.model.heads import KneeMRIClassifier
from src.train import (
    compute_class_pos_weights,
    create_stratified_split,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


def test_multilabel_metrics_calculation():
    """Verify AUROC and AP calculation on known predictions."""
    # 4 samples, 2 labels
    y_true = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    # Perfect probability predictions
    y_probs = np.array([
        [0.9, 0.1],
        [0.1, 0.8],
        [0.85, 0.95],
        [0.05, 0.05],
    ])
    metrics = calculate_multilabel_metrics(y_true, y_probs, label_names=["FindingA", "FindingB"])

    assert metrics["mean_auroc"] == 1.0
    assert metrics["mean_ap"] == 1.0
    assert metrics["valid_labels_evaluated"] == 2
    assert metrics["mean_bce_loss"] > 0.0


def test_multilabel_metrics_single_class_edge_case():
    """Verify metric calculator handles degenerate unobserved classes gracefully."""
    # Label 0 has both classes, Label 1 is all zeros
    y_true = np.array([[1, 0], [0, 0], [1, 0]])
    y_probs = np.array([[0.8, 0.2], [0.2, 0.1], [0.9, 0.3]])

    metrics = calculate_multilabel_metrics(y_true, y_probs, label_names=["FindingA", "FindingB"])
    assert metrics["per_label_auroc"]["FindingA"] == 1.0
    assert metrics["per_label_auroc"]["FindingB"] is None
    # Evaluated only over valid classes
    assert metrics["valid_labels_evaluated"] == 1
    assert metrics["mean_auroc"] == 1.0


def test_stratified_split_proportions_and_disjointness():
    """Verify stratified split creates non-overlapping splits with correct proportions."""
    rows = []
    for i in range(100):
        row = {"StudyInstanceUID": f"study_{i}"}
        for l in TARGET_LABELS:
            row[l] = int(np.random.rand() > 0.8)
        rows.append(row)
    df = pd.DataFrame(rows)

    train_df, val_df = create_stratified_split(df, val_size=0.2, seed=42)

    assert len(train_df) == 80
    assert len(val_df) == 20

    # Ensure zero overlap
    train_uids = set(train_df["StudyInstanceUID"])
    val_uids = set(val_df["StudyInstanceUID"])
    assert len(train_uids.intersection(val_uids)) == 0


def test_compute_class_pos_weights():
    """Verify calculation of pos_weight for loss balancing."""
    df = pd.DataFrame({
        "LabelA": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 pos, 8 neg -> weight = 4.0
        "LabelB": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 8 pos, 2 neg -> weight = 0.25
    })
    weights = compute_class_pos_weights(df, target_columns=["LabelA", "LabelB"])

    assert weights.shape == (2,)
    assert pytest.approx(weights[0].item(), 0.01) == 4.0
    assert pytest.approx(weights[1].item(), 0.01) == 0.25


def test_checkpoint_save_and_resumption():
    """Verify saving checkpoint and restoring model weights and optimizer state."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "checkpoint.pt"

        model = KneeMRIClassifier(slice_dim=32, d_model=64, n_heads=2, num_classes=12)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        # Record initial parameter value
        param_before = list(model.parameters())[0].clone()

        save_checkpoint(
            filepath=ckpt_path,
            epoch=3,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            metrics={"mean_auroc": 0.85},
            config={"lr": 1e-3},
            best_metric=0.85,
        )

        assert ckpt_path.exists()

        # Mutate model weights
        with torch.no_grad():
            list(model.parameters())[0].add_(1.0)

        assert not torch.allclose(list(model.parameters())[0], param_before)

        # Restore from checkpoint
        new_model = KneeMRIClassifier(slice_dim=32, d_model=64, n_heads=2, num_classes=12)
        new_optimizer = AdamW(new_model.parameters(), lr=1e-3)
        start_epoch, best_metric = load_checkpoint(ckpt_path, new_model, new_optimizer)

        assert start_epoch == 4  # Next epoch to run
        assert best_metric == 0.85
        assert torch.allclose(list(new_model.parameters())[0], param_before)


def test_full_training_loop_execution_and_resumption():
    """Verify train_loop end-to-end on mock dataset and test resumption."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        raw_dir = tmp_p / "raw"
        processed_dir = tmp_p / "processed"
        emb_dir = processed_dir / "embeddings" / "resnet50"
        raw_dir.mkdir(parents=True)
        processed_dir.mkdir(parents=True)
        emb_dir.mkdir(parents=True)

        # Create 4 studies with embeddings
        studies = ["study_1", "study_2", "study_3", "study_4"]
        label_rows = []
        series_rows = []

        for s_idx, s_id in enumerate(studies):
            s_folder = emb_dir / s_id
            s_folder.mkdir(parents=True)
            np.save(str(s_folder / "series_01.npy"), np.random.randn(8, 2048).astype(np.float32))

            row = {"StudyInstanceUID": s_id}
            for l_idx, l in enumerate(TARGET_LABELS):
                row[l] = 1 if (s_idx + l_idx) % 2 == 0 else 0
            label_rows.append(row)

            series_rows.append({
                "StudyInstanceUID": s_id,
                "SeriesInstanceUID": "series_01",
                "Anatomical_Plane": "Sagittal",
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 1,
            })

        pd.DataFrame(label_rows).to_csv(processed_dir / "train_labels_llm.csv", index=False)
        pd.DataFrame(series_rows).to_parquet(processed_dir / "series_index.parquet", index=False)

        # Write config
        cfg = {
            "data": {
                "raw_dir": str(raw_dir),
                "processed_dir": str(processed_dir),
            },
            "embedding": {
                "encoder_name": "resnet50",
            },
            "model": {
                "series_encoder": {"d_model": 64, "n_heads": 2, "n_layers": 1, "dropout": 0.0},
                "fusion": {"d_model": 64, "n_heads": 2, "n_layers": 1, "dropout": 0.0},
            },
            "training": {
                "batch_size": 2,
                "lr": 1e-3,
                "val_size": 0.5,
                "seed": 42,
                "early_stopping_patience": 5,
            },
        }
        cfg_path = tmp_p / "test_config.yaml"
        import yaml
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        # Run 2 epochs
        from src.train import train_loop
        result1 = train_loop(config_path=cfg_path, resume=False, override_epochs=2, device_name="cpu")
        assert Path(result1["best_checkpoint"]).exists()
        assert len(result1["history"]) == 2

        # Resume and run up to epoch 3
        result2 = train_loop(config_path=cfg_path, resume=True, override_epochs=3, device_name="cpu")
        assert len(result2["history"]) == 1  # Ran only epoch 3!
        assert result2["history"][0]["epoch"] == 3

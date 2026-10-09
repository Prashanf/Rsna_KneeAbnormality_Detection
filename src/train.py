"""
Phase 5: Model training, stratified splitting, AdamW optimization, and checkpointing.
Resumes automatically from checkpoints and tracks per-label AUROC/AP metrics.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit import TARGET_LABELS
from src.dataset import StudyDataset, collate_study_batch
from src.encoders.base import get_device
from src.evaluate import calculate_multilabel_metrics, evaluate_model, print_evaluation_summary
from src.model.heads import KneeMRIClassifier, compute_multilabel_bce_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_stratified_split(
    labels_df: pd.DataFrame,
    val_size: float = 0.2,
    seed: int = 42,
    target_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates study-level stratified split across 12 imbalanced binary labels.
    Uses positive label count and multi-label rarity frequency to ensure
    balanced representation in both train and validation splits.
    """
    target_columns = target_columns or TARGET_LABELS
    df = labels_df.copy().reset_index(drop=True)

    # Compute positive count and rarity score per study
    present_cols = [c for c in target_columns if c in df.columns]
    label_matrix = df[present_cols].fillna(0).astype(int).values

    # Inverse prevalence weight for each label
    col_prevalences = label_matrix.sum(axis=0) + 1.0
    label_weights = len(df) / col_prevalences

    # Weighted rarity score per study
    rarity_scores = (label_matrix * label_weights).sum(axis=1)
    df["_strata_score"] = pd.qcut(rarity_scores, q=min(10, max(2, len(df) // 10)), labels=False, duplicates="drop")

    rng = np.random.RandomState(seed)
    val_indices = []

    for _, group in df.groupby("_strata_score"):
        indices = rng.permutation(group.index.to_numpy())
        n_val = int(round(len(indices) * val_size))
        val_indices.extend(indices[:n_val])

    val_idx_set = set(val_indices)
    train_df = df[~df.index.isin(val_idx_set)].drop(columns=["_strata_score"]).reset_index(drop=True)
    val_df = df[df.index.isin(val_idx_set)].drop(columns=["_strata_score"]).reset_index(drop=True)

    logger.info(f"Split created: {len(train_df)} train studies ({100-val_size*100:.1f}%), {len(val_df)} val studies ({val_size*100:.1f}%)")
    return train_df, val_df


def compute_class_pos_weights(
    train_df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    max_weight: float = 20.0,
) -> torch.Tensor:
    """
    Computes BCE pos_weight: (num_negatives / num_positives) per class.
    Clamps extreme weights to max_weight to maintain numerical stability.
    """
    target_columns = target_columns or TARGET_LABELS
    weights = []

    for col in target_columns:
        if col in train_df.columns:
            pos = float((train_df[col] == 1).sum())
            neg = float((train_df[col] == 0).sum())
            w = neg / max(pos, 1.0)
            weights.append(min(w, max_weight))
        else:
            weights.append(1.0)

    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(
    filepath: Union[str, Path],
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    best_metric: float,
):
    """Saves complete training checkpoint with optimizer state for resumability."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "best_metric": best_metric,
            "config": config,
        },
        filepath,
    )
    logger.info(f"Checkpoint saved to: {filepath}")


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Tuple[int, float]:
    """Loads checkpoint weights and optimizer states, returning (start_epoch, best_metric)."""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_metric = checkpoint.get("best_metric", 0.0)
    logger.info(f"Resumed from {filepath} at epoch {start_epoch} (best metric: {best_metric:.4f})")
    return start_epoch, best_metric


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
    grad_clip: float = 1.0,
) -> float:
    """Runs a single training epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    for batch in dataloader:
        optimizer.zero_grad()

        slice_emb = batch["slice_embeddings"].to(device)
        slice_mask = batch["slice_masks"].to(device)
        series_mask = batch["series_masks"].to(device)
        series_meta = batch["series_meta"].to(device)
        targets = batch["labels"].to(device)

        out = model(
            slice_embeddings=slice_emb,
            slice_masks=slice_mask,
            series_masks=series_mask,
            series_meta=series_meta,
        )

        loss = loss_fn(out["logits"], targets)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_loop(
    config_path: Union[str, Path] = "configs/config.yaml",
    resume: bool = True,
    override_epochs: Optional[int] = None,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """
    Main Phase 5 training pipeline:
    - Loads config & data
    - Performs stratified splitting
    - Builds KneeMRIClassifier
    - Resumes from checkpoints if found
    - Optimizes with AdamW + Cosine LR scheduler
    - Evaluates per-label AUROC and saves best checkpoint
    """
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Extract configs with defaults
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    embed_cfg = cfg.get("embedding", {})

    encoder_name = embed_cfg.get("encoder_name", "resnet50")
    slice_dim = 2048 if "resnet" in encoder_name else (384 if "small" in encoder_name else 768)
    d_model = model_cfg.get("series_encoder", {}).get("d_model", 128)
    n_heads = model_cfg.get("series_encoder", {}).get("n_heads", 4)
    epochs = override_epochs or train_cfg.get("epochs", 15)
    batch_size = train_cfg.get("batch_size", 8)
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grad_clip = float(train_cfg.get("gradient_clip_val", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 5))

    # Resolve paths
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    embeddings_dir = processed_dir / "embeddings" / encoder_name
    checkpoints_dir = processed_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Prefer Phase 2 LLM labels if available, else original train.csv
    llm_labels_path = processed_dir / "train_labels_llm.csv"
    orig_labels_path = raw_dir / "train.csv"
    if llm_labels_path.exists():
        logger.info(f"Using Phase 2 LLM-derived labels from {llm_labels_path}")
        labels_df = pd.read_csv(llm_labels_path)
    elif orig_labels_path.exists():
        logger.info(f"Phase 2 labels not found. Falling back to original labels at {orig_labels_path}")
        labels_df = pd.read_csv(orig_labels_path)
    else:
        raise FileNotFoundError(f"Neither {llm_labels_path} nor {orig_labels_path} exists!")

    # Series metadata
    series_index_path = processed_dir / "series_index.parquet"
    series_csv_path = raw_dir / "train_series.csv"
    series_meta_path = series_index_path if series_index_path.exists() else series_csv_path

    # Stratified split
    train_df, val_df = create_stratified_split(labels_df, val_size=train_cfg.get("val_size", 0.2), seed=train_cfg.get("seed", 42))

    # Compute positive class weights
    pos_weights = compute_class_pos_weights(train_df)

    # Datasets and Loaders
    train_dataset = StudyDataset(train_df, embeddings_dir=embeddings_dir, series_df_or_path=series_meta_path)
    val_dataset = StudyDataset(val_df, embeddings_dir=embeddings_dir, series_df_or_path=series_meta_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_study_batch,
        num_workers=0,  # Single worker safe for all platforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_study_batch,
        num_workers=0,
    )

    # Device
    device = get_device(device_name)
    logger.info(f"Training on device: {device}")

    # Model
    model = KneeMRIClassifier(
        slice_dim=slice_dim,
        d_model=d_model,
        n_heads=n_heads,
        series_layers=model_cfg.get("series_encoder", {}).get("n_layers", 2),
        fusion_layers=model_cfg.get("fusion", {}).get("n_layers", 2),
        dropout=model_cfg.get("series_encoder", {}).get("dropout", 0.1),
        num_classes=12,
    ).to(device)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Resumption
    start_epoch = 1
    best_val_auroc = 0.0
    latest_ckpt_path = checkpoints_dir / "checkpoint_latest.pt"
    best_ckpt_path = checkpoints_dir / "checkpoint_best.pt"

    if resume and latest_ckpt_path.exists():
        start_epoch, best_val_auroc = load_checkpoint(latest_ckpt_path, model, optimizer, scheduler)

    # Training loop
    epochs_no_improve = 0
    history = []

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            pos_weight=pos_weights,
            grad_clip=grad_clip,
        )
        scheduler.step()

        # Validation
        val_loss, val_metrics, _, _ = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            pos_weight=pos_weights,
        )
        epoch_time = time.time() - t0
        val_auroc = val_metrics["mean_auroc"]

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) — "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean AUROC: {val_auroc:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_auroc": val_auroc,
            "val_mean_ap": val_metrics["mean_ap"],
        })

        # Save latest checkpoint
        save_checkpoint(
            filepath=latest_ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=val_metrics,
            config=cfg,
            best_metric=best_val_auroc,
        )

        # Check for best checkpoint
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_no_improve = 0
            save_checkpoint(
                filepath=best_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=val_metrics,
                config=cfg,
                best_metric=best_val_auroc,
            )
            logger.info(f"★ New best model achieved (AUROC: {best_val_auroc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    # Load and print final summary from best model
    if best_ckpt_path.exists():
        load_checkpoint(best_ckpt_path, model)
        _, final_metrics, _, _ = evaluate_model(model, val_loader, device=device)
        print_evaluation_summary(final_metrics)

    return {
        "best_val_auroc": best_val_auroc,
        "best_checkpoint": str(best_ckpt_path),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Knee MRI Multilabel Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from latest checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cuda, mps, cpu")
    args = parser.parse_args()

    train_loop(
        config_path=args.config,
        resume=not args.no_resume,
        override_epochs=args.epochs,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()

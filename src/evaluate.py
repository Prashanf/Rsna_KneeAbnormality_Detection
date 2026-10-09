"""
Evaluation metrics and validation utilities for Knee MRI Multilabel Classification.
Calculates per-label AUROC, Average Precision (AP), and macro averages.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from src.data_audit import TARGET_LABELS


def calculate_multilabel_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes per-label AUROC, Average Precision (PR-AUC), and macro summaries.

    Inputs:
        y_true: shape [N, num_labels], binary 0/1 integers
        y_probs: shape [N, num_labels], continuous probabilities in [0, 1]
        label_names: list of label names
    Returns:
        dict containing per-label metrics and macro aggregates.
    """
    label_names = label_names or TARGET_LABELS
    num_labels = len(label_names)

    per_label_auroc: Dict[str, Optional[float]] = {}
    per_label_ap: Dict[str, Optional[float]] = {}
    valid_aurocs = []
    valid_aps = []

    for idx, name in enumerate(label_names):
        col_true = y_true[:, idx]
        col_probs = y_probs[:, idx]

        # Check if fold has at least 1 positive and 1 negative instance
        num_pos = int((col_true == 1).sum())
        num_neg = int((col_true == 0).sum())

        if num_pos > 0 and num_neg > 0:
            try:
                auc = float(roc_auc_score(col_true, col_probs))
                ap = float(average_precision_score(col_true, col_probs))
                per_label_auroc[name] = auc
                per_label_ap[name] = ap
                valid_aurocs.append(auc)
                valid_aps.append(ap)
            except Exception:
                per_label_auroc[name] = None
                per_label_ap[name] = None
        else:
            # Degenerate single-class distribution in this fold
            per_label_auroc[name] = None
            per_label_ap[name] = None

    mean_auroc = float(np.mean(valid_aurocs)) if len(valid_aurocs) > 0 else 0.5
    mean_ap = float(np.mean(valid_aps)) if len(valid_aps) > 0 else 0.0

    # Overall binary log loss
    if len(y_true) > 0:
        clipped_probs = np.clip(y_probs, 1e-7, 1.0 - 1e-7)
        bce = float(-np.mean(y_true * np.log(clipped_probs) + (1.0 - y_true) * np.log(1.0 - clipped_probs)))
    else:
        bce = 0.0

    return {
        "mean_auroc": mean_auroc,
        "mean_ap": mean_ap,
        "mean_bce_loss": float(bce),
        "valid_labels_evaluated": len(valid_aurocs),
        "total_labels": num_labels,
        "per_label_auroc": per_label_auroc,
        "per_label_ap": per_label_ap,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluates model on validation DataLoader.

    Returns:
        (avg_val_loss, metrics_dict, all_preds, all_targets)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    all_preds_list = []
    all_targets_list = []

    for batch in dataloader:
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

        logits = out["logits"]
        probs = out["probabilities"]

        loss = loss_fn(logits, targets)
        total_loss += loss.item()
        num_batches += 1

        all_preds_list.append(probs.cpu().numpy())
        all_targets_list.append(targets.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    all_preds = np.concatenate(all_preds_list, axis=0) if all_preds_list else np.empty((0, 12))
    all_targets = np.concatenate(all_targets_list, axis=0) if all_targets_list else np.empty((0, 12))

    metrics = calculate_multilabel_metrics(all_targets, all_preds)
    return avg_loss, metrics, all_preds, all_targets


def print_evaluation_summary(metrics: Dict[str, Any]):
    """Pretty prints evaluation metrics to console."""
    print("\n" + "=" * 55)
    print("           VALIDATION PERFORMANCE SUMMARY")
    print("=" * 55)
    print(f"Mean AUROC:     {metrics['mean_auroc']:.4f}")
    print(f"Mean AP (PR):   {metrics['mean_ap']:.4f}")
    print(f"BCE Log-Loss:   {metrics['mean_bce_loss']:.4f}")
    print("-" * 55)
    print(f"{'Finding':<20} | {'AUROC':<10} | {'AP':<10}")
    print("-" * 55)
    for name in TARGET_LABELS:
        auc = metrics["per_label_auroc"].get(name)
        ap = metrics["per_label_ap"].get(name)
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        ap_str = f"{ap:.4f}" if ap is not None else "N/A"
        print(f"{name:<20} | {auc_str:<10} | {ap_str:<10}")
    print("=" * 55 + "\n")

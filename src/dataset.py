"""
PyTorch Dataset and dynamic collation for Knee MRI studies.
Loads precomputed slice embeddings, series metadata, and multilabel targets.
Handles variable series counts and variable slice sequences per study.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_audit import TARGET_LABELS


PLANE_TO_IDX = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
    "unknown": 3,
}


class StudyDataset(Dataset):
    """
    Loads precomputed slice embeddings for all series belonging to a study.
    Each study returns its list of series embedding arrays and metadata.
    """

    def __init__(
        self,
        labels_df_or_path: Union[pd.DataFrame, str, Path],
        embeddings_dir: Union[str, Path],
        series_df_or_path: Union[pd.DataFrame, str, Path],
        is_test: bool = False,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.is_test = is_test

        # Load labels DataFrame
        if isinstance(labels_df_or_path, (str, Path)):
            self.labels_df = pd.read_csv(labels_df_or_path)
        else:
            self.labels_df = labels_df_or_path.copy()

        # Load series metadata DataFrame
        if isinstance(series_df_or_path, (str, Path)):
            p = Path(series_df_or_path)
            if p.suffix == ".parquet":
                self.series_df = pd.read_parquet(p)
            else:
                self.series_df = pd.read_csv(p)
        else:
            self.series_df = series_df_or_path.copy()

        self.series_df["StudyInstanceUID"] = self.series_df["StudyInstanceUID"].astype(str)
        self.series_df["SeriesInstanceUID"] = self.series_df["SeriesInstanceUID"].astype(str)

        # Index studies
        self.study_uids = self.labels_df["StudyInstanceUID"].astype(str).tolist()

        # Group series by study for O(1) lookup
        self.study_to_series = {}
        for study_id, group in self.series_df.groupby("StudyInstanceUID"):
            self.study_to_series[study_id] = group

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        study_uid = self.study_uids[idx]
        study_series_group = self.study_to_series.get(study_uid, pd.DataFrame())

        series_items = []
        for _, s_row in study_series_group.iterrows():
            series_uid = str(s_row["SeriesInstanceUID"])
            emb_file = self.embeddings_dir / study_uid / f"{series_uid}.npy"

            # Parse metadata
            plane_str = str(s_row.get("Anatomical_Plane", "unknown")).lower()
            plane_idx = PLANE_TO_IDX.get(plane_str, 3)
            fluid_idx = int(s_row.get("Fluid_Sensitive", 0)) if pd.notna(s_row.get("Fluid_Sensitive")) else 0
            fat_idx = int(s_row.get("Fat_Suppression", 0)) if pd.notna(s_row.get("Fat_Suppression")) else 0

            meta = (plane_idx, fluid_idx, fat_idx)

            if emb_file.exists():
                embed_arr = np.load(str(emb_file))  # [num_slices, D]
            else:
                # Fallback if series file not found (e.g. mock test)
                embed_arr = np.zeros((1, 128), dtype=np.float32)

            series_items.append((series_uid, embed_arr, meta))

        # Target label vector
        if not self.is_test:
            label_row = self.labels_df[self.labels_df["StudyInstanceUID"] == study_uid].iloc[0]
            label_vals = [
                float(label_row[lbl]) if (lbl in label_row and pd.notna(label_row[lbl])) else 0.0
                for lbl in TARGET_LABELS
            ]
            labels = torch.tensor(label_vals, dtype=torch.float32)
        else:
            labels = torch.zeros(len(TARGET_LABELS), dtype=torch.float32)

        return {
            "study_uid": study_uid,
            "series_items": series_items,
            "labels": labels,
        }


def collate_study_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function handling variable number of series per study
    and variable number of slices per series with boolean masks.
    """
    B = len(batch)

    # 1. Determine batch dimensions
    max_series = max(max(len(item["series_items"]), 1) for item in batch)

    # Find max slices and feature dimension D
    max_slices = 1
    slice_dim = 128
    for item in batch:
        for _, arr, _ in item["series_items"]:
            if len(arr) > 0:
                max_slices = max(max_slices, arr.shape[0])
                slice_dim = arr.shape[1]

    # 2. Allocate padded tensors
    slice_embeddings = torch.zeros((B, max_series, max_slices, slice_dim), dtype=torch.float32)
    slice_masks = torch.zeros((B, max_series, max_slices), dtype=torch.bool)
    series_masks = torch.zeros((B, max_series), dtype=torch.bool)
    series_meta = torch.zeros((B, max_series, 3), dtype=torch.long)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    study_uids = [item["study_uid"] for item in batch]

    # 3. Populate batch
    for b_idx, item in enumerate(batch):
        for s_idx, (s_uid, arr, meta) in enumerate(item["series_items"]):
            num_slices = arr.shape[0]
            slice_embeddings[b_idx, s_idx, :num_slices, :] = torch.from_numpy(arr)
            slice_masks[b_idx, s_idx, :num_slices] = True
            series_masks[b_idx, s_idx] = True
            series_meta[b_idx, s_idx] = torch.tensor(meta, dtype=torch.long)

    return {
        "slice_embeddings": slice_embeddings,
        "slice_masks": slice_masks,
        "series_masks": series_masks,
        "series_meta": series_meta,
        "labels": labels,
        "study_uids": study_uids,
    }

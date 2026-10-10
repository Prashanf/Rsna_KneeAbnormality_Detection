"""
Phase 3: Offline slice feature extraction and embedding caching.
Encodes DICOM slices with vision backbones (ResNet50 / DINOv2),
preserves anatomical z-axis order, and incrementally caches [num_slices, D]
feature matrices to data/processed/embeddings/<encoder>/<study>/<series>.npy.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dicom_io import load_series_slices, read_dicom_slice
from src.encoders import BaseSliceEncoder, get_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def embed_series(
    series_dir: Union[str, Path],
    encoder: BaseSliceEncoder,
    output_path: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    overwrite: bool = False,
) -> Optional[np.ndarray]:
    """
    Extracts fixed-dim feature vectors for every slice in a single MRI series.
    Slices are loaded in true anatomical order (ImagePositionPatient Z-coord).
    Saves atomically to output_path if provided.

    Returns:
        np.ndarray of shape [num_slices, feature_dim] or None if series is empty.
    """
    series_path = Path(series_dir)
    if output_path is not None:
        out_p = Path(output_path)
        if out_p.exists() and not overwrite:
            # Idempotent skip: load and return cached
            return np.load(str(out_p))

    # 1. Retrieve all slices sorted anatomically
    slice_files = load_series_slices(series_path)
    if not slice_files:
        logger.warning(f"No DICOM slices found in {series_path}")
        return None

    # 2. Process slices in batches through encoder
    img_size = encoder.img_size
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(slice_files), batch_size):
        batch_paths = slice_files[i : i + batch_size]
        batch_images = []

        for f in batch_paths:
            try:
                img_3ch = read_dicom_slice(
                    f,
                    target_size=(img_size, img_size),
                    normalize=True,
                    to_rgb=True,
                )
                batch_images.append(img_3ch)
            except Exception as e:
                logger.error(f"Failed decoding slice {f.name}: {e}. Replacing with zero frame.")
                batch_images.append(np.zeros((3, img_size, img_size), dtype=np.float32))

        # Shape [B, 3, H, W]
        batch_arr = np.stack(batch_images, axis=0)

        # Encode through model
        batch_feats = encoder.encode(batch_arr)
        all_embeddings.append(batch_feats)

    # 3. Concatenate all slice embeddings -> shape [num_slices, feature_dim]
    series_matrix = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # Sanity checks
    if np.isnan(series_matrix).any() or np.isinf(series_matrix).any():
        raise ValueError(f"Series {series_path.name} produced NaN/Inf embeddings!")

    expected_slices = len(slice_files)
    if series_matrix.shape != (expected_slices, encoder.feature_dim):
        raise ValueError(
            f"Shape mismatch: expected ({expected_slices}, {encoder.feature_dim}), got {series_matrix.shape}"
        )

    # 4. Atomic disk persistence
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Write to temporary file first then replace to prevent corrupted half-writes on crash
        tmp_fd, tmp_file = tempfile.mkstemp(dir=out_p.parent, suffix=".tmp.npy")
        os.close(tmp_fd)
        try:
            np.save(tmp_file, series_matrix)
            os.replace(tmp_file, str(out_p))
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    return series_matrix


def embed_all_series(
    raw_dir: Union[str, Path] = "data/raw",
    processed_dir: Union[str, Path] = "data/processed",
    encoder_name: str = "resnet50",
    train_series_csv: str = "train_series.csv",
    train_series_dir: str = "train_series",
    batch_size: int = 32,
    overwrite: bool = False,
    limit: Optional[int] = None,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Main Phase 3 runner: iterates over all series in the dataset,
    encodes each with the chosen backbone, and saves embeddings incrementally.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    embeddings_root = processed_path / "embeddings" / encoder_name
    embeddings_root.mkdir(parents=True, exist_ok=True)

    # Load series list (prefer series_index.parquet if available from Phase 1)
    series_index_path = processed_path / "series_index.parquet"
    if series_index_path.exists():
        logger.info(f"Loading series list from {series_index_path}")
        series_df = pd.read_parquet(series_index_path)
    else:
        csv_path = raw_path / train_series_csv
        logger.info(f"Loading series list from {csv_path}")
        series_df = pd.read_csv(csv_path)

    if limit is not None:
        series_df = series_df.iloc[:limit].copy()

    total_series = len(series_df)
    logger.info(f"Preparing to embed {total_series} series with '{encoder_name}'...")

    # Instantiate encoder
    encoder = get_encoder(name=encoder_name, device=device)
    logger.info(f"Initialized encoder '{encoder.name}' (dim={encoder.feature_dim}, size={encoder.img_size}) on {encoder.target_device}")

    # Check already completed series
    skipped_count = 0
    embedded_count = 0
    failed_count = 0

    pbar = tqdm(series_df.iterrows(), total=total_series, desc=f"Embedding with {encoder_name}")
    for _, row in pbar:
        study_uid = str(row["StudyInstanceUID"])
        series_uid = str(row["SeriesInstanceUID"])
        out_file = embeddings_root / study_uid / f"{series_uid}.npy"

        if out_file.exists() and not overwrite:
            skipped_count += 1
            continue

        s_dir = raw_path / train_series_dir / study_uid / series_uid
        if not s_dir.exists():
            for alt in [
                raw_path / study_uid / series_uid,
                raw_path / "train" / study_uid / series_uid,
                raw_path / "train_images" / study_uid / series_uid,
            ]:
                if alt.exists():
                    s_dir = alt
                    break
        try:
            res = embed_series(
                series_dir=s_dir,
                encoder=encoder,
                output_path=out_file,
                batch_size=batch_size,
                overwrite=overwrite,
            )
            if res is not None:
                embedded_count += 1
            else:
                failed_count += 1
        except Exception as e:
            logger.error(f"Error embedding series {study_uid}/{series_uid}: {e}")
            failed_count += 1

    # Save manifest
    manifest = {
        "encoder_name": encoder_name,
        "feature_dim": encoder.feature_dim,
        "img_size": encoder.img_size,
        "total_series": total_series,
        "embedded_count": embedded_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "timestamp": time.time(),
    }
    manifest_path = embeddings_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Phase 3 Embedding Summary: {embedded_count} newly embedded, {skipped_count} skipped, {failed_count} failed.")
    logger.info(f"Manifest written to: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Run Knee MRI Slice Feature Extraction")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw dataset")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed output")
    parser.add_argument("--encoder", type=str, default="resnet50", help="Encoder: resnet50, dinov2_small, dinov2_base")
    parser.add_argument("--batch-size", type=int, default=32, help="Slice batch size")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cuda, mps, cpu")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of series to embed")
    args = parser.parse_args()

    embed_all_series(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        encoder_name=args.encoder,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        limit=args.limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()

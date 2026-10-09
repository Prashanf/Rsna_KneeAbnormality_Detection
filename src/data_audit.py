"""
Phase 1: Data structure and integrity audit for Knee MRI Multilabel Classification.
Verifies disk vs CSV parity, orphan files, slice counts, transfer syntax decodability,
and saves statistical distributions to data/processed/integrity_report.json.
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dicom_io import extract_pixel_array, get_dicom_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


class DataAuditor:
    """
    Performs comprehensive verification of the Knee MRI dataset.
    Can be run locally or inside a Kaggle notebook.
    """

    def __init__(
        self,
        raw_dir: Union[str, Path],
        processed_dir: Union[str, Path] = "data/processed",
        train_csv: str = "train.csv",
        train_series_csv: str = "train_series.csv",
        train_series_dir: str = "train_series",
        sample_slices_to_decode: int = 2,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.train_csv_path = self.raw_dir / train_csv
        self.train_series_csv_path = self.raw_dir / train_series_csv
        self.series_folder_path = self.raw_dir / train_series_dir
        self.sample_slices_to_decode = sample_slices_to_decode

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.report: Dict[str, Any] = {
            "status": "NOT_RUN",
            "checks": {},
            "distributions": {},
            "flagged_issues": [],
            "summary": {},
        }

    def run_audit(self) -> Dict[str, Any]:
        """Runs all 7 checks and produces integrity report."""
        logger.info(f"Starting Data Audit on raw_dir: {self.raw_dir}")

        # Basic path verification
        if not self.train_csv_path.exists():
            raise FileNotFoundError(f"train.csv not found at {self.train_csv_path}")
        if not self.train_series_csv_path.exists():
            raise FileNotFoundError(f"train_series.csv not found at {self.train_series_csv_path}")
        if not self.series_folder_path.exists():
            raise FileNotFoundError(f"train_series folder not found at {self.series_folder_path}")

        # Load CSVs
        train_df = pd.read_csv(self.train_csv_path)
        series_df = pd.read_csv(self.train_series_csv_path)

        train_studies = set(train_df["StudyInstanceUID"].astype(str))
        series_studies = set(series_df["StudyInstanceUID"].astype(str))

        logger.info(f"Loaded train.csv: {len(train_df)} studies")
        logger.info(f"Loaded train_series.csv: {len(series_df)} series entries ({len(series_studies)} unique studies)")

        # -------------------------------------------------------------
        # Check 1 & 6: Cross-check train.csv studies vs train_series.csv and disk
        # -------------------------------------------------------------
        logger.info("Executing Check 1 & 6: Study-level parity...")
        studies_in_train_not_in_series = list(train_studies - series_studies)
        studies_in_series_not_in_train = list(series_studies - train_studies)

        disk_study_dirs = {
            p.name for p in self.series_folder_path.iterdir() if p.is_dir() and not p.name.startswith(".")
        }

        train_studies_missing_on_disk = list(train_studies - disk_study_dirs)
        disk_studies_not_in_train = list(disk_study_dirs - train_studies)

        self.report["checks"]["check_1_train_studies_have_disk_folders"] = {
            "passed": len(train_studies_missing_on_disk) == 0,
            "missing_count": len(train_studies_missing_on_disk),
            "missing_examples": train_studies_missing_on_disk[:10],
        }

        self.report["checks"]["check_6_train_vs_series_study_parity"] = {
            "passed": (len(studies_in_train_not_in_series) == 0 and len(studies_in_series_not_in_train) == 0),
            "studies_in_train_not_in_series": len(studies_in_train_not_in_series),
            "studies_in_series_not_in_train": len(studies_in_series_not_in_train),
        }

        # -------------------------------------------------------------
        # Check 2 & 3 & 4 & 5: Series and slice verification
        # -------------------------------------------------------------
        logger.info("Executing Checks 2, 3, 4, 5: Series folders, slice counts, decodability...")

        missing_series_folders = []
        series_zero_slices = []
        series_large_slice_count = []
        unreadable_slices = []
        transfer_syntaxes = Counter()
        slice_counts = []
        series_index_records = []

        # Find all actual series directories on disk to check for orphans
        disk_series_pairs = set()
        for study_dir in self.series_folder_path.iterdir():
            if study_dir.is_dir() and not study_dir.name.startswith("."):
                study_uid = study_dir.name
                for s_dir in study_dir.iterdir():
                    if s_dir.is_dir() and not s_dir.name.startswith("."):
                        disk_series_pairs.add((study_uid, s_dir.name))

        csv_series_pairs = set(
            zip(
                series_df["StudyInstanceUID"].astype(str),
                series_df["SeriesInstanceUID"].astype(str),
            )
        )

        missing_series_on_disk = list(csv_series_pairs - disk_series_pairs)
        orphan_series_on_disk = list(disk_series_pairs - csv_series_pairs)

        self.report["checks"]["check_2_series_subfolders_exist"] = {
            "passed": len(missing_series_on_disk) == 0,
            "missing_count": len(missing_series_on_disk),
            "missing_examples": [f"{s}/{se}" for s, se in missing_series_on_disk[:10]],
        }

        self.report["checks"]["check_3_no_orphan_series_on_disk"] = {
            "passed": len(orphan_series_on_disk) == 0,
            "orphan_count": len(orphan_series_on_disk),
            "orphan_examples": [f"{s}/{se}" for s, se in orphan_series_on_disk[:10]],
        }

        # Scan each series listed in train_series.csv
        pbar = tqdm(series_df.iterrows(), total=len(series_df), desc="Auditing series")
        for _, row in pbar:
            study_uid = str(row["StudyInstanceUID"])
            series_uid = str(row["SeriesInstanceUID"])
            series_path = self.series_folder_path / study_uid / series_uid

            if not series_path.exists():
                missing_series_folders.append(f"{study_uid}/{series_uid}")
                continue

            dcm_files = [f for f in series_path.iterdir() if f.suffix.lower() == ".dcm" and not f.name.startswith(".")]
            n_slices = len(dcm_files)
            slice_counts.append(n_slices)

            if n_slices == 0:
                series_zero_slices.append(f"{study_uid}/{series_uid}")
            elif n_slices > 120:  # Review candidate threshold for long tail
                series_large_slice_count.append({"series": f"{study_uid}/{series_uid}", "slice_count": n_slices})

            # Check 5: Decode test on a sample of slices
            sample_files = dcm_files[: self.sample_slices_to_decode]
            sample_transfer_syntax = "Unknown"
            is_valid = True

            for dcm_f in sample_files:
                try:
                    dcm = pydicom.dcmread(str(dcm_f), stop_before_pixels=False)
                    ts = getattr(getattr(dcm, "file_meta", None), "TransferSyntaxUID", "Unknown")
                    sample_transfer_syntax = str(ts)
                    transfer_syntaxes[sample_transfer_syntax] += 1

                    # Pixel array decode test
                    arr = extract_pixel_array(dcm)
                    if arr is None or arr.size == 0 or np.isnan(arr).any():
                        unreadable_slices.append(
                            {"file": str(dcm_f), "reason": "empty or NaN array", "syntax": str(ts)}
                        )
                        is_valid = False
                except Exception as e:
                    unreadable_slices.append({"file": str(dcm_f), "reason": str(e), "syntax": "FailedToRead"})
                    is_valid = False

            series_index_records.append(
                {
                    "StudyInstanceUID": study_uid,
                    "SeriesInstanceUID": series_uid,
                    "Anatomical_Plane": str(row.get("Anatomical_Plane", "Unknown")),
                    "Fluid_Sensitive": int(row.get("Fluid_Sensitive", 0)) if pd.notna(row.get("Fluid_Sensitive")) else 0,
                    "Fat_Suppression": int(row.get("Fat_Suppression", 0)) if pd.notna(row.get("Fat_Suppression")) else 0,
                    "slice_count": n_slices,
                    "transfer_syntax": sample_transfer_syntax,
                    "is_valid": is_valid,
                }
            )

        self.report["checks"]["check_4_slice_counts"] = {
            "passed": len(series_zero_slices) == 0,
            "zero_slice_series_count": len(series_zero_slices),
            "zero_slice_series": series_zero_slices[:10],
            "large_slice_series_count": len(series_large_slice_count),
            "large_slice_examples": series_large_slice_count[:5],
        }

        self.report["checks"]["check_5_dicom_decoding"] = {
            "passed": len(unreadable_slices) == 0,
            "unreadable_count": len(unreadable_slices),
            "failures": unreadable_slices[:10],
            "transfer_syntaxes_encountered": dict(transfer_syntaxes),
        }

        # -------------------------------------------------------------
        # Check 7: Compute distributions
        # -------------------------------------------------------------
        logger.info("Executing Check 7: Computing distributions...")
        series_per_study = series_df.groupby("StudyInstanceUID").size().tolist()

        label_prevalences = {}
        for col in TARGET_LABELS:
            if col in train_df.columns:
                val_counts = train_df[col].value_counts(dropna=False).to_dict()
                pos_rate = float(train_df[col].mean()) if pd.api.types.is_numeric_dtype(train_df[col]) else None
                label_prevalences[col] = {"counts": {str(k): int(v) for k, v in val_counts.items()}, "positive_rate": pos_rate}

        self.report["distributions"] = {
            "series_per_study": {
                "min": int(np.min(series_per_study)) if series_per_study else 0,
                "median": float(np.median(series_per_study)) if series_per_study else 0,
                "max": int(np.max(series_per_study)) if series_per_study else 0,
                "counts_distribution": dict(pd.Series(series_per_study).value_counts().sort_index()),
            },
            "slices_per_series": {
                "min": int(np.min(slice_counts)) if slice_counts else 0,
                "median": float(np.median(slice_counts)) if slice_counts else 0,
                "max": int(np.max(slice_counts)) if slice_counts else 0,
                "percentile_25": float(np.percentile(slice_counts, 25)) if slice_counts else 0,
                "percentile_75": float(np.percentile(slice_counts, 75)) if slice_counts else 0,
            },
            "anatomical_planes": series_df["Anatomical_Plane"].value_counts(dropna=False).to_dict()
            if "Anatomical_Plane" in series_df.columns
            else {},
            "fluid_sensitive": series_df["Fluid_Sensitive"].value_counts(dropna=False).to_dict()
            if "Fluid_Sensitive" in series_df.columns
            else {},
            "fat_suppression": series_df["Fat_Suppression"].value_counts(dropna=False).to_dict()
            if "Fat_Suppression" in series_df.columns
            else {},
            "label_prevalence_raw_train_csv": label_prevalences,
        }

        # Overall pass/fail
        all_hard_passed = (
            self.report["checks"]["check_1_train_studies_have_disk_folders"]["passed"]
            and self.report["checks"]["check_2_series_subfolders_exist"]["passed"]
            and self.report["checks"]["check_4_slice_counts"]["passed"]
            and self.report["checks"]["check_5_dicom_decoding"]["passed"]
        )

        self.report["status"] = "PASSED" if all_hard_passed else "FAILED"
        self.report["summary"] = {
            "total_train_studies": len(train_studies),
            "total_series": len(series_df),
            "all_hard_checks_passed": all_hard_passed,
        }

        # Save integrity report
        report_out = self.processed_dir / "integrity_report.json"
        with open(report_out, "w") as f:
            json.dump(self.report, f, indent=2, default=str)
        logger.info(f"Integrity report written to: {report_out}")

        # Save series index parquet
        if series_index_records:
            series_index_df = pd.DataFrame(series_index_records)
            parquet_out = self.processed_dir / "series_index.parquet"
            series_index_df.to_parquet(parquet_out, index=False)
            logger.info(f"Saved series index ({len(series_index_df)} rows) to: {parquet_out}")

        self.print_summary()
        return self.report

    def print_summary(self):
        """Prints a human-readable audit summary to console."""
        print("\n" + "=" * 60)
        print("          KNEE MRI DATA INTEGRITY AUDIT SUMMARY")
        print("=" * 60)
        print(f"Overall Status: {self.report['status']}")
        for check_name, res in self.report["checks"].items():
            status_str = "✓ PASS" if res.get("passed", False) else "✗ FAIL"
            print(f"  [{status_str}] {check_name}")

        dists = self.report.get("distributions", {})
        if "series_per_study" in dists:
            sps = dists["series_per_study"]
            print(f"\nSeries Per Study: min={sps['min']}, median={sps['median']}, max={sps['max']}")
        if "slices_per_series" in dists:
            sps = dists["slices_per_series"]
            print(f"Slices Per Series: min={sps['min']}, median={sps['median']}, max={sps['max']}")
        print("=" * 60 + "\n")


def run_integrity_audit(
    raw_dir: Union[str, Path] = "data/raw",
    processed_dir: Union[str, Path] = "data/processed",
    sample_slices: int = 2,
    train_csv: str = "train.csv",
    train_series_csv: str = "train_series.csv",
    train_series_dir: str = "train_series",
) -> Dict[str, Any]:
    """
    Top-level helper function to run the full Phase 1 Data Integrity Audit.
    Used by notebooks and external scripts.
    """
    auditor = DataAuditor(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        train_csv=train_csv,
        train_series_csv=train_series_csv,
        train_series_dir=train_series_dir,
        sample_slices_to_decode=sample_slices,
    )
    return auditor.run_audit()


def main():
    parser = argparse.ArgumentParser(description="Run Knee MRI Data Integrity Audit")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed output directory")
    parser.add_argument("--sample-slices", type=int, default=2, help="Number of slices to test-decode per series")
    args = parser.parse_args()

    run_integrity_audit(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        sample_slices=args.sample_slices,
    )


if __name__ == "__main__":
    main()

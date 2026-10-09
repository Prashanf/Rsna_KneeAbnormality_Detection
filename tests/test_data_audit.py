"""
Unit tests for Phase 1 Data Auditor (src/data_audit.py).
Tests mock dataset audits, pass conditions, missing study detection,
orphan series detection, and zero-slice series detection.
"""

import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit import DataAuditor, TARGET_LABELS
from tests.test_dicom_io import create_synthetic_dicom


def create_mock_dataset(
    base_dir: Path,
    num_studies: int = 2,
    series_per_study: int = 2,
    slices_per_series: int = 3,
    add_missing_disk_study: bool = False,
    add_orphan_series: bool = False,
    add_empty_series: bool = False,
):
    """Creates a miniature mock dataset with CSVs and synthetic DICOM files."""
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_series_dir = raw_dir / "train_series"
    train_series_dir.mkdir(parents=True, exist_ok=True)

    study_rows = []
    series_rows = []

    for i in range(num_studies):
        study_uid = f"mock.study.{i+1}"
        row = {"StudyInstanceUID": study_uid, "Report": f"Patient knee report {i+1}."}
        for label in TARGET_LABELS:
            row[label] = np.random.choice([0, 1])
        study_rows.append(row)

        for s in range(series_per_study):
            series_uid = f"mock.series.{i+1}.{s+1}"
            series_rows.append(
                {
                    "StudyInstanceUID": study_uid,
                    "SeriesInstanceUID": series_uid,
                    "Anatomical_Plane": "Sagittal" if s == 0 else "Coronal",
                    "Fluid_Sensitive": 1 if s == 0 else 0,
                    "Fat_Suppression": 1,
                }
            )

            # Create series directory and DICOM slices
            s_dir = train_series_dir / study_uid / series_uid
            s_dir.mkdir(parents=True, exist_ok=True)

            num_slices = 0 if (add_empty_series and i == 0 and s == 0) else slices_per_series
            for slice_idx in range(num_slices):
                dcm_file = s_dir / f"slice_{slice_idx+1}.dcm"
                create_synthetic_dicom(
                    dcm_file,
                    instance_number=slice_idx + 1,
                    z_position=float(slice_idx * 3.0),
                )

    if add_missing_disk_study:
        # Add a study to CSV that has no folder on disk
        missing_uid = "mock.study.missing"
        missing_row = {"StudyInstanceUID": missing_uid, "Report": "Missing study report."}
        for label in TARGET_LABELS:
            missing_row[label] = 0
        study_rows.append(missing_row)
        series_rows.append(
            {
                "StudyInstanceUID": missing_uid,
                "SeriesInstanceUID": "mock.series.missing.1",
                "Anatomical_Plane": "Axial",
                "Fluid_Sensitive": 0,
                "Fat_Suppression": 0,
            }
        )

    if add_orphan_series:
        # Add a folder on disk not in CSV
        orphan_dir = train_series_dir / "mock.study.1" / "mock.series.orphan"
        orphan_dir.mkdir(parents=True, exist_ok=True)
        create_synthetic_dicom(orphan_dir / "orphan.dcm")

    # Write CSVs
    pd.DataFrame(study_rows).to_csv(raw_dir / "train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(raw_dir / "train_series.csv", index=False)

    return raw_dir, processed_dir


def test_audit_passes_on_valid_dataset():
    """Verify that auditor completes and all checks pass on a sound dataset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir, processed_dir = create_mock_dataset(Path(tmp_dir))
        auditor = DataAuditor(raw_dir=raw_dir, processed_dir=processed_dir)
        report = auditor.run_audit()

        assert report["status"] == "PASSED"
        assert report["checks"]["check_1_train_studies_have_disk_folders"]["passed"] is True
        assert report["checks"]["check_2_series_subfolders_exist"]["passed"] is True
        assert report["checks"]["check_3_no_orphan_series_on_disk"]["passed"] is True
        assert report["checks"]["check_4_slice_counts"]["passed"] is True
        assert report["checks"]["check_5_dicom_decoding"]["passed"] is True
        assert report["checks"]["check_6_train_vs_series_study_parity"]["passed"] is True

        # Check outputs exist
        assert (processed_dir / "integrity_report.json").exists()
        assert (processed_dir / "series_index.parquet").exists()


def test_audit_detects_missing_study_folder():
    """Verify that auditor catches studies missing their folders on disk."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir, processed_dir = create_mock_dataset(Path(tmp_dir), add_missing_disk_study=True)
        auditor = DataAuditor(raw_dir=raw_dir, processed_dir=processed_dir)
        report = auditor.run_audit()

        assert report["status"] == "FAILED"
        check1 = report["checks"]["check_1_train_studies_have_disk_folders"]
        assert check1["passed"] is False
        assert check1["missing_count"] == 1
        assert "mock.study.missing" in check1["missing_examples"]


def test_audit_detects_orphan_series():
    """Verify that auditor flags orphan series on disk not listed in CSV."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir, processed_dir = create_mock_dataset(Path(tmp_dir), add_orphan_series=True)
        auditor = DataAuditor(raw_dir=raw_dir, processed_dir=processed_dir)
        report = auditor.run_audit()

        check3 = report["checks"]["check_3_no_orphan_series_on_disk"]
        assert check3["passed"] is False
        assert check3["orphan_count"] == 1


def test_audit_detects_zero_slice_series():
    """Verify that auditor flags series with 0 slices."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir, processed_dir = create_mock_dataset(Path(tmp_dir), add_empty_series=True)
        auditor = DataAuditor(raw_dir=raw_dir, processed_dir=processed_dir)
        report = auditor.run_audit()

        assert report["status"] == "FAILED"
        check4 = report["checks"]["check_4_slice_counts"]
        assert check4["passed"] is False
        assert check4["zero_slice_series_count"] == 1

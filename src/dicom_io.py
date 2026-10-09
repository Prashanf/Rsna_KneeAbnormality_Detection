"""
DICOM I/O and preprocessing utilities for Knee MRI Multilabel Classification.
Handles multi-syntax decoding (uncompressed, JPEG Lossless, JPEG 2000),
intensity normalization, anatomical sorting, and channel formatting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pydicom
try:
    from pydicom.pixels import apply_voi_lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image


def get_slice_order_key(dcm: pydicom.Dataset) -> float:
    """
    Extract anatomical ordering key for a DICOM slice.
    Prefers ImagePositionPatient z-coordinate (true 3D position);
    falls back to InstanceNumber or 0.0.
    """
    if hasattr(dcm, "ImagePositionPatient") and dcm.ImagePositionPatient is not None:
        try:
            return float(dcm.ImagePositionPatient[2])
        except (IndexError, TypeError, ValueError):
            pass

    if hasattr(dcm, "InstanceNumber") and dcm.InstanceNumber is not None:
        try:
            return float(dcm.InstanceNumber)
        except (TypeError, ValueError):
            pass

    return 0.0


def extract_pixel_array(dcm: pydicom.Dataset) -> np.ndarray:
    """
    Safely extract and scale raw pixel array from a pydicom Dataset.
    Handles VOI LUT, RescaleSlope/Intercept, and MONOCHROME1 inversion.
    """
    try:
        # Try VOI LUT first (standard windowing applied by scanner/protocol)
        img = apply_voi_lut(dcm.pixel_array, dcm)
        img = img.astype(np.float32)
    except Exception:
        # Fall back to raw array + slope/intercept rescale
        img = dcm.pixel_array.astype(np.float32)
        slope = getattr(dcm, "RescaleSlope", 1)
        intercept = getattr(dcm, "RescaleIntercept", 0)
        try:
            slope = float(slope)
            intercept = float(intercept)
            img = img * slope + intercept
        except (TypeError, ValueError):
            pass

    # Handle Photometric Interpretation:
    # MONOCHROME1: 0 represents white (maximum intensity), needs inversion
    photometric = getattr(dcm, "PhotometricInterpretation", "")
    if photometric == "MONOCHROME1":
        img = np.amax(img) - img

    return img


def normalize_intensity(
    img: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """
    Robust intensity normalization using percentile clipping.
    Clips extreme outliers and scales linearly to [0.0, 1.0].
    """
    p_low = np.percentile(img, percentile_low)
    p_high = np.percentile(img, percentile_high)

    if p_high > p_low:
        clipped = np.clip(img, p_low, p_high)
        normalized = (clipped - p_low) / (p_high - p_low)
    else:
        # Constant or degenerated slice
        normalized = np.zeros_like(img, dtype=np.float32)

    return normalized.astype(np.float32)


def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize 2D image (H, W) to (target_size[0], target_size[1]) using bilinear interpolation.
    """
    pil_img = Image.fromarray((img * 255.0).clip(0, 255).astype(np.uint8))
    pil_resized = pil_img.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
    return np.array(pil_resized, dtype=np.float32) / 255.0


def read_dicom_slice(
    path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = (224, 224),
    normalize: bool = True,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    to_rgb: bool = True,
) -> np.ndarray:
    """
    Reads a single DICOM slice, decodes pixel data, normalizes, resizes,
    and returns a float32 array in [0, 1].

    Returns:
        np.ndarray:
            - If to_rgb is True: shape (3, H, W)
            - If to_rgb is False: shape (H, W)
    """
    path = Path(path)
    dcm = pydicom.dcmread(str(path))
    img = extract_pixel_array(dcm)

    if normalize:
        img = normalize_intensity(img, percentile_low, percentile_high)

    if target_size is not None and (img.shape[0] != target_size[0] or img.shape[1] != target_size[1]):
        img = resize_image(img, target_size)

    if to_rgb:
        # Replicate 1 channel to 3 channels: (3, H, W)
        return np.stack([img, img, img], axis=0)

    return img


def get_dicom_metadata(path_or_dcm: Union[str, Path, pydicom.Dataset]) -> Dict[str, Union[str, int, float]]:
    """
    Extracts essential metadata tags for data auditing and series validation.
    """
    if isinstance(path_or_dcm, (str, Path)):
        dcm = pydicom.dcmread(str(path_or_dcm), stop_before_pixels=True)
    else:
        dcm = path_or_dcm

    transfer_syntax = getattr(getattr(dcm, "file_meta", None), "TransferSyntaxUID", "Unknown")
    
    return {
        "StudyInstanceUID": str(getattr(dcm, "StudyInstanceUID", "")),
        "SeriesInstanceUID": str(getattr(dcm, "SeriesInstanceUID", "")),
        "SOPInstanceUID": str(getattr(dcm, "SOPInstanceUID", "")),
        "TransferSyntaxUID": str(transfer_syntax),
        "PhotometricInterpretation": str(getattr(dcm, "PhotometricInterpretation", "")),
        "Rows": int(getattr(dcm, "Rows", 0)),
        "Columns": int(getattr(dcm, "Columns", 0)),
        "InstanceNumber": int(getattr(dcm, "InstanceNumber", 0)) if getattr(dcm, "InstanceNumber", None) is not None else 0,
        "SliceThickness": float(getattr(dcm, "SliceThickness", 0.0)) if getattr(dcm, "SliceThickness", None) is not None else 0.0,
    }


def load_and_sort_series_files(series_dir: Union[str, Path]) -> List[Path]:
    """
    Finds all .dcm files in a series directory and sorts them anatomically.
    """
    series_path = Path(series_dir)
    dcm_files = sorted(list(series_path.glob("*.dcm")))
    if not dcm_files:
        return []

    # Read minimal header to determine anatomical order
    indexed = []
    for f in dcm_files:
        try:
            dcm = pydicom.dcmread(str(f), stop_before_pixels=True)
            order_val = get_slice_order_key(dcm)
            indexed.append((order_val, f))
        except Exception:
            indexed.append((0.0, f))

    indexed.sort(key=lambda x: x[0])
    return [item[1] for item in indexed]


# Alias matching imports.py specification
load_series_slices = load_and_sort_series_files

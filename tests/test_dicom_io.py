"""
Smoke tests and unit tests for DICOM I/O and Phase 0 setup.
Verifies core imports, transfer syntax decoding capabilities,
intensity normalization, and synthetic/real DICOM reading.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dicom_io import (
    extract_pixel_array,
    get_dicom_metadata,
    get_slice_order_key,
    normalize_intensity,
    read_dicom_slice,
    resize_image,
)


def test_core_imports():
    """Verify that all core deep learning and medical imaging libraries import cleanly."""
    import pydicom
    import torch
    import timm
    import pandas
    import pylibjpeg

    assert pydicom.__version__ is not None
    assert torch.__version__ is not None
    assert timm.__version__ is not None
    assert pandas.__version__ is not None
    assert pylibjpeg.__version__ is not None


def create_synthetic_dicom(
    file_path: Path,
    transfer_syntax_uid: str = "1.2.840.10008.1.2.1",
    photometric: str = "MONOCHROME2",
    rows: int = 64,
    cols: int = 64,
    instance_number: int = 1,
    z_position: float = 10.5,
):
    """
    Creates a valid synthetic DICOM file on disk for smoke-testing
    without needing 500GB+ competition data locally.
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = f"1.2.840.10008.1.2.3.{instance_number}"
    file_meta.TransferSyntaxUID = transfer_syntax_uid

    ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.StudyInstanceUID = "1.2.840.10008.1.100"
    ds.SeriesInstanceUID = "1.2.840.10008.1.200"
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "MR"
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.InstanceNumber = instance_number
    ds.ImagePositionPatient = ["0.0", "0.0", str(z_position)]
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0

    # Generate synthetic image data (gradient)
    arr = (np.linspace(0, 1000, rows * cols, dtype=np.uint16)).reshape((rows, cols))
    ds.PixelData = arr.tobytes()

    ds.save_as(str(file_path), enforce_file_format=True)
    return file_path


def test_dicom_decoding_explicit_vr():
    """Test reading and decoding uncompressed Explicit VR Little Endian DICOM."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dcm_path = Path(tmp_dir) / "explicit_vr.dcm"
        create_synthetic_dicom(dcm_path, transfer_syntax_uid="1.2.840.10008.1.2.1")

        img = read_dicom_slice(dcm_path, target_size=(224, 224), to_rgb=True)
        assert img.shape == (3, 224, 224)
        assert img.dtype == np.float32
        assert img.min() >= 0.0 and img.max() <= 1.0


def test_dicom_decoding_implicit_vr():
    """Test reading and decoding uncompressed Implicit VR Little Endian DICOM."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dcm_path = Path(tmp_dir) / "implicit_vr.dcm"
        create_synthetic_dicom(dcm_path, transfer_syntax_uid="1.2.840.10008.1.2")

        img = read_dicom_slice(dcm_path, target_size=(128, 128), to_rgb=False)
        assert img.shape == (128, 128)
        assert img.dtype == np.float32
        assert img.min() >= 0.0 and img.max() <= 1.0


def test_metadata_extraction_and_ordering():
    """Test metadata extraction and anatomical Z-coordinate extraction."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dcm_path = Path(tmp_dir) / "meta_test.dcm"
        create_synthetic_dicom(
            dcm_path,
            instance_number=5,
            z_position=-42.5,
            photometric="MONOCHROME1",
        )

        meta = get_dicom_metadata(dcm_path)
        assert meta["StudyInstanceUID"] == "1.2.840.10008.1.100"
        assert meta["SeriesInstanceUID"] == "1.2.840.10008.1.200"
        assert meta["PhotometricInterpretation"] == "MONOCHROME1"
        assert meta["InstanceNumber"] == 5

        # Check anatomical ordering key
        import pydicom
        dcm = pydicom.dcmread(str(dcm_path))
        key = get_slice_order_key(dcm)
        assert key == -42.5


def test_intensity_normalization():
    """Verify that extreme outliers are clipped and scaled to [0, 1]."""
    test_arr = np.array([0.0, 10.0, 50.0, 90.0, 10000.0], dtype=np.float32)
    norm = normalize_intensity(test_arr, percentile_low=1.0, percentile_high=99.0)
    assert norm.min() == 0.0
    assert norm.max() == 1.0


def test_transfer_syntax_plugins_available():
    """Verify that pydicom decoders for JPEG Lossless and JPEG 2000 are present and operational."""
    import pydicom
    from pydicom.uid import JPEGLossless, JPEG2000Lossless, ExplicitVRLittleEndian

    try:
        from pydicom.pixels.decoders.base import get_decoder
        jpeg_lossless_dec = get_decoder(JPEGLossless)
        jpeg2k_dec = get_decoder(JPEG2000Lossless)
        assert jpeg_lossless_dec is not None
        assert jpeg2k_dec is not None
    except ImportError:
        # pydicom 2.x fallback check
        from pydicom.pixel_data_handlers import pylibjpeg_handler
        assert pylibjpeg_handler.is_available()


if __name__ == "__main__":
    test_core_imports()
    test_dicom_decoding_explicit_vr()
    test_dicom_decoding_implicit_vr()
    test_metadata_extraction_and_ordering()
    test_intensity_normalization()
    test_transfer_syntax_plugins_available()
    print("All Phase 0 tests passed successfully!")

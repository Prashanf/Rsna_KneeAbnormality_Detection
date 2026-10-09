"""
Phase 6: Test inference and submission generation.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd


def run_inference(
    checkpoint_path: Union[str, Path],
    test_csv: Union[str, Path] = "data/raw/test.csv",
    test_series_csv: Union[str, Path] = "data/raw/test_series.csv",
    output_path: Union[str, Path] = "submission.csv",
) -> pd.DataFrame:
    """
    Phase 6 entrypoint: runs inference on test studies and generates submission.csv.
    """
    raise NotImplementedError("run_inference pending Phase 6 implementation.")

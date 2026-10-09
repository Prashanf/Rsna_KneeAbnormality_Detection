"""
Single-file consolidated imports for the Knee MRI Multilabel Classification project.
Imported wholesale into Kaggle notebooks via:
    from environment.imports import *
"""

# Standard library
import os
import sys
import json
import time
import math
import logging
from pathlib import Path
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Third-party numerical and scientific packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pydicom
import timm
from tqdm import tqdm

# Repo modules (local to this project)
from src.data_audit import DataAuditor, run_integrity_audit, TARGET_LABELS
from src.label_llm_relabel import relabel_all_studies
from src.dicom_io import (
    extract_pixel_array,
    get_dicom_metadata,
    get_slice_order_key,
    load_and_sort_series_files,
    load_series_slices,
    normalize_intensity,
    read_dicom_slice,
    resize_image,
)
from src.encoders import get_encoder
from src.embed_slices import embed_series, embed_all_series
from src.dataset import StudyDataset, collate_study_batch
from src.model.series_encoder import SeriesEncoder, SeriesMetadataEmbedding, GatedAttentionPooling
from src.model.fusion import FusionEncoder
from src.model.heads import FindingHeads, KneeMRIClassifier, compute_multilabel_bce_loss
from src.evaluate import calculate_multilabel_metrics, evaluate_model, print_evaluation_summary
from src.train import (
    compute_class_pos_weights,
    create_stratified_split,
    load_checkpoint,
    save_checkpoint,
    train_loop,
    train_one_epoch,
)
from src.predict import run_inference

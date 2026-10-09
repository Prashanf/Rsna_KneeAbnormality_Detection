# Implementation Plan — Knee MRI Multilabel Classification

Companion to `PROBLEM_STATEMENT.md`. Written for a coding agent to execute
step by step. Each phase lists: goal, inputs, outputs, concrete steps, and a
"done when" check. Do not skip the "done when" checks — later phases assume
earlier ones actually passed, not just ran.

---

## Repo layout to create

```
project/
├── PROBLEM_STATEMENT.md
├── PLAN.md
├── configs/
│   └── config.yaml                 # paths, hyperparams, encoder choice, etc.
├── data/
│   ├── raw/                        # symlink or path to train_series/, train.csv, etc.
│   └── processed/
│       ├── integrity_report.json
│       ├── train_labels_llm.csv
│       ├── label_audit.csv         # per-study diff: original vs LLM-derived
│       ├── series_index.parquet    # one row per series with resolved metadata
│       └── embeddings/
│           └── <encoder_name>/<StudyInstanceUID>/<SeriesInstanceUID>.npy
├── src/
│   ├── data_audit.py                # Phase 1
│   ├── label_llm_relabel.py         # Phase 2
│   ├── dicom_io.py                  # shared DICOM loading/decoding utils
│   ├── encoders/
│   │   ├── __init__.py              # registry: resnet50, dinov2, etc.
│   │   ├── resnet.py
│   │   └── dino.py
│   ├── embed_slices.py              # Phase 3
│   ├── dataset.py                   # PyTorch Dataset/DataLoader for study->series->slices
│   ├── model/
│   │   ├── series_encoder.py        # per-series transformer + attention pooling
│   │   ├── fusion.py                # cross-series fusion transformer
│   │   └── heads.py                 # 12 classification heads
│   ├── train.py                     # Phase 5
│   ├── evaluate.py
│   └── predict.py                   # Phase 6, writes submission.csv
├── notebooks/                       # optional exploration
└── tests/
    ├── test_data_audit.py
    ├── test_dicom_io.py
    └── test_model_shapes.py
```

---

## Phase 0 — Setup

1. Create the repo layout above.
2. `configs/config.yaml` should hold, at minimum: raw data paths, label CSV paths,
   LLM provider/model name for relabeling, image encoder choice, embedding cache
   dir, transformer hyperparams (d_model, n_heads, n_layers for both the series
   encoder and fusion encoder), batch size, split seed, output dir.
3. Pin dependencies: `pydicom`, `python-gdcm` or `pylibjpeg` (+ `pylibjpeg-libjpeg`,
   `pylibjpeg-openjpeg`) for JPEG Lossless / JPEG 2000 decoding, `torch`,
   `torchvision`, `timm` (for DINOv2/ResNet backbones), `pandas`, `pyarrow`,
   `scikit-learn`, `tqdm`, an LLM SDK (Anthropic SDK) for Phase 2.
4. **Done when:** `python -c "import pydicom, torch, timm, pandas"` runs clean, and
   one sample `.dcm` from each transfer syntax present in the data can be opened and
   pixel-array-decoded without error (write a tiny smoke-test script for this in
   `tests/`).

---

## Phase 1 — Data structure / integrity audit (`src/data_audit.py`)

**Goal:** confirm the data on disk actually matches the CSVs before trusting
anything downstream.

Checks to implement:
1. Every `StudyInstanceUID` in `train.csv` has a corresponding folder under
   `train_series/`.
2. Every `StudyInstanceUID` in `train_series.csv` matches a study folder, and every
   `SeriesInstanceUID` in `train_series.csv` matches a subfolder under that study.
3. Every `.dcm` file on disk under `train_series/<study>/<series>/` is reachable from
   `train_series.csv` (i.e., no orphan series directories not listed in the CSV), and
   vice versa (no series listed in CSV missing on disk).
4. For every series folder, count `.dcm` files and compare to any expected range
   (flag series with 0 files, or implausibly large counts — the "long tail to a few
   hundred" case — as review candidates, not hard failures).
5. Attempt to open and decode (pixel_array) a sample (or all) slices per series;
   record failures by transfer syntax (`pydicom`'s `file_meta.TransferSyntaxUID`).
6. Cross-check `train.csv` study IDs vs `train_series.csv` study IDs: studies with
   labels but no series, or series but no label row.
7. Compute and save distributions: series-per-study count, slices-per-series count,
   `Anatomical_Plane` value counts, `Fluid_Sensitive` / `Fat_Suppression` value
   counts, label prevalence per class in `train.csv`.

**Output:** `data/processed/integrity_report.json` with pass/fail per check plus
the distribution summaries, and a printed human-readable summary.

**Done when:** report is generated, zero *hard* integrity failures remain
unresolved (missing folders / unreadable series must be fixed or explicitly
excluded with a documented reason), and distribution stats are sane (e.g., no
study with 0 series ends up silently in later training).

---

## Phase 2 — LLM-based label re-derivation (`src/label_llm_relabel.py`)

**Goal:** since only a subset of `train.csv` labels are trustworthy, use an LLM to
derive all 12 binary labels directly from `Report` text for all ~4,500 studies, and
reconcile against the original columns.

Steps:
1. Build a strict extraction prompt listing the 12 findings with short clinical
   definitions (mirror the dataset description's wording) and require a
   structured JSON output: one 0/1 per label, plus a short `evidence_span` or
   `rationale` per positive label (for auditability), plus an `uncertain` flag if
   the report is ambiguous, non-English and not confidently translated, or doesn't
   mention a finding at all (absence of mention → treat as 0, but mark
   `explicitly_mentioned=false` so it can be distinguished from an explicit
   negative).
2. Handle multi-language reports: either let the LLM read/translate internally
   (preferred, single call) or run a translation pass first — keep it simple, one
   LLM call per report, instruct it to handle any language.
3. Batch over all 4,500 studies (this is small — run sequentially or with modest
   concurrency; no need for a complex pipeline). Cache raw LLM outputs per
   `StudyInstanceUID` to disk (e.g., `data/processed/llm_raw/<StudyInstanceUID>.json`)
   so re-runs are cheap and auditable.
4. Parse into `train_labels_llm.csv` with the same 12 columns + `StudyInstanceUID`.
5. Build `label_audit.csv`: per study, per label, compare original `train.csv` value
   vs LLM-derived value; flag disagreements. Compute agreement rate per label.
6. Decide and document a reconciliation policy, e.g.:
   - If original label is `1` and LLM says `0` (or vice versa) → flag for manual
     spot-check; default to **LLM-derived label as the training target**, since the
     dataset description says labels should be derived from the report.
   - Track an `is_flagged` column so training can optionally downweight or exclude
     flagged rows, or use them for a manual review sample.
7. Spot-check a random sample (~50–100 studies) of LLM outputs against the raw
   report text manually (or with a second LLM pass as a self-consistency check) to
   estimate labeling error rate before trusting it as ground truth for training.

**Output:** `data/processed/train_labels_llm.csv` (final training labels),
`data/processed/label_audit.csv` (diagnostics), per-study raw LLM JSON cache.

**Done when:** every study in `train.csv` has an LLM-derived label row, agreement
rate per label is reported, and a documented reconciliation policy decides which
label source feeds `train.py`.

---

## Phase 3 — Slice embedding (`src/embed_slices.py`, `src/encoders/`)

**Goal:** turn every slice into a fixed-size vector once, offline, so training
never touches raw DICOMs.

1. `src/dicom_io.py`: robust slice loader — read pixel array, handle transfer
   syntax via `pydicom` + `pylibjpeg`/`gdcm` plugins, apply `RescaleSlope`/
   `RescaleIntercept` if present, normalize intensity (e.g., percentile clip +
   min-max or z-score per slice or per series), resize to the encoder's expected
   input size, replicate to 3 channels if the encoder expects RGB.
2. `src/encoders/`: a small registry exposing at least:
   - `resnet50` (torchvision, ImageNet-pretrained) — fast baseline.
   - `dinov2` (via `timm` or `torch.hub`) — stronger general features.
   - Leave a slot/interface for a radiology-pretrained backbone (e.g.
     RadImageNet-style weights) to be swapped in later without changing the
     pipeline — document this as a known upgrade path, not a blocker for v1.
   Each encoder exposes `encode(batch_of_images) -> np.ndarray [B, D]`.
3. Iterate every series, batch its slices through the chosen encoder, save one
   `.npy` (or a single consolidated `.parquet`/HDF5/`.npz` per series) of shape
   `[num_slices, D]` to `data/processed/embeddings/<encoder_name>/<study>/<series>.npy`.
   Keep slice order (sorted by `InstanceNumber` or `ImagePositionPatient` z-coord if
   available — prefer true anatomical order over filename order).
4. Make this resumable/idempotent (skip series already embedded) since it will run
   over ~820k files once and must survive interruption.
5. Record, per series, the embedding encoder name/version and D (dim), so the
   training config can validate it's reading embeddings from the expected encoder.

**Done when:** every series that passed Phase 1's integrity check has a
corresponding embedding file, with slice order preserved and documented, and a
spot-check decode-and-embed of a few dozen series succeeds without NaNs.

---

## Phase 4 — Architecture (`src/dataset.py`, `src/model/`)

Implements the design already agreed on:

### 4.1 Series metadata conditioning
For each series, build a small metadata embedding from `Fluid_Sensitive`,
`Fat_Suppression`, `Anatomical_Plane` (categorical → learned embedding or one-hot +
linear). This is added to (or concatenates with, then projects) each slice token
before the per-series transformer, so the model knows what kind of series it's
looking at — this is what lets the model handle **variable series composition**
instead of assuming a fixed slot per sequence type.

### 4.2 Per-series transformer encoder (`src/model/series_encoder.py`)
- Input: `[num_slices, D]` slice embeddings (D = encoder output dim) for one series,
  padded to a max length within the batch, plus an attention mask.
- Project D → `d_model` (small: 64–128).
- Add positional encoding reflecting true slice order (normalized index, or
  z-position if available) + the series-type metadata embedding (broadcast to all
  tokens, or prepended as an extra token).
- 1–3 transformer encoder layers, small number of heads (2–4), dropout 0.1–0.3.
- **Attention pooling** (gated attention-MIL, Ilse et al. style) over the masked
  token outputs → one series-level summary vector. Keep the per-slice attention
  weights accessible for interpretability (e.g., return them alongside the pooled
  vector).
- This encoder is **shared** across all series within a study and across the
  dataset (single set of weights), since series share the same underlying slice
  embedding space and data is limited — do not instantiate one encoder per series
  "slot."

### 4.3 Cross-series fusion transformer (`src/model/fusion.py`)
- Input: the study's series-level vectors (variable count — 2 to however many
  series a study has), padded/masked to the batch max.
- No fixed "5 series" assumption: this must work with however many series a given
  study actually has (per Phase 1's distribution analysis).
- Optionally add a learned "missing/present" or series-type embedding here too if
  useful beyond what's already encoded per series.
- 1–3 transformer layers over the series tokens, then pool (attention pooling again,
  or a `[CLS]` token) to a single study-level representation vector.

### 4.4 Twelve classification heads (`src/model/heads.py`)
- Shared trunk (the study representation) → 12 independent linear heads, each
  outputting a single logit (not one shared softmax — labels are not mutually
  exclusive).
- Loss: `BCEWithLogitsLoss` with a per-class `pos_weight` computed from the final
  training label distribution (from Phase 2's output) to address imbalance.
- `src/dataset.py`: PyTorch `Dataset` that, given a `StudyInstanceUID`, loads all its
  series' cached embeddings (Phase 3 output), builds the padded/masked tensors
  described above, and returns `(series_embeddings, series_masks, series_meta,
  slice_masks_per_series, label_vector)`.

**Done when:** a forward pass runs end-to-end on a batch of synthetic (random)
data with the real shapes (variable series count, variable slice count per series)
without shape errors, and `tests/test_model_shapes.py` covers at least: one series
missing, one series with only 1 slice, one series near the max slice-count tail.

---

## Phase 5 — Training (`src/train.py`)

1. **Split:** study-level (= patient-level here, since each row is one study;
   confirm there's no multi-study-per-patient linkage in the metadata — if there
   is, split by patient ID instead) stratified split, ideally multilabel-stratified
   (e.g., iterative stratification) given 12 imbalanced labels, not a plain random
   split. 80/10/10 or similar train/val/test (or train/val if a held-out test set
   isn't needed yet since the real test set is separate).
2. Use `train_labels_llm.csv` (Phase 2 output) as the training target, per the
   reconciliation policy decided there.
3. Optimizer: AdamW, moderate LR (e.g., 1e-4 to 3e-4) with warmup + cosine or linear
   decay, weight decay, gradient clipping. Given the model is small and data is
   small, watch for overfitting: dropout, weight decay, early stopping on val loss
   or val mean-AUROC, and consider freezing the image encoder (Phase 3 embeddings
   are precomputed/frozen already, which naturally limits overfitting surface to the
   two small transformers + heads).
4. Track per-label metrics (AUROC, AP) on validation each epoch, not just aggregate
   loss, since class imbalance means aggregate loss can hide a collapsed rare class.
5. Log flagged/uncertain-label studies (from Phase 2) separately — consider an
   ablation: train with vs without flagged studies, to quantify their impact.
6. Save best checkpoint by a chosen validation metric (document the choice — e.g.
   mean per-label AUROC), plus the config used to produce it.

**Done when:** a full training run completes on the real data, produces per-label
validation metrics, and a checkpoint + config are saved together reproducibly.

---

## Phase 6 — Inference / submission (`src/predict.py`)

1. For each test study: load its series list from `test_series.csv`, embed its
   slices with the **same** encoder/version used in Phase 3 (or embed on the fly if
   a separate test embedding cache step is cleaner — mirror Phase 3's pipeline),
   run the trained model, sigmoid the 12 logits.
2. Write `submission.csv` matching `sample_submission.csv`'s exact column order and
   `StudyInstanceUID` set.
3. Sanity-check: no NaNs, all probabilities in [0, 1], every required test study ID
   present exactly once.

**Done when:** `submission.csv` validates against `sample_submission.csv`'s schema
and a dry run on the 3 example test studies in `test.csv`/`test_series/` succeeds.

---

## Open items to confirm before/while building

- Exact competition scoring metric (affects whether to optimize threshold-free
  AUROC vs. calibrated log-loss) — check the actual competition rules.
- Whether any studies share a patient across multiple rows (affects split
  strategy).
- Final choice of image encoder (start with ResNet50 or DINOv2 as a fast baseline;
  evaluate a radiology-pretrained backbone as a follow-up, not a v1 blocker).
- LLM reconciliation policy thresholds (how disagreement/uncertainty translates
  into label weighting or exclusion) — finalize after reviewing Phase 2's agreement
  rate.

---

## Suggested execution order for the coding agent

1. Phase 0 (setup) → 2. Phase 1 (audit) → 3. Phase 2 (LLM relabeling, can run in
parallel with Phase 3 once Phase 1 passes) → 4. Phase 3 (embeddings, the long-running
step — kick off early) → 5. Phase 4 (architecture, can be built/unit-tested with
synthetic data while Phase 3 embeddings are still computing) → 6. Phase 5 (training)
→ 7. Phase 6 (inference/submission).

# Problem Statement — Knee MRI Multilabel Finding Classification

## 1. Task

Given a knee MRI **study** (a collection of DICOM **series**, each series being one MRI
acquisition of 20–100+ slices), predict the per-study probability of **12 binary
findings**:

1. `ACL` — anterior cruciate ligament injury
2. `MCL` — medial collateral ligament injury
3. `Medial Meniscus` — tear
4. `Lateral Meniscus` — tear
5. `Medial OA` — osteoarthritis, medial tibiofemoral compartment
6. `Lateral OA` — osteoarthritis, lateral tibiofemoral compartment
7. `PF OA` — patellofemoral osteoarthritis
8. `Effusion` — joint effusion
9. `Synovitis`
10. `Baker's` — Baker's cyst
11. `Contusion` — bone contusion/bruise
12. `Fracture`

This is a **study-level, multilabel classification** problem — a study can have 0, 1,
or several findings simultaneously. Output: 12 independent probabilities per study.

## 2. Data

- ~4,500 training studies, ~1,300 test studies (569.76 GB, ~820k DICOM files total).
- Each study = several series (folders of `.dcm` slice files). Series typically have
  20–45 slices (median 30), long tail to a few hundred.
- `train.csv`: `StudyInstanceUID`, free-text `Report` (multi-language), 12 binary
  label columns.
- `train_series.csv`: one row per series — `StudyInstanceUID`, `SeriesInstanceUID`,
  `Fluid_Sensitive` (0/1), `Fat_Suppression` (0/1), `Anatomical_Plane`
  (Sagittal/Coronal/Axial).
- `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm` — one
  file per slice.
- `test.csv` / `test_series.csv` / `test_series/` — same schema, `Report` withheld at
  test time.
- `sample_submission.csv` — all labels at 0.5, defines the exact submission schema.
- DICOMs vary in intensity scale, orientation, resolution, transfer syntax
  (uncompressed Explicit VR LE, JPEG Lossless, JPEG 2000, Implicit VR LE), scanner
  vendor, protocol, and institution. Metadata stripped to an 86-tag allowlist.
- **Series composition is not fixed** per study — number and type of series varies.
  Series must be characterized by their *attributes* (plane, fluid-sensitivity,
  fat-sat), not by a fixed positional slot (no assumption of "always exactly 5
  series" / "series #3 is always X").

## 3. Critical Data Quality Issue: Label Noise

Per the dataset description: *"Only a small subset of training studies carry
per-condition labels. We also provide the original text of the radiology report from
which you may wish to derive the labels for the remaining studies."*

This means:
- The 12 binary columns in `train.csv` are **not uniformly reliable** — some rows'
  labels may be missing, blank-as-zero, or simply not derived from the report at all.
- The `Report` free-text field is the **more authoritative source of truth** for most
  studies.
- Training directly on `train.csv` labels as given risks learning from systematically
  wrong targets, which would undermine every downstream modeling step.
- **Mitigation:** since there are only ~4,500 studies, use an LLM to read each
  `Report` and (re-)derive all 12 binary labels directly from the text, in a
  structured, auditable way. Reconcile LLM-derived labels against the original
  `train.csv` columns, flag disagreements, and produce a cleaned
  `train_labels_llm.csv` that downstream training uses as the source of truth (with
  the ability to fall back to / blend with original labels where confidence is low).

## 4. Evaluation

Per-study, per-finding probability predictions across 12 labels. Matches
`sample_submission.csv` format (`StudyInstanceUID` + 12 probability columns).
Likely scored via mean per-label AUROC or log-loss across findings (confirm against
official competition rules before finalizing the training loss/metric choice — note
this in the plan as an open item).

## 5. Key Challenges

1. **Label noise / incompleteness** in `train.csv` — requires LLM-based re-labeling
   from `Report` text before any model training.
2. **Variable, inconsistent series composition** — no fixed series count or order;
   must condition on series metadata, tolerate missing/extra series.
3. **Variable slice counts** (20 to a few hundred) within a series — requires padding
   + masking or attention pooling.
4. **Domain gap** — natural-image-pretrained encoders (ResNet/DINO) vs. MRI grayscale
   intensity distributions.
5. **Multi-site heterogeneity** — scanners, protocols, populations, multilingual
   reports.
6. **Class imbalance** across 12 findings; prevalence may differ between train/public
   leaderboard/final eval splits (per the dataset notice).
7. **Scale** — 569 GB / ~820k DICOM files; requires an offline
   preprocessing/embedding pipeline (cache slice embeddings to disk) rather than
   decoding raw DICOMs during each training step.
8. **Mixed DICOM transfer syntaxes** — decoder must handle uncompressed, JPEG
   Lossless, and JPEG 2000 streams correctly.

## 6. Proposed Approach (high level)

1. **Data integrity audit** — verify every study/series referenced in the CSVs
   exists on disk (and vice versa), check slice counts, detect corrupt/unreadable
   DICOMs, summarize series-count and slice-count distributions.
2. **LLM-based label re-derivation** — extract/validate the 12 binary labels from
   `Report` text for all training studies; produce cleaned, auditable labels.
3. **Slice embedding** — encode every slice with a frozen (or fine-tuned) image
   encoder (ResNet50, DINOv2, or a radiology-pretrained backbone), cache vectors to
   disk, keyed by study/series/slice.
4. **Per-series transformer encoder** — a shared, small transformer with attention
   pooling consumes a series' slice-embedding sequence (masked for variable length),
   conditioned on series metadata (plane, fluid-sensitive, fat-sat) via a learned
   embedding, producing one series-level summary vector.
5. **Cross-series fusion transformer** — a second small transformer attends over a
   study's (variable-count, padded/masked) series vectors to produce one study-level
   representation.
6. **12 classification heads** — independent sigmoid heads (shared trunk + 12 linear
   heads) trained with `BCEWithLogitsLoss` + per-class `pos_weight` for imbalance.
7. **Patient/study-level stratified split**, careful validation, submission generation
   matching `sample_submission.csv`.

See `PLAN.md` for the concrete, ordered implementation plan and file/module layout.

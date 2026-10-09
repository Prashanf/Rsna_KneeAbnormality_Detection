# Kaggle Workflow — Repo-Style Instructions

Companion to `PROBLEM_STATEMENT.md` and `PLAN.md`. This file governs **how** every
phase of `PLAN.md` gets built and run. All real compute happens on Kaggle; all code
authorship happens locally in a git repo that Kaggle notebooks clone and call into.
Read this before writing or editing any script or notebook.

## 0. The core constraint and the chosen model

We have no local compute for anything beyond writing code. All real execution (data
audit, LLM relabeling, embedding ~820k DICOM slices, training) happens in **Kaggle
notebooks**. Two things make this risky if handled naively:

1. Re-uploading a brand-new notebook = a fresh kernel = losing every already-executed
   cell's in-memory state. Phase 3 alone can take many hours; re-running it from
   scratch is the single biggest risk to this project's timeline.
2. Editing code directly inside Kaggle's notebook cells scatters the real logic
   across notebook versions, making it hard to track, diff, or reuse.

**Chosen model:** all real logic lives as **plain `.py` scripts/modules in this git
repo**. Kaggle notebooks are kept intentionally thin — a handful of cells whose only
job is: clone/update the repo, install/import dependencies, then call functions from
the repo. Fixing a bug means editing the script locally, pushing, and re-pulling it
into the already-running Kaggle session — the notebook itself barely changes, and the
Kaggle session's already-computed state (variables, loaded data, GPU allocation) is
never disturbed just to fix a script.

```
local repo (source of truth, edited here, with Claude)
      │  git push
      ▼
   GitHub repo
      │  git clone / git pull (inside a Kaggle cell)
      ▼
Kaggle notebook (thin: clone → import → call repo functions → persist outputs)
```

## 1. Repo layout

Same layout as `PLAN.md` §"Repo layout to create", with one addition:

```
project/
├── PROBLEM_STATEMENT.md
├── PLAN.md
├── INSTRUCTIONS.md
├── environment/
│   └── imports.py              # <-- single file, every import any script needs
├── configs/
│   └── config.yaml
├── src/
│   ├── data_audit.py
│   ├── label_llm_relabel.py
│   ├── dicom_io.py
│   ├── encoders/...
│   ├── embed_slices.py
│   ├── dataset.py
│   ├── model/...
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_label_llm_relabel.ipynb
│   ├── 03_embed_slices.ipynb
│   ├── 04_model_dev.ipynb
│   ├── 05_train.ipynb
│   └── 06_predict_submit.ipynb
└── tests/...
```

### `environment/imports.py`

One file, plain Python (not a notebook), containing **every** import any script in
`src/` needs — both third-party packages and the repo's own modules — written as
real `import` statements, e.g.:

```python
# environment/imports.py
# Third-party
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import pydicom
import timm

# Repo modules (local to this project)
from src.data_audit import run_integrity_audit
from src.label_llm_relabel import relabel_all_studies
from src.dicom_io import load_series_slices
from src.embed_slices import embed_series
from src.dataset import StudyDataset
from src.model.series_encoder import SeriesEncoder
from src.model.fusion import FusionEncoder
from src.model.heads import FindingHeads
from src.train import train_loop
from src.predict import run_inference
```

This is the file you update whenever a script gains a new dependency or a new
function worth calling directly from a notebook. It's the thing Kaggle notebooks
load wholesale into one cell (see §3).

## 2. The golden rule: edit the repo, not the Kaggle notebook

When something needs a fix (a bug, a changed hyperparameter, a new check):

1. **Default action: edit the `.py` file locally in the repo**, not the Kaggle
   notebook cell. Commit and push.
2. In the running Kaggle session, re-run only the **repo-sync cell** (§3) — this
   pulls the new code and makes it live in the kernel. Do **not** restart the kernel
   and do **not** upload a new notebook.
3. Any cell downstream of the sync cell that depends on the changed function should
   be re-run too (e.g., if you fixed `embed_series`, re-run the embedding loop cell;
   you do not need to re-run the data-audit cell above it).
4. Long-running, expensive cells (slice embedding loop, LLM relabeling loop, training
   loop) must stay **checkpointed and resumable** (§5) specifically so that a code fix
   never forces recomputation of already-finished work — re-running the loop cell
   after a fix should just skip everything already done and continue.

## 3. The repo-sync cell (goes near the top of every notebook)

Standard pattern, same in every notebook:

```python
# --- Cell 1: sync repo ---
import os, subprocess

REPO_URL = "https://github.com/<you>/<repo>.git"   # use a Kaggle Secret for the
                                                      # token if the repo is private
REPO_DIR = "/kaggle/working/repo"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

commit = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
print(f"Repo synced at commit {commit}")

import sys
sys.path.insert(0, REPO_DIR)
```

```python
# --- Cell 2: load all imports + purge stale module cache, every re-run ---
for mod in list(sys.modules):
    if mod.startswith("src.") or mod.startswith("environment."):
        del sys.modules[mod]

from environment.imports import *   # pulls in every third-party + repo import
```

**Why the purge matters:** `git pull` updates files on disk, but Python caches
already-imported modules in `sys.modules`. Without the purge, re-running Cell 2 after
a `git pull` would silently keep using the *old* function bodies. The purge forces a
genuinely fresh import of every `src.*`/`environment.*` module, every time Cell 2 is
re-run — so "pull latest code, re-run Cell 2, continue" is always safe and always
actually picks up the fix.

**Always print the commit hash** (as above) right after syncing, so every run's
output/checkpoint is traceable to an exact script version.

## 4. Private repo / credentials

If the repo is private, do **not** hardcode a token in the notebook (it gets saved in
Kaggle's version history and outputs, visible to anyone the notebook is shared with).
Use Kaggle's **Add-ons → Secrets** to store a GitHub PAT, then in the sync cell:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_PAT")
REPO_URL = f"https://{token}@github.com/<you>/<repo>.git"
```

## 5. Resumability still applies — the repo-sync solves code staleness, not compute loss

Repo-style working fixes "losing code changes." It does **not** by itself fix
"losing hours of compute" if a Kaggle session dies. Keep everything from the earlier
resumability design:

- **Persist to disk incrementally**, not only at the end — one output file per
  study/series/checkpoint as it's produced, not accumulated in memory.
- **Every long loop skips already-done work on start** — check whether an output
  file exists before (re)computing it.
- **Push `/kaggle/working/` outputs to a Kaggle Dataset periodically** so they survive
  a session actually closing (not just a cell re-run within one session). Each
  notebook's early cells should check for and load from that dataset if present,
  instead of recomputing from scratch.
- **Checkpoint training** (model + optimizer + epoch) every N steps, and resume from
  the latest checkpoint rather than reinitializing.
- **Log progress at the top of every long notebook** (e.g. "3120/4500 studies
  embedded, resuming") so it's obvious at a glance whether a re-run does real work or
  fast-skips to where it left off.

## 6. Exception: the final submission notebook may need no internet

Kaggle competition scoring often runs with **internet access disabled**. `git clone`
requires internet, so `06_predict_submit.ipynb`, specifically the version used for
actual submission, needs a fallback:

- Package the repo's current state as a **Kaggle Dataset** (zip the repo, upload as a
  dataset) or publish `src/` as a **Kaggle Utility Script**, and attach it to the
  submission notebook instead of cloning.
- Keep this as a separate, clearly-labeled step done only when preparing an actual
  submission — day-to-day development notebooks (01–05) keep using the git-clone
  pattern since they run with internet on.

## 7. Practical Kaggle mechanics

- Keep the **dataset attach** (competition data + any of our own output datasets from
  §5) configured in the notebook's sidebar, not assumed via hardcoded paths that only
  exist in one session.
- GPU only on the notebooks that need it (Phase 3 embedding, Phase 5 training); keep
  Phase 1/2 on CPU to conserve GPU quota.
- Respect Kaggle's session time limits (commonly ~9–12h GPU) —§5's resumability is
  what lets Phase 3 safely span multiple sessions.
- When sharing a notebook's state back here for review, paste the specific
  cell(s)/output in question, not the whole notebook.

## 8. Summary checklist before touching anything

- [ ] Did the fix go into the repo (`src/*.py` or `environment/imports.py`) locally,
      not into a Kaggle notebook cell?
- [ ] Was it committed and pushed?
- [ ] In Kaggle: did I re-run the sync cell (clone/pull) and the import-purge cell,
      rather than restarting the kernel or re-uploading the notebook?
- [ ] Does the printed commit hash match what I just pushed?
- [ ] If this is a long/expensive loop, does it skip already-completed
      studies/series/epochs on re-run?
- [ ] Is output persisted to disk incrementally, and pushed to a Kaggle Dataset if it
      must survive a session close?
- [ ] For the submission notebook only: am I using a Dataset/Utility Script attach
      instead of `git clone`, since scoring may run offline?

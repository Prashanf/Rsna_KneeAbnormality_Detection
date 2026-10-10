"""
Unit tests for Phase 2: LLM-based label re-derivation (src/label_llm_relabel.py).
Tests extraction parsing, negation handling, incremental disk caching,
and label reconciliation diagnostics.
"""

import json
import sys
import tempfile
from pathlib import Path
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit import TARGET_LABELS
from src.label_llm_relabel import (
    LLMExtractor,
    ReportRelabeler,
    clean_json_response,
    relabel_all_studies,
)


def test_clean_json_response():
    """Verify JSON parser strips markdown syntax correctly."""
    markdown_wrapped = """Here is the result:
```json
{
  "detected_language": "en",
  "findings": {"ACL": {"present": 1, "explicitly_mentioned": true, "evidence": "tear", "uncertain": false}},
  "summary": "Torn ACL"
}
```
Hope this helps!"""
    parsed = clean_json_response(markdown_wrapped)
    assert parsed["detected_language"] == "en"
    assert parsed["findings"]["ACL"]["present"] == 1

    raw_json = '{"detected_language": "es", "summary": "normal"}'
    parsed2 = clean_json_response(raw_json)
    assert parsed2["detected_language"] == "es"


def test_heuristic_extractor_negation_and_findings():
    """Verify that heuristic extractor detects tear and handles negation correctly."""
    extractor = LLMExtractor(provider="mock")

    # Positive ACL tear and Baker cyst, negative Meniscus
    report_text = (
        "Complete tear of the anterior cruciate ligament with surrounding joint effusion. "
        "Popliteal Baker's cyst is noted in the posterior compartment. "
        "The medial and lateral menisci are intact with no tear. "
        "No fracture or bone contusion identified."
    )

    result = extractor.extract_from_report(report_text)
    findings = result["findings"]

    assert findings["ACL"]["present"] == 1
    assert "tear" in findings["ACL"]["evidence"].lower()
    assert findings["Baker's"]["present"] == 1
    assert findings["Effusion"]["present"] == 1

    # Negated findings should be 0
    assert findings["Medial Meniscus"]["present"] == 0
    assert findings["Lateral Meniscus"]["present"] == 0
    assert findings["Fracture"]["present"] == 0
    assert findings["Contusion"]["present"] == 0


def test_relabeler_incremental_caching_and_audit():
    """Verify that ReportRelabeler caches per-study JSONs and compiles audit tables."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        raw_dir = base_dir / "raw"
        processed_dir = base_dir / "processed"
        raw_dir.mkdir(parents=True)
        processed_dir.mkdir(parents=True)

        # Create mock train.csv with 2 studies
        train_data = [
            {
                "StudyInstanceUID": "study_001",
                "Report": "Tear of the medial meniscus. Joint effusion present. No ACL injury.",
                "ACL": 0,
                "Medial Meniscus": 0,  # Intentional disagreement (orig=0, report has tear -> llm=1)
                "Effusion": 1,
            },
            {
                "StudyInstanceUID": "study_002",
                "Report": "Unremarkable examination. All ligaments intact. No effusion.",
                "ACL": 0,
                "Medial Meniscus": 0,
                "Effusion": 0,
            },
        ]
        # Fill remaining labels
        for row in train_data:
            for l in TARGET_LABELS:
                if l not in row:
                    row[l] = 0

        pd.DataFrame(train_data).to_csv(raw_dir / "train.csv", index=False)

        extractor = LLMExtractor(provider="mock")
        relabeler = ReportRelabeler(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            extractor=extractor,
        )

        labels_df, audit_df = relabeler.process_all_reports()

        # 1. Check cache files exist on disk
        cache_dir = processed_dir / "llm_raw"
        assert (cache_dir / "study_001.json").exists()
        assert (cache_dir / "study_002.json").exists()

        # 2. Check output CSV files exist
        assert (processed_dir / "train_labels_llm.csv").exists()
        assert (processed_dir / "label_audit.csv").exists()

        # 3. Check labels content
        assert len(labels_df) == 2
        s1_labels = labels_df[labels_df["StudyInstanceUID"] == "study_001"].iloc[0]
        assert s1_labels["Medial Meniscus"] == 1
        assert s1_labels["ACL"] == 0

        # 4. Check audit diagnostics
        assert len(audit_df) == 2
        s1_audit = audit_df[audit_df["StudyInstanceUID"] == "study_001"].iloc[0]
        assert bool(s1_audit["is_flagged"]) is True
        assert bool(s1_audit["Medial Meniscus_disagree"]) is True

        s2_audit = audit_df[audit_df["StudyInstanceUID"] == "study_002"].iloc[0]
        assert bool(s2_audit["is_flagged"]) is False

        # 5. Test idempotency / skipping
        # Modify cached file for study_001 to ensure re-run reads cache and does not re-extract
        with open(cache_dir / "study_001.json", "r+") as f:
            d = json.load(f)
            d["findings"]["ACL"]["present"] = 1  # change in cache
            f.seek(0)
            json.dump(d, f)
            f.truncate()

        labels_df2, _ = relabeler.process_all_reports()
        s1_new = labels_df2[labels_df2["StudyInstanceUID"] == "study_001"].iloc[0]
        assert s1_new["ACL"] == 1  # Read directly from cache without recomputing


def test_gemini_extractor_init_and_mock(monkeypatch):
    """Verify Gemini provider initialization, fallback on missing key, and mock extraction."""
    # 1. Missing key -> graceful fallback to local clinical extractor
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    ext_fallback = LLMExtractor(provider="gemini", api_key=None)
    assert ext_fallback.client is None

    # Report extraction still functions via fallback
    res_fallback = ext_fallback.extract_from_report("Tear of the ACL and medial meniscus.")
    assert res_fallback["findings"]["ACL"]["present"] == 1
    assert res_fallback["findings"]["Medial Meniscus"]["present"] == 1

    # 2. Provided key -> client initialized with gemini-3.8-flash
    ext_gemini = LLMExtractor(provider="gemini", model="gemini-3.8-flash", api_key="test_dummy_key")
    assert ext_gemini.client is not None
    assert ext_gemini.model == "gemini-3.8-flash"

    # Test auto-upgrade of legacy model
    ext_legacy = LLMExtractor(provider="gemini", model="gemini-2.5-flash", api_key="test_dummy_key")
    assert ext_legacy.model == "gemini-3.8-flash"

    # 3. Mock _call_gemini returning structured JSON response
    mock_json_response = json.dumps({
        "detected_language": "en",
        "findings": {
            "ACL": {"present": 1, "explicitly_mentioned": True, "evidence": "full-thickness tear of the ACL", "uncertain": False},
            "MCL": {"present": 0, "explicitly_mentioned": True, "evidence": "", "uncertain": False},
            "Medial Meniscus": {"present": 1, "explicitly_mentioned": True, "evidence": "bucket-handle tear", "uncertain": False},
            "Lateral Meniscus": {"present": 0, "explicitly_mentioned": False, "evidence": "", "uncertain": False},
            "Medial OA": {"present": 1, "explicitly_mentioned": True, "evidence": "medial compartment joint space loss", "uncertain": False},
            "Lateral OA": {"present": 0, "explicitly_mentioned": False, "evidence": "", "uncertain": False},
            "PF OA": {"present": 0, "explicitly_mentioned": False, "evidence": "", "uncertain": False},
            "Effusion": {"present": 1, "explicitly_mentioned": True, "evidence": "large joint effusion", "uncertain": False},
            "Synovitis": {"present": 0, "explicitly_mentioned": False, "evidence": "", "uncertain": False},
            "Baker's": {"present": 1, "explicitly_mentioned": True, "evidence": "Baker cyst in popliteal fossa", "uncertain": False},
            "Contusion": {"present": 0, "explicitly_mentioned": False, "evidence": "", "uncertain": False},
            "Fracture": {"present": 0, "explicitly_mentioned": False, "evidence": "", "uncertain": False},
        },
        "summary": "Torn ACL and medial meniscus with large effusion and Baker cyst."
    })

    monkeypatch.setattr(ext_gemini, "_call_gemini", lambda report_text: mock_json_response)
    extracted = ext_gemini.extract_from_report("Some complex radiology report text.")

    assert extracted["detected_language"] == "en"
    assert extracted["findings"]["ACL"]["present"] == 1
    assert extracted["findings"]["Medial Meniscus"]["present"] == 1
    assert extracted["findings"]["Medial OA"]["present"] == 1
    assert extracted["findings"]["Effusion"]["present"] == 1
    assert extracted["findings"]["Baker's"]["present"] == 1
    assert extracted["findings"]["Lateral Meniscus"]["present"] == 0
    assert extracted["findings"]["Fracture"]["present"] == 0


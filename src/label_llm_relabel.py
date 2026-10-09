"""
Phase 2: LLM-based label re-derivation from free-text radiology reports.
Extracts 12 binary findings per study, caches responses incrementally,
audits agreement against original train.csv labels, and produces clean training targets.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit import TARGET_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an expert board-certified musculoskeletal radiologist.
Your task is to analyze a knee MRI radiology report and determine the presence of 12 specific knee findings.
The report text may be in English, Spanish, Portuguese, French, German, or another language. Translate internally and interpret accurately.

The 12 findings to evaluate are:
1. "ACL": Anterior Cruciate Ligament tear (complete or partial) or rupture. (Intact, normal, or non-visualized without tear is 0).
2. "MCL": Medial Collateral Ligament tear, rupture, sprain, or high-grade injury.
3. "Medial Meniscus": Tear of the medial meniscus (radial, horizontal, bucket handle, complex, root tear, or maceration).
4. "Lateral Meniscus": Tear of the lateral meniscus.
5. "Medial OA": Osteoarthritis, chondropathy, cartilage wear/loss, or osteophytes in the MEDIAL tibiofemoral compartment.
6. "Lateral OA": Osteoarthritis, chondropathy, cartilage wear/loss, or osteophytes in the LATERAL tibiofemoral compartment.
7. "PF OA": Patellofemoral osteoarthritis, chondromalacia patellae (grades 2-4 / cartilage defect), or patellofemoral osteophytes.
8. "Effusion": Joint effusion, fluid in suprapatellar recess or joint capsule. (Trace/physiologic fluid is 0).
9. "Synovitis": Synovial thickening, inflammation, synovial hypertrophy, or synovitis.
10. "Baker's": Baker's cyst, popliteal cyst, or gastrocnemius-semimembranosus bursal distension.
11. "Contusion": Bone contusion, bone marrow edema, or bone bruise secondary to trauma.
12. "Fracture": Fracture, cortical disruption, trabecular impaction fracture (patella, femur, tibia, or fibula).

Guidelines:
- Negation: Explicit negatives (e.g. "no ACL tear", "menisci intact", "absence of effusion") must be 0.
- Absence of mention: If a finding is not mentioned at all, set present=0 and explicitly_mentioned=false.
- Uncertain: If findings are equivocal, ambiguous, or borderline, set uncertain=true.
- Provide a brief exact quote (evidence_span) for each finding where present=1.

Return ONLY a valid JSON object matching this exact schema:
{
  "detected_language": "<e.g. en, es, fr, de>",
  "findings": {
    "ACL": {"present": 0, "explicitly_mentioned": false, "evidence": "", "uncertain": false},
    "MCL": {"present": 0, "explicitly_mentioned": false, "evidence": "", "uncertain": false},
    "Medial Meniscus": {"present": 0, "explicitly_mentioned": false, "evidence": "", "uncertain": false},
    "Lateral Meniscus": {"present": 0, "explicitly_mentioned": false, "evidence": "", "uncertain": false},
    "Medial OA": {"present": 0, "explicitly_mentioned": false, "evidence": "", "uncertain": false},
    "Lateral OA": {"present": 0, "explicitly_mentioned": false, "evidence": "", "uncertain": false},
    "PF OA": {"present": 0, "explicitly_mentioned": false, "evidence": "", "uncertain": false},
    "Effusion": {"present": 0, "explicitly_mentioned": false, "evidence": "", "uncertain": false},
    "Synovitis": {"present": 0, "explicitly_mentioned": false, "evidence": "", "uncertain": false},
    "Baker's": {"present": 0, "explicitly_mentioned": false, "evidence": "", "uncertain": false},
    "Contusion": {"present": 0, "explicitly_mentioned": false, "evidence": "", "uncertain": false},
    "Fracture": {"present": 0, "explicitly_mentioned": false, "evidence": "", "uncertain": false}
  },
  "summary": "<1-2 sentence overall radiologic impression>"
}
"""


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Extracts and parses JSON object from LLM response text."""
    text = raw_text.strip()
    # Match markdown code block ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # Match outermost curly braces
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            text = match.group(1)

    return json.loads(text)


class LLMExtractor:
    """Wrapper for LLM extraction with multiple provider support and offline mock fallback."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.temperature = temperature
        self.max_retries = max_retries

        if self.provider == "anthropic" and self.api_key:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None

    def extract_from_report(self, report_text: str) -> Dict[str, Any]:
        """Extracts structured 12-label findings from a single report string."""
        if not report_text or not isinstance(report_text, str) or report_text.strip() == "":
            return self._empty_response()

        if self.provider == "mock" or self.client is None:
            return self._heuristic_mock_extract(report_text)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                if self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=1000,
                        temperature=self.temperature,
                        system=EXTRACTION_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": f"Radiology Report:\n\n{report_text}"}],
                    )
                    raw_content = response.content[0].text
                    parsed = clean_json_response(raw_content)
                    return self._validate_and_normalize(parsed)
            except Exception as e:
                last_error = e
                logger.warning(f"LLM API attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2**attempt)

        logger.error(f"All {self.max_retries} attempts failed. Falling back to heuristic extractor: {last_error}")
        return self._heuristic_mock_extract(report_text)

    def _empty_response(self) -> Dict[str, Any]:
        return {
            "detected_language": "unknown",
            "findings": {k: {"present": 0, "explicitly_mentioned": False, "evidence": "", "uncertain": False} for k in TARGET_LABELS},
            "summary": "Empty report",
        }

    def _heuristic_mock_extract(self, text: str) -> Dict[str, Any]:
        """Rule-based extractor used for testing, offline runs, or fallback."""
        lower = text.lower()
        findings = {}

        patterns = {
            "ACL": [r"\bacl\b.*tear", r"anterior cruciate ligament.*tear", r"rupture of the acl"],
            "MCL": [r"\bmcl\b.*(?:tear|sprain)", r"medial collateral ligament.*(?:tear|sprain)"],
            "Medial Meniscus": [r"medial meniscus.*tear", r"tear of the medial meniscus"],
            "Lateral Meniscus": [r"lateral meniscus.*tear", r"tear of the lateral meniscus"],
            "Medial OA": [r"medial.*(?:osteoarthritis|cartilage loss|chondropathy|joint space narrowing)"],
            "Lateral OA": [r"lateral.*(?:osteoarthritis|cartilage loss|chondropathy)"],
            "PF OA": [r"patellofemoral.*(?:osteoarthritis|chondromalacia|cartilage defect)"],
            "Effusion": [r"(?:joint|moderate|large|suprapatellar)\s+effusion"],
            "Synovitis": [r"synovitis", r"synovial thickening", r"synovial hypertrophy"],
            "Baker's": [r"baker'?s?\s+cyst", r"popliteal cyst"],
            "Contusion": [r"bone\s+(?:contusion|bruise)", r"trabecular microfracture", r"bone marrow edema"],
            "Fracture": [r"\bfracture\b", r"cortical disruption"],
        }

        negations = ["no ", "without ", "intact", "unremarkable", "ruled out", "negative for"]

        for label in TARGET_LABELS:
            present = 0
            evidence = ""
            matched_patterns = patterns.get(label, [])
            for p in matched_patterns:
                m = re.search(p, lower)
                if m:
                    # Check negative prefix
                    start = max(0, m.start() - 25)
                    prefix = lower[start : m.start()]
                    is_neg = any(neg in prefix for neg in negations)
                    if not is_neg:
                        present = 1
                        evidence = m.group(0)
                        break

            findings[label] = {
                "present": present,
                "explicitly_mentioned": label.lower() in lower or bool(evidence),
                "evidence": evidence,
                "uncertain": False,
            }

        return {
            "detected_language": "en",
            "findings": findings,
            "summary": "Heuristic extraction",
        }

    def _validate_and_normalize(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures all 12 target labels exist with integer 0 or 1 present."""
        findings = parsed.get("findings", {})
        norm_findings = {}
        for label in TARGET_LABELS:
            item = findings.get(label, {})
            pres = int(item.get("present", 0))
            if pres not in (0, 1):
                pres = 1 if pres > 0 else 0
            norm_findings[label] = {
                "present": pres,
                "explicitly_mentioned": bool(item.get("explicitly_mentioned", False)),
                "evidence": str(item.get("evidence", "")),
                "uncertain": bool(item.get("uncertain", False)),
            }
        parsed["findings"] = norm_findings
        return parsed


class ReportRelabeler:
    """Manages batch processing, caching, reconciliation, and audit reporting."""

    def __init__(
        self,
        raw_dir: Union[str, Path] = "data/raw",
        processed_dir: Union[str, Path] = "data/processed",
        train_csv: str = "train.csv",
        extractor: Optional[LLMExtractor] = None,
        cache_subdir: str = "llm_raw",
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.train_csv_path = self.raw_dir / train_csv
        self.cache_dir = self.processed_dir / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = extractor or LLMExtractor()

    def process_all_reports(self, limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes reports, caching each study incrementally to disk.
        Skips already cached studies on start.
        Returns (train_labels_llm_df, label_audit_df).
        """
        train_df = pd.read_csv(self.train_csv_path)
        if limit is not None:
            train_df = train_df.iloc[:limit].copy()

        logger.info(f"Loaded {len(train_df)} studies for LLM relabeling from {self.train_csv_path}")

        # Check existing cached studies
        cached_files = {p.stem for p in self.cache_dir.glob("*.json")}
        logger.info(f"Existing cache found: {len(cached_files)} studies already completed.")

        # Incremental processing loop
        pbar = tqdm(train_df.iterrows(), total=len(train_df), desc="LLM relabeling")
        for _, row in pbar:
            study_uid = str(row["StudyInstanceUID"])
            cache_file = self.cache_dir / f"{study_uid}.json"

            if cache_file.exists():
                continue

            report_text = str(row.get("Report", ""))
            extracted = self.extractor.extract_from_report(report_text)

            # Persist immediately to disk
            with open(cache_file, "w") as f:
                json.dump(extracted, f, indent=2)

        # Aggregate results
        return self.aggregate_and_audit(train_df)

    def aggregate_and_audit(self, original_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compiles cached JSONs into train_labels_llm.csv and label_audit.csv."""
        if original_df is None:
            original_df = pd.read_csv(self.train_csv_path)

        records = []
        audit_records = []

        for _, row in original_df.iterrows():
            study_uid = str(row["StudyInstanceUID"])
            cache_file = self.cache_dir / f"{study_uid}.json"

            if not cache_file.exists():
                logger.warning(f"Missing cache file for study {study_uid}")
                continue

            with open(cache_file) as f:
                cached_data = json.load(f)

            findings = cached_data.get("findings", {})
            study_labels = {"StudyInstanceUID": study_uid}
            audit_entry = {
                "StudyInstanceUID": study_uid,
                "detected_language": cached_data.get("detected_language", "unknown"),
                "total_disagreements": 0,
            }

            has_disagreement = False
            for label in TARGET_LABELS:
                llm_val = int(findings.get(label, {}).get("present", 0))
                uncertain = bool(findings.get(label, {}).get("uncertain", False))
                evidence = str(findings.get(label, {}).get("evidence", ""))

                study_labels[label] = llm_val
                study_labels[f"{label}_uncertain"] = uncertain

                # Original label comparison
                orig_val = row.get(label, np.nan)
                if pd.notna(orig_val):
                    orig_val = int(orig_val)
                    disagree = (orig_val != llm_val)
                    if disagree:
                        has_disagreement = True
                        audit_entry["total_disagreements"] += 1
                else:
                    disagree = False

                audit_entry[f"{label}_orig"] = orig_val
                audit_entry[f"{label}_llm"] = llm_val
                audit_entry[f"{label}_disagree"] = disagree
                audit_entry[f"{label}_evidence"] = evidence

            audit_entry["is_flagged"] = has_disagreement
            records.append(study_labels)
            audit_records.append(audit_entry)

        train_labels_llm = pd.DataFrame(records)
        label_audit = pd.DataFrame(audit_records)

        # Save to disk
        out_labels = self.processed_dir / "train_labels_llm.csv"
        out_audit = self.processed_dir / "label_audit.csv"

        train_labels_llm.to_csv(out_labels, index=False)
        label_audit.to_csv(out_audit, index=False)

        logger.info(f"Saved {len(train_labels_llm)} LLM-derived label rows to: {out_labels}")
        logger.info(f"Saved audit comparisons to: {out_audit}")

        self._print_audit_summary(label_audit)
        return train_labels_llm, label_audit

    def _print_audit_summary(self, audit_df: pd.DataFrame):
        """Calculates and displays agreement rates and summary statistics."""
        print("\n" + "=" * 60)
        print("          PHASE 2 LLM LABEL RE-DERIVATION AUDIT")
        print("=" * 60)
        total_studies = len(audit_df)
        flagged_studies = audit_df["is_flagged"].sum() if "is_flagged" in audit_df.columns else 0
        print(f"Total Studies Audited: {total_studies}")
        print(f"Studies with Label Disagreements: {flagged_studies} ({flagged_studies/max(total_studies,1)*100:.1f}%)")
        print("\nPer-Label Agreement Rates:")

        for label in TARGET_LABELS:
            orig_col = f"{label}_orig"
            llm_col = f"{label}_llm"
            if orig_col in audit_df.columns and llm_col in audit_df.columns:
                valid_mask = audit_df[orig_col].notna()
                if valid_mask.sum() > 0:
                    matches = (audit_df.loc[valid_mask, orig_col] == audit_df.loc[valid_mask, llm_col]).sum()
                    rate = matches / valid_mask.sum()
                    print(f"  {label:<18}: {rate*100:.2f}% agreement ({matches}/{valid_mask.sum()})")
        print("=" * 60 + "\n")


def relabel_all_studies(
    raw_dir: Union[str, Path] = "data/raw",
    processed_dir: Union[str, Path] = "data/processed",
    train_csv: str = "train.csv",
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Top-level helper function for Phase 2 LLM re-labeling.
    Used by notebooks and Kaggle workflows.
    """
    extractor = LLMExtractor(provider=provider, model=model, api_key=api_key)
    relabeler = ReportRelabeler(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        train_csv=train_csv,
        extractor=extractor,
    )
    return relabeler.process_all_reports(limit=limit)


def main():
    parser = argparse.ArgumentParser(description="Run LLM-based Label Re-derivation")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed output directory")
    parser.add_argument("--train-csv", type=str, default="train.csv", help="Filename of training CSV")
    parser.add_argument("--provider", type=str, default="anthropic", help="LLM provider: anthropic, gemini, or mock")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022", help="Model name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of studies for test run")
    args = parser.parse_args()

    relabel_all_studies(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        train_csv=args.train_csv,
        provider=args.provider,
        model=args.model,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

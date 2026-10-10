"""
Phase 2: Multilingual clinical label re-derivation from free-text radiology reports.
Extracts 12 binary findings per study, caches responses incrementally,
audits agreement against ground-truth train.csv labels, and produces clean training targets.
Outputs clean_train.csv in the workspace root and data/processed/train_labels_llm.csv.
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
from src.multilingual_labeler import MultilingualKneeLabeler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are a world-class, board-certified Musculoskeletal (MSK) Radiologist specializing in Knee MRI interpretation and multi-label pathology extraction.

Your task is to analyze a knee MRI radiology report and extract the presence (0 or 1) of 12 distinct knee findings.

The report text may be written in English, Turkish, Spanish, Greek, Croatian/Serbian, Cyrillic (Bulgarian/Russian), Dutch, French, German, or a mixture of languages. 
Translate internally and interpret native clinical and anatomical terminology accurately.

---
### THE 12 TARGET FINDINGS & CLINICAL DEFINITIONS:

1. "ACL" (Anterior Cruciate Ligament Injury):
   - Positive (1): Complete or partial tear, full disruption, high-grade sprain, avulsion, mucoid degeneration with tear, or ruptured ACL reconstruction graft. Also mark 1 if the ACL is non-visualized/absent in the context of trauma.
   - Negative (0): Intact ligament, normal morphology and signal intensity, unremarkable, or intact postoperative graft. Mucoid degeneration without tear is 0.

2. "MCL" (Medial Collateral Ligament Injury):
   - Positive (1): Sprain (Grade I, II, or III), partial tear, complete rupture, avulsion, or periligamentous fluid/edema with fiber discontinuity.
   - Negative (0): Intact ligament, normal thickness and signal, unremarkable.

3. "Medial Meniscus" (Tear):
   - Positive (1): True tear of the medial meniscus (radial, horizontal cleavage, longitudinal, oblique, bucket-handle, complex, flap, root tear, or maceration/substance loss with tear). Grade 3 intrameniscal signal contacting the superior or inferior articular surface.
   - Negative (0): Intact, normal, or isolated Grade 1/2 degenerative intrameniscal signal that does NOT communicate with the articular surface. Mild surface fraying without true tear = 0.

4. "Lateral Meniscus" (Tear):
   - Positive (1): True tear of the lateral meniscus (radial, horizontal, longitudinal, complex, bucket-handle, anterior or posterior horn tear, root tear). Grade 3 intrameniscal signal reaching the articular margin.
   - Negative (0): Intact, normal, or Grade 1/2 intrameniscal signal without articular surface extension.

5. "Medial OA" (Medial Compartment Osteoarthritis):
   - Positive (1): Osteoarthritis, chondropathy, chondromalacia, partial or full-thickness cartilage defect/wear, subchondral sclerosis/geodes, or marginal osteophytes localized to the MEDIAL tibiofemoral compartment (medial femoral condyle or medial tibial plateau). Also positive if generalized "tricompartmental osteoarthritis" is diagnosed.
   - Negative (0): Intact medial compartment cartilage, normal joint space, or OA strictly confined to the lateral or patellofemoral compartments.

6. "Lateral OA" (Lateral Compartment Osteoarthritis):
   - Positive (1): Osteoarthritis, chondropathy, chondromalacia, partial or full-thickness cartilage defect/wear, subchondral sclerosis, or marginal osteophytes localized to the LATERAL tibiofemoral compartment (lateral femoral condyle or lateral tibial plateau). Also positive if generalized "tricompartmental osteoarthritis" is diagnosed.
   - Negative (0): Intact lateral compartment cartilage, normal joint space, or OA strictly confined to the medial or patellofemoral compartments.

7. "PF OA" (Patellofemoral Osteoarthritis):
   - Positive (1): Patellofemoral osteoarthritis, chondromalacia patellae (grades 2 to 4 / cartilage fissure, partial/full-thickness defect of patellar facets or femoral trochlea), patellofemoral joint space narrowing, or patellar/trochlear osteophytes. Also positive if "tricompartmental osteoarthritis" is diagnosed.
   - Negative (0): Normal patellofemoral cartilage, or grade 1 cartilage softening without defect.

8. "Effusion" (Joint Effusion):
   - Positive (1): Pathological joint effusion, moderate or large intra-articular fluid collection, prominent fluid distension of suprapatellar recess, lateral gutters, or capsular distension.
   - Negative (0): Physiological, trace, minimal, or physiological volume of joint fluid. Explicit absence of effusion ("no effusion", "sin derrame", "effüzyon saptanmadı", "kein Erguss") = 0.

9. "Synovitis":
   - Positive (1): Synovial thickening, hypertrophy, inflammatory proliferation, synovial plica syndrome with inflammatory reaction, or synovial pannus.
   - Negative (0): Normal synovium, absence of synovial thickening, or simple non-inflammatory clear effusion.

10. "Baker's" (Baker's / Popliteal Cyst):
    - Positive (1): Baker's cyst, popliteal cyst, or fluid distension of the gastrocnemius-semimembranosus bursa (intact or ruptured).
    - Negative (0): No popliteal cyst mentioned, normal popliteal fossa.

11. "Contusion" (Bone Contusion / Bone Bruise):
    - Positive (1): Bone contusion, bone bruise, trabecular microfracture, or bone marrow edema (BME) secondary to acute/subacute trauma or impaction (femoral condyles, tibial plateaus, patella, fibula).
    - Negative (0): Normal bone marrow signal, or isolated subchondral degenerative edema/geode directly caused by chronic osteoarthritis without acute traumatic contusion.

12. "Fracture":
    - Positive (1): Cortical fracture, macroscopic fracture line, cortical disruption/step-off, avulsion fracture (Segond fracture, tibial spine avulsion), patellar fracture, tibial plateau fracture, or impaction fracture.
    - Negative (0): Intact bony cortex, no fracture. (Isolated bone contusion without cortical fracture line = 0).

---
### CRITICAL REASONING & EXTRACTION GUIDELINES:

1. Negation Handling:
   - Rigorously detect medical negations across all languages (e.g. English: "no tear", "intact", "without"; Spanish: "sin", "no se aprecia", "descartado"; Turkish: "izlenmedi", "saptanmadı", "doğal", "yırtık yok"; Greek: "χωρίς", "φυσιολογικό"; Croatian: "nema", "uredan", "bez"; French: "absence de", "intact"; German: "kein", "regelrecht").
   - An explicitly negated finding MUST be present=0.

2. Absence of Mention:
   - If a finding or structure is not described at all in the report, set present=0 and explicitly_mentioned=false.

3. Compartment Specificity:
   - Do NOT conflate medial vs. lateral pathology. A medial meniscus tear does NOT make Lateral Meniscus positive.
   - Medial OA and Lateral OA require pathology in their specific compartment, UNLESS "tricompartmental OA" is explicitly noted (which marks Medial OA=1, Lateral OA=1, and PF OA=1).

4. Evidence Spans:
   - For every finding where present=1, extract the exact relevant phrase from the original report text as the "evidence" field.

---
### OUTPUT JSON FORMAT:
Return ONLY a valid, parseable JSON object matching this schema:
{
  "detected_language": "<e.g. en, tr, es, el, hr, bg, nl, fr, de>",
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
  "summary": "<Concise 1-2 sentence overall radiological impression>"
}
"""


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Extracts and parses JSON object from LLM response text."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            text = match.group(1)

    try:
        return json.loads(text)
    except Exception:
        # Fallback repair for partially truncated valid JSON
        for suffix in ["}", "}}", '"}', '"}}}', '": 0}}']:
            try:
                return json.loads(text + suffix)
            except Exception:
                pass
        raise


class LLMExtractor:
    """Wrapper for clinical extraction supporting Gemini, Anthropic, local multilingual NLP, or mock."""

    def __init__(
        self,
        provider: str = "gemini",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 4,
        request_delay: float = 2.0,
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.clinical_labeler = MultilingualKneeLabeler()
        self.client = None

        if self.provider in ("gemini", "google"):
            raw_model = model or "gemini-3.8-flash"
            # Auto-upgrade deprecated/legacy Gemini models (e.g. gemini-2.5, gemini-2.0, gemini-1.5)
            deprecated_prefixes = ("gemini-2.5", "gemini-2.0", "gemini-1.5", "gemini-1.0")
            if any(raw_model.startswith(p) for p in deprecated_prefixes):
                logger.warning(
                    f"Model '{raw_model}' is deprecated by Google Gemini API. "
                    f"Automatically upgrading to 'gemini-3.8-flash'."
                )
                self.model = "gemini-3.8-flash"
            else:
                self.model = raw_model

            self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not self.api_key:
                logger.warning(
                    "No GEMINI_API_KEY or GOOGLE_API_KEY set. "
                    "Set environment variable or pass --api-key. Falling back to local clinical extractor."
                )
            else:
                try:
                    from google import genai
                    self.client = genai.Client(api_key=self.api_key)
                    logger.info(f"Initialized Gemini client with model '{self.model}'.")
                except Exception as e:
                    logger.warning(f"Could not initialize Gemini client ({e}). Falling back to clinical extractor.")
        elif self.provider in ("qwen", "local_llm", "huggingface"):
            self.model = model or "Qwen/Qwen2.5-7B-Instruct"
            self.api_key = None
            self.request_delay = 0.0  # Local inference needs no network rate pacing
            self.tokenizer = None
            self.llm_model = None
            self._init_local_llm()
        elif self.provider == "anthropic":
            self.model = model or "claude-3-5-sonnet-20241022"
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                logger.warning(
                    "No ANTHROPIC_API_KEY set. "
                    "Set environment variable or pass --api-key. Falling back to local clinical extractor."
                )
            else:
                try:
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=self.api_key)
                    logger.info(f"Initialized Anthropic client with model '{self.model}'.")
                except Exception as e:
                    logger.warning(f"Could not initialize Anthropic client ({e}). Falling back to clinical extractor.")
        else:
            self.model = model or "clinical-multilingual"
            self.api_key = None

    def _init_local_llm(self):
        """Initializes local open-source LLM (e.g. Qwen2.5-7B) on GPU."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading local open-source model '{self.model}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                trust_remote_code=True,
            )

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA GPU detected: {gpu_name}")
                # Models <= 3B params (e.g. Qwen2.5-1.5B/3B) fit in 16GB VRAM in pure float16,
                # which eliminates 4-bit dequantization overhead and runs 4-5x faster.
                is_small_model = any(sz in self.model.lower() for sz in ["0.5b", "1.5b", "3b", "tiny", "mini", "small"])
                if is_small_model:
                    logger.info(f"Model '{self.model}' is <= 3B params; loading in native float16 for maximum inference speed...")
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        self.model,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                else:
                    try:
                        from transformers import BitsAndBytesConfig
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                        self.llm_model = AutoModelForCausalLM.from_pretrained(
                            self.model,
                            quantization_config=bnb_config,
                            device_map="auto",
                            trust_remote_code=True,
                        )
                        logger.info("Loaded 7B model in 4-bit precision (efficient VRAM footprint ~5GB).")
                    except Exception as bnb_err:
                        logger.info(f"4-bit loading not available ({bnb_err}), loading in float16...")
                        self.llm_model = AutoModelForCausalLM.from_pretrained(
                            self.model,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            trust_remote_code=True,
                        )
            elif torch.backends.mps.is_available():
                logger.info("MPS (Apple Silicon) detected, loading in float16...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                logger.warning("No GPU found; loading on CPU (generation may be slow).")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                )

            self.client = self.llm_model
            logger.info(f"✓ Local model '{self.model}' successfully initialized.")
        except Exception as e:
            logger.warning(f"Could not initialize local model '{self.model}': {e}. Falling back to clinical extractor.")
            self.client = None

    def _call_local_llm(self, report_text: str) -> str:
        """Runs local generation on GPU using Qwen2.5 chat template."""
        import torch

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Radiology Report:\n\n{report_text}\n\nReturn ONLY the JSON object. Do not output any commentary or explanations.",
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        device = getattr(self.llm_model, "device", "cuda" if torch.cuda.is_available() else "cpu")
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        eos_ids = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end is not None and im_end != getattr(self.tokenizer, "unk_token_id", None):
                eos_ids.append(im_end)

        with torch.no_grad():
            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=250,  # Structured JSON findings only require ~160-200 tokens
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=eos_ids,
            )

        input_len = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][input_len:]
        response_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response_text

    def _call_gemini(self, report_text: str) -> str:
        """Invokes Gemini API via google-genai SDK with JSON output formatting."""
        prompt = f"Radiology Report:\n\n{report_text}"

        # 1. Check if interactions API is supported (google-genai v2+)
        if hasattr(self.client, "interactions") and callable(getattr(self.client.interactions, "create", None)):
            try:
                interaction = self.client.interactions.create(
                    model=self.model,
                    input=prompt,
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    response_format=[{"type": "text", "mime_type": "application/json"}],
                )
                if interaction and getattr(interaction, "output_text", None):
                    return interaction.output_text
            except Exception as err:
                logger.debug(f"Interactions API call unsuccessful ({err}), attempting models.generate_content fallback")

        # 2. Standard models.generate_content (google-genai 1.x and 2.x)
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            temperature=self.temperature,
            response_mime_type="application/json",
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text

    def extract_from_report(self, report_text: str) -> Dict[str, Any]:
        """Extracts structured 12-label findings from a single report string."""
        if not report_text or not isinstance(report_text, str) or report_text.strip() == "":
            return self._empty_response()

        if self.provider in ("multilingual", "clinical", "local", "mock") or self.client is None:
            return self._clinical_extract(report_text)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                if self.provider in ("gemini", "google"):
                    raw_content = self._call_gemini(report_text)
                    parsed = clean_json_response(raw_content)
                    return self._validate_and_normalize(parsed)
                elif self.provider in ("qwen", "local_llm", "huggingface"):
                    raw_content = self._call_local_llm(report_text)
                    parsed = clean_json_response(raw_content)
                    return self._validate_and_normalize(parsed)
                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=1500,
                        temperature=self.temperature,
                        system=EXTRACTION_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": f"Radiology Report:\n\n{report_text}"}],
                    )
                    raw_content = response.content[0].text
                    parsed = clean_json_response(raw_content)
                    return self._validate_and_normalize(parsed)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                    wait_time = 15 * (attempt + 1)
                    logger.warning(
                        f"Gemini rate limit (429 / Quota) reached on attempt {attempt + 1}/{self.max_retries}. "
                        f"Backing off for {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    wait_time = 2**attempt
                    logger.warning(f"LLM API attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        logger.error(f"All {self.max_retries} attempts failed. Falling back to clinical extractor: {last_error}")
        return self._clinical_extract(report_text)

    def _empty_response(self) -> Dict[str, Any]:
        return {
            "detected_language": "unknown",
            "findings": {k: {"present": 0, "explicitly_mentioned": False, "evidence": "", "uncertain": False} for k in TARGET_LABELS},
            "summary": "Empty report",
        }

    def _clinical_extract(self, text: str) -> Dict[str, Any]:
        """Clinical multilingual parser covering all 9 languages."""
        preds, evidence = self.clinical_labeler.parse_report_with_details(text)
        lang = self.clinical_labeler.detect_language(text)

        findings = {}
        for label in TARGET_LABELS:
            val = preds.get(label, 0)
            findings[label] = {
                "present": int(val),
                "explicitly_mentioned": True,
                "evidence": str(evidence.get(label, "")),
                "uncertain": False,
            }

        return {
            "detected_language": lang,
            "findings": findings,
            "summary": f"Clinical NLP extraction ({lang})",
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

        # Check raw_dir first, then PROJECT_ROOT, then train_csv
        raw_candidate = self.raw_dir / train_csv
        root_candidate = PROJECT_ROOT / train_csv
        if raw_candidate.is_file():
            self.train_csv_path = raw_candidate
        elif root_candidate.is_file():
            self.train_csv_path = root_candidate
        elif Path(train_csv).is_file():
            self.train_csv_path = Path(train_csv)
        else:
            self.train_csv_path = raw_candidate

        self.cache_dir = self.processed_dir / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = extractor or LLMExtractor(provider="multilingual")

    def process_all_reports(self, limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes reports, caching each study incrementally to disk.
        Skips already cached studies on start.
        Returns (train_labels_llm_df, label_audit_df).
        """
        train_df = pd.read_csv(self.train_csv_path)
        if limit is not None:
            train_df = train_df.iloc[:limit].copy()

        logger.info(f"Loaded {len(train_df)} studies for relabeling from {self.train_csv_path}")

        cached_files = {p.stem for p in self.cache_dir.glob("*.json")}
        logger.info(f"Existing cache found: {len(cached_files)} studies already completed.")

        # Incremental processing loop
        pbar = tqdm(train_df.iterrows(), total=len(train_df), desc="Clinical relabeling")
        for _, row in pbar:
            study_uid = str(row["StudyInstanceUID"])
            cache_file = self.cache_dir / f"{study_uid}.json"

            if cache_file.exists():
                continue

            report_text = str(row.get("Report", ""))
            extracted = self.extractor.extract_from_report(report_text)

            with open(cache_file, "w") as f:
                json.dump(extracted, f, indent=2)

            # Polite pacing to respect RPM limits when calling external LLMs
            if getattr(self.extractor, "request_delay", 0) > 0 and getattr(self.extractor, "client", None) is not None:
                time.sleep(self.extractor.request_delay)

        return self.aggregate_and_audit(train_df)

    def aggregate_and_audit(self, original_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compiles cached JSONs into:
        1. clean_train.csv (in project root) - all 4,407 rows with full report and non-null labels
        2. data/processed/train_labels_llm.csv - StudyInstanceUID and 12 binary labels
        3. data/processed/label_audit.csv - comparative audit against original labels
        """
        if original_df is None:
            original_df = pd.read_csv(self.train_csv_path)

        clean_rows = []
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
            clean_study_row = row.to_dict()

            audit_entry = {
                "StudyInstanceUID": study_uid,
                "detected_language": cached_data.get("detected_language", "unknown"),
                "total_disagreements": 0,
            }

            has_disagreement = False
            for label in TARGET_LABELS:
                nlp_val = int(findings.get(label, {}).get("present", 0))
                uncertain = bool(findings.get(label, {}).get("uncertain", False))
                evidence = str(findings.get(label, {}).get("evidence", ""))

                orig_val = row.get(label, np.nan)
                if pd.notna(orig_val):
                    clean_val = int(orig_val)
                    disagree = (clean_val != nlp_val)
                    if disagree:
                        has_disagreement = True
                        audit_entry["total_disagreements"] += 1
                else:
                    clean_val = nlp_val
                    disagree = False

                # In train_labels_llm.csv, store the extracted label
                study_labels[label] = nlp_val
                # In clean_train.csv, store the reconciled label
                clean_study_row[label] = clean_val

                audit_entry[f"{label}_orig"] = orig_val
                audit_entry[f"{label}_nlp"] = nlp_val
                audit_entry[f"{label}_final"] = clean_val
                audit_entry[f"{label}_disagree"] = disagree
                audit_entry[f"{label}_evidence"] = evidence

            audit_entry["is_flagged"] = has_disagreement
            records.append(study_labels)
            audit_records.append(audit_entry)
            clean_rows.append(clean_study_row)

        train_labels_llm = pd.DataFrame(records)
        label_audit = pd.DataFrame(audit_records)
        clean_train = pd.DataFrame(clean_rows)

        # Save to disk
        out_clean = PROJECT_ROOT / "clean_train.csv"
        out_labels = self.processed_dir / "train_labels_llm.csv"
        out_audit = self.processed_dir / "label_audit.csv"

        clean_train.to_csv(out_clean, index=False)
        train_labels_llm.to_csv(out_labels, index=False)
        label_audit.to_csv(out_audit, index=False)

        logger.info(f"Saved complete cleaned train set ({len(clean_train)} rows) to: {out_clean}")
        logger.info(f"Saved {len(train_labels_llm)} label rows to: {out_labels}")
        logger.info(f"Saved audit comparisons to: {out_audit}")

        self._print_audit_summary(label_audit, clean_train)
        return train_labels_llm, label_audit

    def _print_audit_summary(self, audit_df: pd.DataFrame, clean_df: pd.DataFrame):
        """Calculates and displays agreement rates and summary statistics."""
        print("\n" + "=" * 65)
        print("          PHASE 2 MULTILINGUAL LABEL RE-DERIVATION AUDIT")
        print("=" * 65)
        total_studies = len(clean_df)
        nan_count = clean_df[TARGET_LABELS].isna().sum().sum()
        print(f"Total Studies in clean_train.csv: {total_studies}")
        print(f"Missing (NaN) Labels: {nan_count} (must be 0)")

        # Ground truth agreement
        gt_mask = audit_df["ACL_orig"].notna()
        n_gt = gt_mask.sum()
        print(f"\nStudies with Original Ground Truth: {n_gt}")
        if n_gt > 0:
            print("Agreement with Ground Truth (Clinical NLP vs Ground Truth):")
            for label in TARGET_LABELS:
                orig_col = f"{label}_orig"
                nlp_col = f"{label}_nlp"
                matches = (audit_df.loc[gt_mask, orig_col] == audit_df.loc[gt_mask, nlp_col]).sum()
                rate = matches / n_gt
                print(f"  {label:<18}: {rate*100:6.2f}% agreement ({matches}/{n_gt})")

        print("\nFinal Label Prevalence Across Entire Clean Dataset (N=4,407):")
        for label in TARGET_LABELS:
            pos_cnt = int(clean_df[label].sum())
            prev = pos_cnt / total_studies * 100
            print(f"  {label:<18}: {pos_cnt:5d} positive ({prev:5.1f}%)")
        print("=" * 65 + "\n")


def relabel_all_studies(
    raw_dir: Union[str, Path] = "data/raw",
    processed_dir: Union[str, Path] = "data/processed",
    train_csv: str = "train.csv",
    provider: str = "gemini",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Top-level helper function for Phase 2 re-labeling.
    Used by scripts and Kaggle notebooks.
    Providers supported: 'qwen', 'local_llm', 'gemini', 'anthropic', 'multilingual', 'mock'.
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
    parser = argparse.ArgumentParser(description="Run Multilingual Label Re-derivation with Qwen, Gemini, or local NLP")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed output directory")
    parser.add_argument("--train-csv", type=str, default="train.csv", help="Filename of training CSV")
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["gemini", "google", "qwen", "local_llm", "anthropic", "multilingual", "mock"],
        help="Extractor provider: qwen (local GPU), gemini, anthropic, multilingual, or mock",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g. Qwen/Qwen2.5-7B-Instruct, gemini-3.8-flash, claude-3-5-sonnet-20241022)",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Gemini or Anthropic API key")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of studies for test/dry run")
    args = parser.parse_args()

    relabel_all_studies(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        train_csv=args.train_csv,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

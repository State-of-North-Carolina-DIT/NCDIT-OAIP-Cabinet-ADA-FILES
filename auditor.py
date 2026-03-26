#!/usr/bin/env python3
"""Standalone quality auditor for the 3-step pipeline.

Step 3 of: extract_structured_json.py → render_json.py → auditor.py

Self-contained script — no external src/ dependencies.
Reads extraction-test JSON (pages[] format), source PDF, and rendered HTML.
Runs 3 fidelity LLM calls (Content, Structural, Visual) via Gemini,
optional Final Decider (4th LLM call), hard vetoes, signal collection,
quality score computation, and routing decision.

Outputs audit-report.json.

Usage:
    # Single document
    python auditor.py output/newsletter.json --pdf data/newsletter/source.pdf --html output/newsletter.html

    # Batch: directory of JSON files
    python auditor.py output/ --pdf-dir data/ --html-dir output/

    # Skip fidelity LLM calls (programmatic-only)
    python auditor.py output/newsletter.json --pdf data/newsletter/source.pdf --skip-llm

    # Skip Final Decider only (still run 3 fidelity calls)
    python auditor.py output/newsletter.json --pdf data/newsletter/source.pdf --skip-decider

    # Skip axe-core accessibility scan:
    python auditor.py output/newsletter.json --pdf data/newsletter/source.pdf --skip-axe

    # Directory layout mode (subfolders with source.pdf + .json + .html)
    python auditor.py json_to_html_to_auditor/

    # API key mode (Developer API, no Vertex AI needed):
    python auditor.py output/doc.json --pdf data/source.pdf --api-keys KEY1,KEY2,KEY3

    # API key mode via environment variables:
    export GEMINI_API_KEYS=KEY1,KEY2,KEY3,KEY4    # comma-separated
    # OR individual vars:
    export GEMINI_API_KEY_1=KEY1
    export GEMINI_API_KEY_2=KEY2
    # ...
    python auditor.py output/doc.json --pdf data/source.pdf --api-mode
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sanitize import sanitize_error

try:
    from ada_analytics import PipelineAnalyticsCollector as _PipelineAnalyticsCollector
    _HAS_ANALYTICS = True
except ImportError:
    _HAS_ANALYTICS = False

# Load .env from pipeline root (one level up from src/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)


# ============================================================================
# axe-core Accessibility Scanner (JSDOM-based, no browser needed)
# ============================================================================

# Inline JavaScript for axe-core runner. Uses JSDOM to create a virtual DOM
# from HTML, runs axe-core WCAG 2.1 AA checks, outputs JSON to stdout.
# No headless browser required — JSDOM simulates the DOM in pure JS.
_AXE_RUNNER_JS = r"""
const { JSDOM } = require("jsdom");
const axe = require("axe-core");

async function main() {
    const html = require("fs").readFileSync(process.argv[2], "utf-8");
    const dom = new JSDOM(html, { pretendToBeVisual: true });
    globalThis.window = dom.window;
    globalThis.document = dom.window.document;
    globalThis.Node = dom.window.Node;
    globalThis.NodeList = dom.window.NodeList;

    const results = await axe.run(dom.window.document.documentElement, {
        runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] },
        resultTypes: ["violations"]
    });

    const output = {
        axe_version: axe.version,
        violations: results.violations.map(v => ({
            id: v.id,
            impact: v.impact,
            description: v.description,
            help_url: v.helpUrl,
            tags: v.tags.filter(t => t.startsWith("wcag")),
            nodes_count: v.nodes.length
        }))
    };

    console.log(JSON.stringify(output));
    dom.window.close();
}

main().catch(e => {
    console.error(JSON.stringify({ error: e.message }));
    process.exit(1);
});
"""

# Path to axe-runner npm directory (sibling of src/)
_AXE_RUNNER_DIR = Path(__file__).resolve().parent.parent / "axe-runner"


def _run_axe_core(html_path: Path, timeout: float = 30.0) -> dict | None:
    """Run axe-core WCAG 2.1 AA scan on an HTML file via Node.js + JSDOM.

    Returns dict with 'axe_version' and 'violations' list, or None on failure.
    This is informational only — results do NOT affect scoring or routing.

    Prerequisites:
        - Node.js installed and in PATH
        - npm install run in pipeline/axe-runner/ directory
    """
    if not html_path.exists():
        logger.warning("axe-core: HTML file not found: %s", html_path)
        return None

    if shutil.which("node") is None:
        logger.warning("axe-core: Node.js not found in PATH — skipping scan")
        return None

    node_modules = _AXE_RUNNER_DIR / "node_modules"
    if not node_modules.exists():
        logger.warning(
            "axe-core: node_modules not found at %s — run 'npm install' in pipeline/axe-runner/",
            _AXE_RUNNER_DIR,
        )
        return None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", dir=str(_AXE_RUNNER_DIR), delete=False
        ) as f:
            f.write(_AXE_RUNNER_JS)
            js_path = f.name

        try:
            result = subprocess.run(
                ["node", js_path, str(html_path.resolve())],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(_AXE_RUNNER_DIR),
            )
        finally:
            Path(js_path).unlink(missing_ok=True)

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            logger.error("axe-core failed (exit %d): %s", result.returncode, error_msg[:300])
            return None

        return json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        logger.error("axe-core timed out after %.0f seconds", timeout)
        return None
    except json.JSONDecodeError as e:
        logger.error("axe-core returned invalid JSON: %s", e)
        return None
    except Exception as e:
        logger.error("axe-core unexpected error: %s", e)
        return None


# ============================================================================
# Configuration — env vars override defaults, CLI args override both
# ============================================================================

PROJECT_ID = os.environ.get("PROJECT_ID", "camp-ai-nc")
TENANT_ID = os.environ.get("TENANT_ID", "")
REGION = os.environ.get("GEMINI_LOCATION", "global")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
FIDELITY_MODEL = os.environ.get("FIDELITY_MODEL", "gemini-3.1-pro-preview")
FIDELITY_MAX_OUTPUT_TOKENS = int(os.environ.get("FIDELITY_MAX_TOKENS", "65500"))
FIDELITY_TIMEOUT_SECONDS = 300
FIDELITY_MAX_RETRIES = 2
DECIDER_MAX_OUTPUT_TOKENS = 16384
DECIDER_TIMEOUT_SECONDS = 120
DECIDER_MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Gemini client pool with API key rotation
# ---------------------------------------------------------------------------
# Two modes:
#   1. Vertex AI (default): single client using GCP ADC credentials
#   2. API key rotation: multiple clients, one per key, round-robin
#
# API key mode is activated by:
#   --api-keys KEY1,KEY2,...  (CLI flag)
#   GEMINI_API_KEYS=KEY1,KEY2,...  (environment variable)
#
# Keys are rotated round-robin per call. On 429, the failing key is
# skipped for a cooldown period and the next key is tried immediately.
# ---------------------------------------------------------------------------

def _load_env_local() -> None:
    """Load .env.local from project root into os.environ (won't overwrite existing)."""
    # Walk up from this script's directory to find .env.local
    search = Path(__file__).resolve().parent
    for _ in range(5):
        env_file = search / ".env.local"
        if env_file.exists():
            loaded = 0
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
                    loaded += 1
            if loaded:
                logger.debug("Loaded %d vars from %s", loaded, env_file)
            return
        search = search.parent


_api_mode: bool = False
_clients: list[genai.Client] = []
_client_cycle: itertools.cycle | None = None
_client_lock = threading.Lock()
_key_cooldowns: dict[int, float] = {}  # key_index -> cooldown_until timestamp
_KEY_COOLDOWN_SECONDS = 30.0


def _init_vertex_client() -> None:
    """Initialize a single Vertex AI client (default mode)."""
    global _api_mode, _clients, _client_cycle
    _api_mode = False
    _clients = [genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=REGION,
    )]
    _client_cycle = itertools.cycle(range(len(_clients)))


def _init_api_key_clients(keys: list[str]) -> None:
    """Initialize multiple API key clients for rotation."""
    global _api_mode, _clients, _client_cycle
    _api_mode = True
    _clients = [genai.Client(api_key=key.strip()) for key in keys if key.strip()]
    if not _clients:
        raise ValueError("No valid API keys provided")
    _client_cycle = itertools.cycle(range(len(_clients)))
    print(f"  API key rotation enabled: {len(_clients)} keys loaded")


def _get_client() -> genai.Client:
    """Get the next client from the pool (round-robin with cooldown skip)."""
    if not _clients:
        _init_vertex_client()

    with _client_lock:
        now = time.time()
        # Try up to len(_clients) times to find a non-cooled-down key
        for _ in range(len(_clients)):
            idx = next(_client_cycle)
            cooldown_until = _key_cooldowns.get(idx, 0)
            if now >= cooldown_until:
                return _clients[idx]
        # All keys on cooldown — use the one with shortest remaining cooldown
        idx = min(_key_cooldowns, key=_key_cooldowns.get, default=0)
        wait = max(0, _key_cooldowns.get(idx, 0) - now)
        if wait > 0:
            print(f"  RATE LIMIT: All {len(_clients)} keys on cooldown, waiting {wait:.0f}s...")
            time.sleep(wait)
        return _clients[idx]


def _mark_key_rate_limited(client: genai.Client) -> None:
    """Mark a client's key as rate-limited (cooldown period)."""
    with _client_lock:
        try:
            idx = _clients.index(client)
            _key_cooldowns[idx] = time.time() + _KEY_COOLDOWN_SECONDS
            if _api_mode:
                print(f"  RATE LIMIT: Key #{idx+1}/{len(_clients)} hit 429, cooling down {_KEY_COOLDOWN_SECONDS:.0f}s")
        except ValueError:
            pass


def _get_safety_settings() -> list[types.SafetySetting]:
    return [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]


# ============================================================================
# Models — Pydantic data models (inlined from audit_report.py + fidelity.py)
# ============================================================================

# Sentinel for fabricated tables (baseline=0, gemini>0).
FABRICATED_TABLE_SENTINEL = 999.0

# Quality grade bands
_GRADE_BANDS: list[tuple[int, str]] = [
    (85, "Good"),
    (60, "Fair"),
    (35, "Poor"),
    (0, "Critical"),
]

_ROUTING_LABELS = {
    "auto_approve": "Auto-Approve",
    "human_review": "Human Review",
    "reject": "Reject",
    "excluded": "Excluded",
}


def quality_grade(score: int) -> str:
    for threshold, label in _GRADE_BANDS:
        if score >= threshold:
            return label
    return "Critical"


class AuditRouting(StrEnum):
    AUTO_APPROVE = "auto_approve"
    HUMAN_REVIEW = "human_review"
    REJECT = "reject"
    EXCLUDED = "excluded"  # document type not suitable for HTML conversion


class AuditConcern(BaseModel):
    description: str
    severity: str  # critical | major | minor
    analysis: str | None = None
    source_pdf_quote: str | None = None
    output_quote: str | None = None
    location: str | None = None
    source: str = Field(default="unknown")
    check_id: str | None = None
    fidelity_dimension: str | None = None
    corroborated: bool = False


class SignalBreakdown(BaseModel):
    # LLM Fidelity
    fidelity_composite: float | None = None
    fidelity_content: float | None = None
    fidelity_structural: float | None = None
    fidelity_visual: float | None = None
    fidelity_routing: str | None = None
    fidelity_available: bool = False
    # Programmatic text fidelity (shingling)
    shingling_recall: float | None = None
    shingling_precision: float | None = None
    duplication_ratio: float | None = None
    # Baseline ratios
    word_count_ratio: float | None = None
    baseline_words: int | None = None
    gemini_words: int | None = None
    table_count_ratio: float | None = None
    image_count_ratio: float | None = None
    # Link counts
    source_link_count: int | None = None
    output_link_count: int | None = None
    link_count_ratio: float | None = None
    # Image placeholder detection
    image_placeholder_ratio: float | None = None
    # Pipeline confidence
    pipeline_confidence: float | None = None
    pipeline_routing: str | None = None
    pipeline_success: bool = True
    pipeline_error: str | None = None
    # Accessibility
    axe_violations: int = 0
    axe_critical: int = 0
    axe_serious: int = 0
    axe_moderate: int = 0
    axe_minor: int = 0
    axe_available: bool = False
    # Legacy
    legacy_accuracy_score: int | None = None
    # Agreement
    signal_agreement: str = "unknown"
    # Fabrication
    fabrication_detected_count: int = 0
    # Dedup stats
    deduplicated_count: int = 0
    dedup_removed_checks: list[tuple[str, Any]] = Field(default_factory=list, exclude=True)


class ScoreBreakdown(BaseModel):
    fidelity_composite: float | None = None
    raw_score: float = 0.0
    ceiling: float | None = None
    ceiling_reason: str | None = None
    available_components: int = 0


class InternalMetrics(BaseModel):
    decider_confidence: float = Field(ge=0.0, le=1.0)
    score_breakdown: ScoreBreakdown | None = None
    signals: SignalBreakdown
    processing_ms: int = 0
    fidelity_report_path: str | None = None
    routing_escalated_from: str | None = None


class AuditReport(BaseModel):
    document_id: str
    audited_at: datetime
    source_files: dict[str, str | None] = Field(default_factory=dict)
    quality_score: int = Field(ge=0, le=100, default=50)
    quality_grade: str = Field(default="")
    fidelity_composite: float | None = None
    fidelity_content: float | None = None
    fidelity_structural: float | None = None
    fidelity_visual: float | None = None
    routing: AuditRouting
    routing_label: str = Field(default="")
    reasoning: str
    decision_method: str
    concerns: list[AuditConcern] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    routing_changed: bool = Field(default=False)
    pipeline_original_routing: str | None = Field(default=None)
    exclusion_reason: str | None = None
    axe_compliance: dict | None = None  # Standalone axe-core WCAG results (informational only)
    internal: InternalMetrics


# --- Fidelity models ---

class DefectSeverity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


SEVERITY_WEIGHTS: dict[DefectSeverity, int] = {
    DefectSeverity.MINOR: 1,
    DefectSeverity.MAJOR: 5,
    DefectSeverity.CRITICAL: 25,
}


class EvaluationCallType(str, Enum):
    CONTENT = "content_fidelity"
    STRUCTURAL = "structural_fidelity"
    VISUAL = "visual_fidelity"


NORMALIZATION_FACTORS: dict[EvaluationCallType, int] = {
    EvaluationCallType.CONTENT: 86,
    EvaluationCallType.STRUCTURAL: 15,   # S1:1 + S2:5 + S3:1 + S4:1 + S5:1 + S6:1 + S7:5
    EvaluationCallType.VISUAL: 21,       # V1:5 + V2:1 + V3:5 + V4:5 + V5:5
}

CALL_WEIGHTS: dict[EvaluationCallType, float] = {
    EvaluationCallType.CONTENT: 0.35,
    EvaluationCallType.STRUCTURAL: 0.30,
    EvaluationCallType.VISUAL: 0.35,
}

CONTENT_CHECK_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]
STRUCTURAL_CHECK_IDS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
VISUAL_CHECK_IDS = ["V1", "V2", "V3", "V4", "V5"]

# Severity enforcement — each check has one defined severity.
# The LLM is told the severity in the prompt but sometimes overrides it.
# This map ensures scoring always uses the designed severity.
_CHECK_SEVERITY_MAP: dict[str, DefectSeverity] = {
    "C1": DefectSeverity.CRITICAL,
    "C2": DefectSeverity.CRITICAL,
    "C3": DefectSeverity.CRITICAL,
    "C4": DefectSeverity.MAJOR,
    "C5": DefectSeverity.MAJOR,
    "C6": DefectSeverity.MINOR,
    "S1": DefectSeverity.MINOR,     # was MAJOR — content duplication is rendering artifact
    "S2": DefectSeverity.MAJOR,
    "S3": DefectSeverity.MINOR,     # was MAJOR — table format naturally changes
    "S4": DefectSeverity.MINOR,     # was MAJOR — reading order shifts in HTML
    "S5": DefectSeverity.MINOR,
    "S6": DefectSeverity.MINOR,     # footnotes
    "S7": DefectSeverity.MAJOR,     # was MINOR — link injection critical for .gov docs
    "V1": DefectSeverity.MAJOR,     # tables broken (was V2)
    "V2": DefectSeverity.MINOR,     # images missing (was V3)
    "V3": DefectSeverity.MAJOR,     # content duplication (was V4)
    "V4": DefectSeverity.MAJOR,     # images next to wrong text
    "V5": DefectSeverity.MAJOR,     # formatting not preserved (bold/italic/underline)
}


class FidelityDefect(BaseModel):
    check_id: str
    check_name: str
    a_analysis: str
    b_evidence_source: str
    b_evidence_output: str
    b_evidence_location: str
    c_verdict: str
    d_severity: DefectSeverity | None = None


class CallResult(BaseModel):
    call_type: EvaluationCallType
    checks: list[FidelityDefect]
    score: float
    penalty_sum: int
    defect_count: int
    critical_count: int
    major_count: int
    minor_count: int
    raw_response: str | None = None
    evaluation_status: str = "complete"


class FidelityReport(BaseModel):
    document_id: str | None = None
    composite_score: float
    risk_level: str
    routing_action: str
    routing_reason: str
    knockout_triggered: str | None = None
    content_fidelity: CallResult | None = None
    structural_fidelity: CallResult | None = None
    visual_fidelity: CallResult | None = None
    total_defects: int = 0
    total_critical: int = 0
    total_major: int = 0
    total_minor: int = 0
    document_classification: str = "STANDARD"
    document_classification_rationale: str | None = None


# ============================================================================
# Prompts — imported from auditor_prompts.py (separate file for easy iteration)
# ============================================================================

from auditor_prompts import (
    build_content_fidelity_prompt as _build_content_fidelity_prompt,
    build_structural_fidelity_prompt as _build_structural_fidelity_prompt,
    build_visual_fidelity_prompt as _build_visual_fidelity_prompt,
    build_final_decider_prompt as _build_final_decider_prompt,
    _build_signal_summary,
    _build_concern_narratives,
)


# ============================================================================
# Shingling & Text Extraction
# ============================================================================

_NGRAM_SIZE = 4
_MIN_TEXT_LENGTH = 50
_FABRICATION_MIN_ELEMENT_CHARS = 20
_FABRICATION_OVERLAP_THRESHOLD = 0.10
_NUMBERS_ONLY_RE = re.compile(r"^[\d\s\-.,/$%()#:+]+$")

# 1x1 transparent PNG placeholder signature
_PLACEHOLDER_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
IMAGE_PLACEHOLDER_THRESHOLD = 0.75


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_shingles(text: str, n: int = _NGRAM_SIZE) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _compute_duplication_ratio(text: str) -> float:
    if len(text) < _MIN_TEXT_LENGTH:
        return 0.0
    sentences = re.split(r"[.!?]\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if len(sentences) < 2:
        return 0.0
    seen: set[str] = set()
    duplicate_chars = 0
    total_chars = 0
    for sentence in sentences:
        normalized = _normalize_text(sentence)
        total_chars += len(normalized)
        if normalized in seen:
            duplicate_chars += len(normalized)
        else:
            seen.add(normalized)
    return duplicate_chars / max(total_chars, 1)


def _extract_text_from_extraction_json(data: dict) -> str:
    """Extract all text from extraction-test JSON (pages[] format)."""
    parts: list[str] = []
    for page in data.get("pages") or []:
        for item in page.get("content") or []:
            item_type = item.get("type", "")
            if item_type in ("heading", "paragraph"):
                text = item.get("text", "")
                if text:
                    parts.append(text)
            elif item_type == "list":
                for li in item.get("items") or []:
                    if isinstance(li, dict):
                        text = li.get("text", "")
                    else:
                        text = str(li)
                    if text:
                        parts.append(text)
            elif item_type == "table":
                for cell in item.get("cells") or []:
                    text = cell.get("text", "")
                    if text:
                        parts.append(text)
            elif item_type == "link":
                text = item.get("text", "")
                if text:
                    parts.append(text)
            elif item_type == "form":
                for field in item.get("fields") or []:
                    label = field.get("label", "")
                    if label:
                        parts.append(label)
    return " ".join(parts)


def _extract_elements_from_extraction_json(
    data: dict,
) -> list[tuple[str, str]]:
    """Extract per-element text and type from extraction-test JSON."""
    elements: list[tuple[str, str]] = []
    for page in data.get("pages") or []:
        for item in page.get("content") or []:
            item_type = item.get("type", "")
            if item_type == "heading":
                text = item.get("text", "")
                if text:
                    elements.append(("heading", text))
            elif item_type == "paragraph":
                text = item.get("text", "")
                if text:
                    elements.append(("paragraph", text))
            elif item_type == "list":
                parts: list[str] = []
                for li in item.get("items") or []:
                    if isinstance(li, dict):
                        parts.append(li.get("text", ""))
                    else:
                        parts.append(str(li))
                text = " ".join(p for p in parts if p)
                if text:
                    elements.append(("list", text))
            elif item_type == "table":
                parts_t: list[str] = []
                for cell in item.get("cells") or []:
                    parts_t.append(cell.get("text", ""))
                text = " ".join(p for p in parts_t if p)
                if text:
                    elements.append(("table", text))
    return elements


def _extract_image_data_from_extraction_json(data: dict) -> list[str]:
    """Extract all image base64_data values from extraction-test JSON."""
    results: list[str] = []
    for page in data.get("pages") or []:
        for item in page.get("content") or []:
            if item.get("type") == "image":
                b64 = item.get("base64_data", "")
                if b64:
                    results.append(b64)
    return results


def _is_placeholder_image(b64: str) -> bool:
    return _PLACEHOLDER_BASE64 in b64


def _extract_pdf_text(pdf_path: Path) -> str | None:
    try:
        import fitz
    except ImportError:
        logger.info("PyMuPDF not available, skipping PDF text extraction")
        return None
    if not pdf_path.exists():
        return None
    try:
        doc = fitz.open(str(pdf_path))
        try:
            page_texts = []
            for page in doc:
                text = " ".join(
                    block[4] for block in page.get_text("blocks")
                    if block[6] == 0
                )
                page_texts.append(text)
            return " ".join(page_texts)
        finally:
            doc.close()
    except Exception as e:
        logger.warning("Failed to extract PDF text from %s: %s", pdf_path, e)
        return None


def _count_pdf_pages(pdf_path: Path) -> int | None:
    try:
        import fitz
    except ImportError:
        return None
    if not pdf_path.exists():
        return None
    try:
        doc = fitz.open(str(pdf_path))
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return None


def _count_pdf_links(pdf_path: Path) -> int | None:
    """Count URI links in source PDF using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        return None
    if not pdf_path.exists():
        return None
    try:
        doc = fitz.open(str(pdf_path))
        count = 0
        for page in doc:
            for link in page.get_links():
                if link.get("uri"):
                    count += 1
        doc.close()
        return count
    except Exception:
        return None


# ============================================================================
# Signal Collection (adapted for extraction-test JSON format)
# ============================================================================

def collect_signals(
    extraction_data: dict,
    pdf_path: Path | None = None,
    fidelity_report: FidelityReport | None = None,
    html_content: str | None = None,
) -> tuple[SignalBreakdown, list[AuditConcern]]:
    """Collect all signals from extraction JSON + optional PDF + fidelity report.

    Unlike the backend version which reads files from a directory, this version
    takes the data directly as arguments (standalone mode).
    """
    signals = SignalBreakdown()
    concerns: list[AuditConcern] = []

    # 1. Fidelity report signals
    if fidelity_report is not None:
        _collect_fidelity_signals_from_report(fidelity_report, signals, concerns)

    # 2. Compute baseline metrics from extraction JSON + PDF
    _collect_baseline_signals(extraction_data, pdf_path, signals, concerns)

    # 3. Compute text fidelity (shingling)
    _collect_text_fidelity_signals(extraction_data, pdf_path, signals, concerns)

    # 4. Image placeholder detection
    _collect_image_placeholder_signals(extraction_data, signals, concerns)

    # 5. Fabrication detection
    _detect_fabricated_content(extraction_data, pdf_path, signals, concerns)

    # 6. Additional programmatic detectors
    _detect_page_number_artifacts(extraction_data, html_content, signals, concerns)
    _detect_repeated_footer_text(extraction_data, concerns)
    _detect_duplicate_table_headers(extraction_data, concerns)
    _detect_table_caption_issues(extraction_data, concerns)
    _detect_degenerate_images(extraction_data, concerns)
    _detect_email_link_duplication(extraction_data, concerns)
    _detect_fullpage_image_with_text(extraction_data, concerns)
    _detect_scanned_doc_text_duplication(signals, concerns)
    _detect_image_count_inflation(signals, concerns)
    _detect_standalone_link_elements(extraction_data, concerns)
    _detect_header_footer_leakage(extraction_data, concerns)
    _detect_missing_hyperlinks(extraction_data, signals, concerns)
    _detect_hyperlinks_in_tables(extraction_data, signals, concerns)
    _detect_watermark_text(extraction_data, concerns)
    _detect_flat_lists(extraction_data, concerns)
    _detect_list_numbering_issues(extraction_data, concerns)
    _detect_flattened_sublists(extraction_data, concerns)
    _detect_form_elements(extraction_data, concerns)
    _detect_table_structure_issues(extraction_data, concerns)
    _detect_markdown_as_text(extraction_data, html_content, concerns)
    _detect_image_alt_as_text(extraction_data, concerns)
    _detect_link_displacement(extraction_data, concerns)

    # 7. Assess agreement
    signals.signal_agreement = _assess_agreement(signals, concerns)

    # 8. Deduplicate concerns
    concerns, removed_checks = _deduplicate_concerns(concerns, signals)
    signals.dedup_removed_checks = removed_checks

    return signals, concerns


def _collect_fidelity_signals_from_report(
    report: FidelityReport,
    signals: SignalBreakdown,
    concerns: list[AuditConcern],
) -> None:
    signals.fidelity_available = True
    signals.fidelity_composite = report.composite_score
    signals.fidelity_routing = report.routing_action.upper() if report.routing_action else None

    _call_labels = {
        "content_fidelity": ("fidelity_content", "Content"),
        "structural_fidelity": ("fidelity_structural", "Structural"),
        "visual_fidelity": ("fidelity_visual", "Visual"),
    }
    for call_key, (attr, label) in _call_labels.items():
        call_data = getattr(report, call_key, None)
        if call_data and call_data.evaluation_status == "complete":
            setattr(signals, attr, call_data.score)
        elif call_data and call_data.evaluation_status == "failed":
            concerns.append(AuditConcern(
                description=f"Fidelity {label} evaluation failed — dimension not assessed",
                source="fidelity_llm",
                severity="major",
                fidelity_dimension=label.lower(),
                analysis=f"The {label.lower()} fidelity call did not produce valid results.",
            ))

    # Extract defect concerns from fidelity report.
    # Severity is preserved as-is here — downgrading of non-corroborated
    # concerns happens later in _assess_agreement after corroboration
    # marking, so corroborated issues keep their full severity.
    for call_key in ("content_fidelity", "structural_fidelity", "visual_fidelity"):
        call_data = getattr(report, call_key, None)
        if not call_data:
            continue
        dimension = call_key.replace("_fidelity", "")
        for check in call_data.checks:
            if check.c_verdict == "FAIL":
                severity = (check.d_severity.value if check.d_severity else "major").lower()
                concerns.append(AuditConcern(
                    description=check.check_name or check.check_id,
                    source="fidelity_llm",
                    severity=severity,
                    check_id=check.check_id,
                    fidelity_dimension=dimension,
                    analysis=(check.a_analysis or "")[:500],
                    source_pdf_quote=(check.b_evidence_source or "")[:500],
                    output_quote=(check.b_evidence_output or "")[:500],
                    location=check.b_evidence_location,
                    corroborated=False,
                ))


def _collect_baseline_signals(
    extraction_data: dict,
    pdf_path: Path | None,
    signals: SignalBreakdown,
    concerns: list[AuditConcern],
) -> None:
    """Compute word/table/image counts from extraction JSON + PDF."""
    # Count from extraction JSON
    output_text = _extract_text_from_extraction_json(extraction_data)
    gemini_words = len(output_text.split())
    signals.gemini_words = gemini_words

    # Count tables and images from extraction JSON
    gemini_tables = 0
    gemini_images = 0
    for page in extraction_data.get("pages") or []:
        for item in page.get("content") or []:
            if item.get("type") == "table":
                gemini_tables += 1
            elif item.get("type") == "image":
                gemini_images += 1

    # Count baseline from PDF
    baseline_words = 0
    baseline_tables = 0
    baseline_images = 0

    if pdf_path and pdf_path.exists():
        baseline_raw = _extract_pdf_text(pdf_path)
        if baseline_raw:
            baseline_words = len(baseline_raw.split())

        # Use PyMuPDF for image counting (PyMuPDF has no native table detection)
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            for page in doc:
                baseline_images += len(page.get_images(full=True))
            doc.close()
        except Exception:
            pass

    signals.baseline_words = baseline_words

    if baseline_words > 0:
        signals.word_count_ratio = gemini_words / baseline_words
    else:
        signals.word_count_ratio = None

    # Table count ratio — PyMuPDF cannot count tables reliably, so we
    # only compute this when a baseline is available. Without a baseline,
    # set to None (unknown) rather than falsely flagging fabrication.
    signals.table_count_ratio = None

    # Image count ratio
    if baseline_images > 0:
        signals.image_count_ratio = gemini_images / baseline_images
    elif gemini_images == 0:
        signals.image_count_ratio = 1.0
    else:
        signals.image_count_ratio = None

    # Link counts — source PDF URI annotations vs extraction JSON link elements
    output_links = 0
    for page in extraction_data.get("pages") or []:
        for item in page.get("content") or []:
            if item.get("type") == "link":
                output_links += 1

    source_links = _count_pdf_links(pdf_path) if pdf_path else None
    signals.source_link_count = source_links
    signals.output_link_count = output_links
    if source_links is not None and source_links > 0:
        signals.link_count_ratio = output_links / source_links
    elif source_links == 0 and output_links == 0:
        signals.link_count_ratio = 1.0
    else:
        signals.link_count_ratio = None

    # Flag extreme word count deviations
    if signals.word_count_ratio is not None and baseline_words > 50:
        if signals.word_count_ratio < 0.30:
            concerns.append(AuditConcern(
                description=f"Severe word count loss: {signals.word_count_ratio:.0%} of baseline",
                source="programmatic",
                severity="critical",
                analysis=f"baseline={baseline_words}, output={gemini_words}",
            ))
        elif signals.word_count_ratio < 0.60:
            concerns.append(AuditConcern(
                description=f"Significant word count loss: {signals.word_count_ratio:.0%} of baseline",
                source="programmatic",
                severity="major",
                analysis=f"baseline={baseline_words}, output={gemini_words}",
            ))


def _collect_text_fidelity_signals(
    extraction_data: dict,
    pdf_path: Path | None,
    signals: SignalBreakdown,
    concerns: list[AuditConcern],
) -> None:
    output_text = _normalize_text(_extract_text_from_extraction_json(extraction_data))

    if pdf_path is None:
        return
    baseline_raw = _extract_pdf_text(pdf_path)
    if baseline_raw is None:
        return

    baseline_text = _normalize_text(baseline_raw)
    if len(baseline_text) < _MIN_TEXT_LENGTH or len(output_text) < _MIN_TEXT_LENGTH:
        return

    baseline_shingles = _extract_shingles(baseline_text)
    output_shingles = _extract_shingles(output_text)

    if baseline_shingles:
        overlap = len(baseline_shingles & output_shingles)
        signals.shingling_recall = overlap / len(baseline_shingles)
    else:
        signals.shingling_recall = 1.0

    if output_shingles:
        overlap = len(baseline_shingles & output_shingles)
        signals.shingling_precision = overlap / len(output_shingles)
    else:
        signals.shingling_precision = 1.0

    signals.duplication_ratio = _compute_duplication_ratio(output_text)

    # Guard: when both recall AND precision are near-zero despite
    # substantial text on both sides, PyMuPDF text extraction is
    # unreliable (non-Latin scripts, garbled encoding, or complex
    # layouts).  Don't generate hallucination/truncation concerns from
    # garbage shingling metrics — the LLM fidelity calls are far more
    # reliable in these cases.
    _pymupdf_unreliable = (
        signals.shingling_recall is not None
        and signals.shingling_precision is not None
        and signals.shingling_recall < 0.05
        and signals.shingling_precision < 0.05
        and (signals.baseline_words or 0) > 20
        and (signals.gemini_words or 0) > 50
    )
    if _pymupdf_unreliable:
        logger.info(
            "Skipping shingling concerns: recall=%.3f, precision=%.3f — "
            "PyMuPDF baseline likely garbled (non-Latin or encoding issue)",
            signals.shingling_recall,
            signals.shingling_precision,
        )
    else:
        if signals.shingling_precision is not None and signals.shingling_precision < 0.40:
            severity = "critical" if signals.shingling_precision < 0.20 else "major"
            concerns.append(AuditConcern(
                description=f"Low text precision (possible hallucination): {signals.shingling_precision:.0%}",
                source="programmatic",
                severity=severity,
            ))

        if signals.shingling_recall is not None and signals.shingling_recall < 0.40:
            severity = "critical" if signals.shingling_recall < 0.20 else "major"
            concerns.append(AuditConcern(
                description=f"Low text recall (possible truncation): {signals.shingling_recall:.0%}",
                source="programmatic",
                severity=severity,
            ))

    if signals.duplication_ratio is not None and signals.duplication_ratio > 0.15:
        severity = "major" if signals.duplication_ratio > 0.30 else "minor"
        concerns.append(AuditConcern(
            description=f"Content duplication detected: {signals.duplication_ratio:.0%}",
            source="programmatic",
            severity=severity,
        ))


def _collect_image_placeholder_signals(
    extraction_data: dict,
    signals: SignalBreakdown,
    concerns: list[AuditConcern],
    threshold: float = IMAGE_PLACEHOLDER_THRESHOLD,
) -> None:
    image_data = _extract_image_data_from_extraction_json(extraction_data)
    total_images = len(image_data)
    if total_images == 0:
        return

    placeholder_count = sum(1 for b64 in image_data if _is_placeholder_image(b64))
    ratio = placeholder_count / total_images
    signals.image_placeholder_ratio = ratio

    if ratio >= threshold:
        severity = "critical" if ratio >= 0.90 else "major"
        concerns.append(AuditConcern(
            description=(
                f"Image placeholder ratio {ratio:.0%}: "
                f"{placeholder_count}/{total_images} images are "
                f"broken/missing (transparent 1x1 placeholders)"
            ),
            source="programmatic",
            severity=severity,
            analysis=f"placeholder_count={placeholder_count}, total_images={total_images}",
        ))


def _detect_fabricated_content(
    extraction_data: dict,
    pdf_path: Path | None,
    signals: SignalBreakdown,
    concerns: list[AuditConcern],
) -> None:
    elements = _extract_elements_from_extraction_json(extraction_data)
    if not elements or pdf_path is None:
        return

    baseline_raw = _extract_pdf_text(pdf_path)
    if baseline_raw is None:
        return

    baseline_text = _normalize_text(baseline_raw)
    if len(baseline_text) < _MIN_TEXT_LENGTH:
        return

    source_shingles = _extract_shingles(baseline_text)
    if not source_shingles:
        return

    # Guard: skip fabrication detection when PyMuPDF text extraction is
    # unreliable.  This happens with non-Latin scripts (Arabic, Thai, CJK)
    # and some scanned docs where PyMuPDF returns garbled text.  Shingling
    # produces near-zero overlap → every paragraph looks "fabricated."
    # When shingling already ran and found < 5% recall with substantial
    # text on both sides, the baseline is garbage — don't trust it.
    if (
        signals.shingling_recall is not None
        and signals.shingling_recall < 0.05
        and (signals.gemini_words or 0) > 50
        and (signals.baseline_words or 0) > 20
    ):
        logger.info(
            "Skipping fabrication detection: shingling_recall=%.3f "
            "indicates unreliable PyMuPDF text extraction (non-Latin or garbled)",
            signals.shingling_recall,
        )
        return

    fabrication_count = 0
    for el_type, raw_text in elements:
        if el_type not in ("heading", "paragraph"):
            continue
        normalized = _normalize_text(raw_text)
        if len(normalized) < _FABRICATION_MIN_ELEMENT_CHARS:
            continue
        if _NUMBERS_ONLY_RE.match(normalized):
            continue
        el_shingles = _extract_shingles(normalized)
        if not el_shingles:
            continue
        overlap = len(el_shingles & source_shingles)
        overlap_ratio = overlap / len(el_shingles)
        if overlap_ratio < _FABRICATION_OVERLAP_THRESHOLD:
            fabrication_count += 1
            snippet = raw_text[:120].replace("\n", " ")
            if len(raw_text) > 120:
                snippet += "..."
            if el_type == "paragraph":
                severity = "critical"
                desc = f'Fabricated paragraph (0 source overlap): "{snippet}"'
            else:
                severity = "major"
                desc = f'Fabricated heading (0 source overlap): "{snippet}"'
            concerns.append(AuditConcern(
                description=desc,
                source="programmatic",
                severity=severity,
                analysis=f"shingle_overlap={overlap_ratio:.2%}, threshold={_FABRICATION_OVERLAP_THRESHOLD:.0%}",
                output_quote=raw_text[:500],
            ))
    signals.fabrication_detected_count = fabrication_count


# ============================================================================
# Additional Programmatic Detectors
# ============================================================================

_PAGE_NUMBER_PATTERNS = [
    re.compile(r"^Page\s+\d+\s*(of\s+\d+)?$", re.IGNORECASE),
    re.compile(r"^-\s*\d+\s*-$"),
    re.compile(r"^\d+\s*$"),
]


def _detect_page_number_artifacts(
    extraction_data: dict,
    html_content: str | None,
    signals: SignalBreakdown,
    concerns: list[AuditConcern],
) -> None:
    """Detect page numbers and footers retained in the output.

    Page numbers like 'Page 1 of 3' or '- 2 -' should be stripped during
    conversion since the HTML is one continuous document.
    """
    page_number_count = 0
    page_x_of_y_count = 0
    examples: list[str] = []

    # Check extraction JSON paragraphs
    for page in extraction_data.get("pages") or []:
        for item in page.get("content") or []:
            if item.get("type") != "paragraph":
                continue
            text = (item.get("text") or "").strip()
            if not text or len(text) > 40:
                continue
            for pat in _PAGE_NUMBER_PATTERNS:
                if pat.match(text):
                    page_number_count += 1
                    if "of" in text.lower():
                        page_x_of_y_count += 1
                    if len(examples) < 3:
                        examples.append(text)
                    break

    # Also check HTML if available
    if html_content:
        import re as _re
        for m in _re.finditer(r">\s*(Page\s+\d+\s*(?:of\s+\d+)?)\s*<", html_content, _re.IGNORECASE):
            txt = m.group(1).strip()
            page_x_of_y_count += 1
            if len(examples) < 3 and txt not in examples:
                examples.append(txt)

    # "Page X of Y" is NEVER legitimate body content — flag even 1 occurrence
    # Page numbers in HTML break accessibility (screen readers see them as content)
    if page_x_of_y_count > 0:
        concerns.append(AuditConcern(
            description=f"Page number artifacts retained: {', '.join(examples[:3])}",
            source="programmatic",
            severity="major",
            analysis=f"page_x_of_y_count={page_x_of_y_count}, total={page_number_count}",
        ))
    elif page_number_count >= 3:
        concerns.append(AuditConcern(
            description=f"Page number artifacts retained ({page_number_count} found): {', '.join(examples[:3])}",
            source="programmatic",
            severity="major",
            analysis=f"page_number_count={page_number_count}",
        ))


def _detect_repeated_footer_text(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect repeated short text blocks that look like headers/footers.

    A short text block (< 80 chars) appearing 3+ times in different pages
    is likely a running header/footer that should have been stripped.
    """
    short_texts: dict[str, int] = {}
    for page in extraction_data.get("pages") or []:
        seen_on_page: set[str] = set()
        for item in page.get("content") or []:
            if item.get("type") not in ("paragraph", "heading"):
                continue
            text = (item.get("text") or "").strip()
            if not text or len(text) > 80 or len(text) < 5:
                continue
            normalized = text.lower().strip()
            if normalized not in seen_on_page:
                seen_on_page.add(normalized)
                short_texts[normalized] = short_texts.get(normalized, 0) + 1

    total_pages = len(extraction_data.get("pages") or [])

    # Semantic exclusion: skip text that looks like a section header, not a footer
    _section_header_words = {
        "objective", "objectives", "goal", "goals", "topic", "topics",
        "article", "overview", "summary", "introduction", "conclusion",
        "feature", "section", "chapter", "module", "lesson", "unit",
        "learning", "covered", "description", "purpose", "background",
        "discussion", "recommendation", "recommendations", "reference",
        "references", "appendix", "agenda", "review",
    }

    def _looks_like_section_header(t: str) -> bool:
        """Return True if text looks like a content section header, not a footer."""
        words = set(t.replace(":", "").replace("*", "").split())
        # Single common word (no digits) — likely a section label
        if len(words) == 1 and not re.search(r"\d", t):
            return words.issubset(_section_header_words)
        # Multi-word but contains section-header vocabulary
        if words & _section_header_words and not re.search(r"\d{4}|page\s*\d|\bp\.?\s*\d", t):
            return True
        return False

    for text, count in short_texts.items():
        if count >= 3:
            # Proportional threshold: require ≥40% of pages for docs with 8+ pages
            # (short docs keep absolute threshold of 3)
            if total_pages >= 8 and count / total_pages < 0.40:
                continue
            # Skip text that looks like a section header
            if _looks_like_section_header(text):
                continue
            concerns.append(AuditConcern(
                description=f"Repeated footer/header text ({count}x): \"{text[:60]}\"",
                source="programmatic",
                severity="minor",
                analysis=f"Short text block appears on {count}/{total_pages} pages",
            ))
        elif count == 2:
            # 2-page repetition: only flag if it looks like a footer/header
            # (contains page-like numbers, dates, or org names)
            if re.search(r"\d{4}|page|\bp\.?\s*\d|minutes|agenda|committee", text, re.IGNORECASE):
                concerns.append(AuditConcern(
                    description=f"Repeated footer/header text ({count}x): \"{text[:60]}\"",
                    source="programmatic",
                    severity="minor",
                    analysis=f"Short text with footer-like pattern appears on {count} pages",
                ))

    # --- Positional repetition: first/last element same across pages ---
    # Track first and last text on each page (headings/paragraphs only)
    first_texts: dict[str, int] = {}
    last_texts: dict[str, int] = {}
    for page in extraction_data.get("pages") or []:
        content = page.get("content") or []
        # Find first text element
        for item in content:
            if item.get("type") in ("heading", "paragraph"):
                text = (item.get("text") or "").strip()
                if text and len(text) < 80:
                    norm = text.lower().strip()
                    first_texts[norm] = first_texts.get(norm, 0) + 1
                break
        # Find last text element
        for item in reversed(content):
            if item.get("type") in ("heading", "paragraph"):
                text = (item.get("text") or "").strip()
                if text and len(text) < 80:
                    norm = text.lower().strip()
                    last_texts[norm] = last_texts.get(norm, 0) + 1
                break

    for text, count in {**first_texts, **last_texts}.items():
        if count >= 2 and text not in short_texts:
            concerns.append(AuditConcern(
                description=f"Positional header/footer text ({count}x): \"{text[:60]}\"",
                source="programmatic",
                severity="minor",
                analysis=(
                    f"Text consistently appears as first or last element "
                    f"on {count}/{total_pages} pages — likely a running header/footer"
                ),
            ))

    # --- Repeated images across pages (e.g., logos/seals in headers) ---
    # Count how many pages each image description appears on (any position)
    img_desc_pages: dict[str, int] = {}
    for page in extraction_data.get("pages") or []:
        seen_descs: set[str] = set()
        for item in page.get("content") or []:
            if item.get("type") == "image":
                desc = (item.get("description") or "").strip()
                if desc and len(desc) > 10:
                    norm = desc[:50].lower()
                    if norm not in seen_descs:
                        seen_descs.add(norm)
                        img_desc_pages[norm] = img_desc_pages.get(norm, 0) + 1

    for desc, count in img_desc_pages.items():
        if count >= 3 and total_pages >= 3:
            ratio = count / total_pages
            if ratio >= 0.5:
                concerns.append(AuditConcern(
                    description=(
                        f"Repeated header/decorative image on {count}/{total_pages} pages: "
                        f"\"{desc[:50]}...\""
                    ),
                    source="programmatic",
                    severity="minor" if ratio < 0.8 else "major",
                    analysis=(
                        f"Same image appears on {count}/{total_pages} pages "
                        f"({ratio:.0%}). Likely a decorative header/logo/seal that "
                        "should appear once in HTML, not on every page."
                    ),
                ))


def _detect_duplicate_table_headers(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect adjacent tables with identical header rows.

    When a PDF table spans pages, headers repeat on each page. Gemini may
    produce separate HTML tables for each page, each with the same header.
    In HTML the duplicate header is redundant.
    """
    prev_header: list[str] | None = None
    dup_count = 0
    for page in extraction_data.get("pages") or []:
        for item in page.get("content") or []:
            if item.get("type") != "table":
                continue
            rows = item.get("rows") or []
            if not rows:
                continue
            # Find header row
            header_row = None
            for row in rows:
                if row.get("is_header_row") or row.get("is_header"):
                    header_row = [
                        (cell.get("text") or "").strip().lower()
                        for cell in (row.get("cells") or [])
                    ]
                    break
            if header_row is None and rows:
                # Use first row as header
                header_row = [
                    (cell.get("text") or "").strip().lower()
                    for cell in (rows[0].get("cells") or [])
                ]
            if header_row and prev_header and header_row == prev_header:
                dup_count += 1
            prev_header = header_row

    if dup_count > 0:
        concerns.append(AuditConcern(
            description=f"Duplicate table headers from page breaks: {dup_count} adjacent table(s) share identical headers",
            source="programmatic",
            severity="minor",
            analysis=f"dup_header_count={dup_count}",
        ))


def _detect_table_caption_issues(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect table caption/header misidentification.

    Pattern A: First row has a single cell spanning all columns while subsequent
    rows have multiple columns — the first row is likely a caption that was
    coded as a header row.

    Pattern B: First row contains data values (numbers, dates, dollar amounts)
    rather than column labels — data row may be marked as header.

    Handles both table formats:
    - Format A: {"rows": [{"cells": [...], "is_header_row": ...}]}
    - Format B: {"cells": [{"text": ..., "row_start": ..., "column_start": ...}]}
    """
    _DATA_PATTERN = re.compile(
        r"^[\s$]*\d[\d,./\-%]*\s*$"  # numbers, currency, dates, percentages
        r"|^\$[\d,.]+"               # dollar amounts
        r"|^\d{1,2}/\d{1,2}/\d{2,4}$"  # dates mm/dd/yyyy
        r"|^N/?A$"                   # N/A
    )

    caption_as_header = 0
    data_as_header = 0
    caption_examples: list[str] = []
    data_examples: list[str] = []

    for page in extraction_data.get("pages") or []:
        page_num = page.get("page_number", "?")
        for item in page.get("content") or []:
            if item.get("type") != "table":
                continue

            rows_data = item.get("rows") or []
            if rows_data:
                # Format A: rows[].cells[]
                if len(rows_data) < 2:
                    continue
                first_row_cells = rows_data[0].get("cells") or []
                second_row_cells = rows_data[1].get("cells") or []
                # Pattern A: single-cell first row, multi-cell second row
                if (len(first_row_cells) == 1
                        and len(second_row_cells) >= 2):
                    caption_as_header += 1
                    text = (first_row_cells[0].get("text") or "")[:60]
                    if len(caption_examples) < 3:
                        caption_examples.append(f"p{page_num}: \"{text}\"")
                    continue
                # Pattern B: header row with data values
                if (rows_data[0].get("is_header_row")
                        or rows_data[0].get("is_header")):
                    data_cells = [
                        (c.get("text") or "").strip()
                        for c in first_row_cells if (c.get("text") or "").strip()
                    ]
                    if data_cells and all(
                        _DATA_PATTERN.match(t) for t in data_cells
                    ):
                        data_as_header += 1
                        if len(data_examples) < 3:
                            data_examples.append(
                                f"p{page_num}: [{', '.join(data_cells[:3])}]"
                            )
            else:
                # Format B: flat cells[] with row_start/column_start
                cells = item.get("cells") or []
                if not cells:
                    continue
                # Group cells by row_start
                rows_by_idx: dict[int, list[dict]] = {}
                for cell in cells:
                    ri = cell.get("row_start", 0)
                    rows_by_idx.setdefault(ri, []).append(cell)
                sorted_row_idxs = sorted(rows_by_idx.keys())
                if len(sorted_row_idxs) < 2:
                    continue
                first_row_idx = sorted_row_idxs[0]
                second_row_idx = sorted_row_idxs[1]
                first_cells = rows_by_idx[first_row_idx]
                second_cells = rows_by_idx[second_row_idx]
                # Determine effective column count from second row
                second_col_count = sum(
                    c.get("num_columns", 1) for c in second_cells
                )
                # Pattern A: single cell in first row spanning all columns
                if (len(first_cells) == 1
                        and first_cells[0].get("num_columns", 1) >= second_col_count
                        and second_col_count >= 2):
                    caption_as_header += 1
                    text = (first_cells[0].get("text") or "")[:60]
                    if len(caption_examples) < 3:
                        caption_examples.append(f"p{page_num}: \"{text}\"")
                    continue
                # Pattern B: first row all data values (no explicit header marker
                # in Format B, so only flag when ALL first-row cells are numeric/data
                # and second row looks like labels)
                first_texts = [
                    (c.get("text") or "").strip()
                    for c in first_cells if (c.get("text") or "").strip()
                ]
                second_texts = [
                    (c.get("text") or "").strip()
                    for c in second_cells if (c.get("text") or "").strip()
                ]
                if (first_texts and len(first_texts) >= 2
                        and all(_DATA_PATTERN.match(t) for t in first_texts)
                        and second_texts
                        and not all(_DATA_PATTERN.match(t) for t in second_texts)):
                    data_as_header += 1
                    if len(data_examples) < 3:
                        data_examples.append(
                            f"p{page_num}: [{', '.join(first_texts[:3])}]"
                        )

    if caption_as_header > 0:
        concerns.append(AuditConcern(
            description=(
                f"Table caption coded as header row: {caption_as_header} table(s) "
                f"have a single spanning first row that is likely a caption/title"
            ),
            source="programmatic",
            severity="major",
            check_id="S3",
            analysis=(
                f"caption_as_header={caption_as_header}. "
                f"Examples: {'; '.join(caption_examples)}. "
                "First row spans all columns — should be a <caption> element, "
                "not a header row."
            ),
        ))
    if data_as_header > 0:
        concerns.append(AuditConcern(
            description=(
                f"Data row marked as header: {data_as_header} table(s) have "
                f"numeric/data values in the header row"
            ),
            source="programmatic",
            severity="major",
            check_id="S3",
            analysis=(
                f"data_as_header={data_as_header}. "
                f"Examples: {'; '.join(data_examples)}. "
                "Header rows should contain column labels, not data values."
            ),
        ))


def _detect_degenerate_images(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect images with extreme aspect ratios (slivers) or tiny dimensions.

    These are typically rendering artifacts — thin strips from page boundaries
    or 1-pixel scan artifacts that provide no value.
    """
    import base64
    from io import BytesIO

    degen_count = 0
    for page in extraction_data.get("pages") or []:
        for item in page.get("content") or []:
            if item.get("type") != "image":
                continue
            b64 = item.get("base64_data", "")
            if not b64 or _is_placeholder_image(b64):
                continue
            # Try to get image dimensions
            try:
                from PIL import Image
                raw = base64.b64decode(b64[:100000])  # limit decode size
                img = Image.open(BytesIO(raw))
                w, h = img.size
                if w < 20 or h < 20:
                    degen_count += 1
                elif max(w, h) / max(min(w, h), 1) > 10:
                    degen_count += 1
            except Exception:
                continue

    if degen_count >= 2:
        concerns.append(AuditConcern(
            description=f"Degenerate images detected: {degen_count} images with extreme aspect ratio or tiny dimensions",
            source="programmatic",
            severity="minor",
            analysis=f"degenerate_count={degen_count}, threshold=2",
        ))


def _detect_email_link_duplication(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect emails/URLs appearing both inside and outside tables.

    The link injection pipeline may extract contacts from tables and also
    add them as standalone links, causing duplication.
    """
    import re as _re
    email_re = _re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    table_emails: set[str] = set()
    non_table_emails: set[str] = set()

    for page in extraction_data.get("pages") or []:
        for item in page.get("content") or []:
            item_type = item.get("type", "")
            if item_type == "table":
                for row in item.get("rows") or []:
                    for cell in row.get("cells") or []:
                        text = cell.get("text", "")
                        for m in email_re.finditer(text):
                            table_emails.add(m.group().lower())
            elif item_type in ("paragraph", "link"):
                text = item.get("text", "") or item.get("url", "")
                for m in email_re.finditer(text):
                    non_table_emails.add(m.group().lower())

    duplicated = table_emails & non_table_emails
    if duplicated:
        examples = list(duplicated)[:3]
        concerns.append(AuditConcern(
            description=f"Emails duplicated outside tables: {', '.join(examples)}",
            source="programmatic",
            severity="major",
            analysis=f"duplicated_count={len(duplicated)}, in_table={len(table_emails)}, standalone={len(non_table_emails)}",
        ))


def _detect_fullpage_image_with_text(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect full-page-sized images alongside extracted text on the same page.

    When a scanned page is brought in as both an image AND OCR'd text,
    the user sees the content twice — once in the image and once as text.
    This is redundant and confusing for screen readers.

    Works for ANY document (not just scanned docs) — a page with a large
    image AND substantial text content alongside it is suspicious.
    """
    import base64
    suspect_pages = 0
    for page in extraction_data.get("pages") or []:
        content_items = page.get("content") or []
        images = [i for i in content_items if i.get("type") == "image"]
        text_items = [
            i for i in content_items
            if i.get("type") in ("paragraph", "heading", "list")
        ]
        text_word_count = sum(
            len((i.get("text") or "").split())
            for i in text_items
        )
        if not images or text_word_count < 20:
            continue
        # Check for large images (page-sized)
        for img in images:
            img_data = img.get("image_data") or img.get("data") or ""
            if not img_data:
                continue
            try:
                raw = base64.b64decode(img_data)
                if len(raw) < 10000:
                    continue  # Too small to be a page scan
                # Try to get dimensions
                from io import BytesIO
                from PIL import Image
                pil_img = Image.open(BytesIO(raw))
                w, h = pil_img.size
                # Page-sized: both dimensions > 500px, aspect ratio 1.0-2.0
                if w > 500 and h > 500:
                    aspect = max(w, h) / min(w, h)
                    if aspect < 2.5:
                        suspect_pages += 1
                        break
            except Exception:
                # If image is large (> 50KB raw), treat as suspicious
                if len(raw) > 50000:
                    suspect_pages += 1
                    break

    if suspect_pages > 0:
        concerns.append(AuditConcern(
            description=f"Full-page image alongside extracted text ({suspect_pages} page(s))",
            source="programmatic",
            severity="major",
            analysis=f"suspect_pages={suspect_pages}. A scanned/screenshot page image "
                      "duplicates content already present as text elements.",
        ))


def _detect_scanned_doc_text_duplication(
    signals: SignalBreakdown,
    concerns: list[AuditConcern],
) -> None:
    """Flag fully-scanned documents where all text was OCR'd from images.

    When baseline_words == 0 (PyMuPDF found no text in the source PDF) but
    Gemini extracted substantial text, the source is image-only (scanned).
    The output likely shows both the scanned page image AND the OCR'd text,
    creating redundant content that confuses screen readers.
    """
    baseline = signals.baseline_words or 0
    gemini = signals.gemini_words or 0
    if baseline == 0 and gemini >= 50:
        concerns.append(AuditConcern(
            description=(
                f"Scanned document: source has 0 extractable text but output "
                f"contains {gemini} words (OCR'd from images)"
            ),
            source="programmatic",
            severity="major",
            analysis=(
                f"baseline_words=0, gemini_words={gemini}. "
                "Entire document is image-based; output likely duplicates "
                "content as both page image and extracted text."
            ),
        ))


def _detect_image_count_inflation(
    signals: SignalBreakdown,
    concerns: list[AuditConcern],
) -> None:
    """Flag when output has MORE images than source PDF — indicates duplication.

    image_count_ratio > 1.0 means Gemini produced more image elements than
    PyMuPDF found in the source, which typically means images were duplicated
    (e.g., same image placed in sidebar AND inline, or decorative graphics
    repeated).
    """
    ratio = signals.image_count_ratio
    if ratio is not None and ratio > 1.0:
        concerns.append(AuditConcern(
            description=(
                f"Output has more images than source PDF "
                f"(ratio {ratio:.1f}x) — possible image duplication"
            ),
            source="programmatic",
            severity="major",
            analysis=(
                f"image_count_ratio={ratio:.2f}. "
                "More images in output than in source indicates duplicated or "
                "fabricated image elements."
            ),
        ))


def _detect_standalone_link_elements(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect top-level link elements that should be inline within text.

    In properly structured HTML, hyperlinks are inline within paragraphs.
    When the extraction produces standalone 'link' type elements at the
    page content level, it means links were displaced from their original
    inline position — a structural defect for accessibility.
    """
    standalone_count = 0
    examples: list[str] = []
    for page in extraction_data.get("pages") or []:
        for item in page.get("content") or []:
            if item.get("type") == "link":
                standalone_count += 1
                url = item.get("url") or item.get("text") or ""
                if len(examples) < 3:
                    examples.append(url[:60])

    if standalone_count > 0:
        # ≤2 displaced links is a minor formatting issue;
        # ≥3 indicates a systemic structural problem
        sev = "major" if standalone_count >= 3 else "minor"
        concerns.append(AuditConcern(
            description=(
                f"Displaced hyperlinks: {standalone_count} link(s) extracted "
                f"as standalone elements instead of inline text"
            ),
            source="programmatic",
            severity=sev,
            analysis=(
                f"standalone_links={standalone_count}, "
                f"examples={examples}. "
                "Links should be inline within paragraphs, not separate blocks."
            ),
        ))


def _detect_header_footer_leakage(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect header/footer elements retained in the output content.

    Elements with type 'header_footer' in the extraction JSON represent
    running headers, footers, or page numbers from the source PDF. These
    should be stripped during conversion since HTML is a continuous document.
    Their presence indicates the pipeline failed to remove them.
    """
    leaked_count = 0
    examples: list[str] = []
    for page in extraction_data.get("pages") or []:
        for item in page.get("content") or []:
            if item.get("type") == "header_footer":
                leaked_count += 1
                text = (item.get("text") or "").strip()
                if text and len(examples) < 3:
                    examples.append(text[:60])

    if leaked_count > 0:
        example_str = "; ".join(examples) if examples else "(no text)"
        concerns.append(AuditConcern(
            description=(
                f"Header/footer elements retained in output ({leaked_count}): "
                f"{example_str}"
            ),
            source="programmatic",
            severity="major",
            analysis=(
                f"header_footer_count={leaked_count}. "
                "Running headers/footers/page numbers should be stripped "
                "during PDF-to-HTML conversion."
            ),
        ))


# URL pattern for detecting bare URLs in extraction text
_URL_PATTERN = re.compile(
    r'https?://\S+'            # http:// or https://
    r'|www\.\S+'               # www. prefix
    r'|\S+\.(?:gov|edu|org|com|net|mil)\b(?:/\S*)?',  # domain with common TLDs
    re.IGNORECASE,
)


def _detect_missing_hyperlinks(
    extraction_data: dict,
    signals: SignalBreakdown,
    concerns: list[AuditConcern],
) -> None:
    """Detect missing hyperlinks: source PDF links lost + bare URLs in text.

    Two distinct problems:
    1. Source PDF has URI annotations that were not carried into the output.
    2. URL-like text in paragraphs/lists was not converted to clickable links.
    """
    source = signals.source_link_count
    output = signals.output_link_count

    # --- Problem 1: PDF link annotations lost during extraction ---
    if source is not None and source >= 3 and (output or 0) == 0:
        concerns.append(AuditConcern(
            description=(
                f"Source PDF has {source} hyperlinks but output has none — "
                f"all links lost during extraction"
            ),
            source="programmatic",
            severity="critical",
            check_id="S7",
            analysis=(
                f"source_link_count={source}, output_link_count={output}. "
                "PDF URI annotations were not preserved as link elements."
            ),
        ))
    elif (
        source is not None
        and source >= 3
        and output is not None
        and source > 0
        and (output / source) < 0.25
    ):
        concerns.append(AuditConcern(
            description=(
                f"Only {output}/{source} ({output/source:.0%}) of source "
                f"hyperlinks preserved in output"
            ),
            source="programmatic",
            severity="major",
            check_id="S7",
            analysis=(
                f"source_link_count={source}, output_link_count={output}, "
                f"link_count_ratio={output/source:.2f}. "
                "Most PDF URI annotations were lost during extraction."
            ),
        ))

    # --- Problem 2: Bare URLs in text not hyperlinked ---
    # Collect URLs that appear in text of known link elements
    linked_urls: set[str] = set()
    bare_urls: list[str] = []

    for page in extraction_data.get("pages") or []:
        for item in page.get("content") or []:
            item_type = item.get("type", "")

            if item_type == "link":
                url = item.get("url") or item.get("text") or ""
                linked_urls.add(url.strip().rstrip("/").lower())
                continue

            # Scan text content of paragraphs, headings, list items
            texts_to_scan: list[str] = []
            if item_type in ("paragraph", "heading"):
                texts_to_scan.append(item.get("text") or "")
            elif item_type == "list":
                for li in item.get("items") or []:
                    if isinstance(li, dict):
                        texts_to_scan.append(li.get("text") or "")
                    elif isinstance(li, str):
                        texts_to_scan.append(li)

            for text in texts_to_scan:
                for m in _URL_PATTERN.finditer(text):
                    url = m.group(0).rstrip(".,;:)").lower()
                    normalized = url.rstrip("/")
                    if normalized not in linked_urls:
                        bare_urls.append(m.group(0)[:80])

    if bare_urls:
        unique_bare = list(dict.fromkeys(bare_urls))  # dedupe preserving order
        examples = unique_bare[:5]
        concerns.append(AuditConcern(
            description=(
                f"{len(unique_bare)} bare URL(s) in text not converted to "
                f"hyperlinks: {'; '.join(examples)}"
            ),
            source="programmatic",
            severity="minor" if len(unique_bare) <= 2 else "major",
            analysis=(
                f"bare_url_count={len(unique_bare)}. "
                "URLs appear as plain text instead of clickable links."
            ),
        ))


def _detect_hyperlinks_in_tables(
    extraction_data: dict,
    signals: SignalBreakdown,
    concerns: list[AuditConcern],
) -> None:
    """Detect hyperlinks in table cells that were lost during extraction.

    Government documents frequently have hyperlinked text inside table cells
    (organization names, reference codes, etc.). The extraction pipeline
    often strips these links, leaving plain text where clickable links
    should be.
    """
    # Only relevant when source has links
    if (signals.source_link_count or 0) < 1:
        return

    bare_urls_in_tables = 0
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    examples: list[str] = []

    for page in extraction_data.get("pages") or []:
        page_num = page.get("page_number", "?")
        for item in page.get("content") or []:
            if item.get("type") != "table":
                continue
            # Handle both table formats:
            # Format A: {"rows": [{"cells": [...]}]}
            # Format B: {"cells": [{"text": ..., "row_start": ..., "column_start": ...}]}
            all_cells: list[dict | str] = []
            rows = item.get("rows") or []
            if rows:
                for row in rows:
                    all_cells.extend(row.get("cells") or [])
            else:
                all_cells = item.get("cells") or []

            for cell in all_cells:
                text = cell.get("text", "") if isinstance(cell, dict) else str(cell)
                if not text:
                    continue
                # Check for URLs in cell text
                for m in _URL_PATTERN.finditer(text):
                    bare_urls_in_tables += 1
                    if len(examples) < 5:
                        examples.append(f"p{page_num}: {m.group(0)[:50]}")
                # Check for emails in cell text
                for m in email_pattern.finditer(text):
                    bare_urls_in_tables += 1
                    if len(examples) < 5:
                        examples.append(f"p{page_num}: {m.group(0)[:50]}")

    if bare_urls_in_tables >= 2:
        concerns.append(AuditConcern(
            description=(
                f"Table cells contain {bare_urls_in_tables} unlinked URL(s)/email(s) "
                f"that should be hyperlinks"
            ),
            source="programmatic",
            severity="major",
            check_id="S7",
            analysis=(
                f"bare_urls_in_tables={bare_urls_in_tables}. "
                f"Examples: {'; '.join(examples[:3])}. "
                "URLs and emails in table cells should be clickable links."
            ),
        ))


# Patterns that indicate watermark/stamp text in PDFs
_WATERMARK_PATTERNS = re.compile(
    r"^(?:DRAFT|CONFIDENTIAL|DO NOT DISTRIBUTE|SAMPLE|UNOFFICIAL|"
    r"PRELIMINARY|WORKING DRAFT|FOR REVIEW|NOT FOR DISTRIBUTION|"
    r"APPROVED(?:\s+BY\b)?)",
    re.IGNORECASE,
)


def _detect_watermark_text(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect watermark/stamp text repeated across pages.

    Watermarks like DRAFT, APPROVED, CONFIDENTIAL are PDF overlays that
    sometimes get extracted as short paragraph text. When the same watermark
    text appears on multiple pages, it indicates a document-level stamp
    that should be noted for review.
    """
    watermark_hits: dict[str, int] = {}
    total_pages = len(extraction_data.get("pages") or [])

    for page in extraction_data.get("pages") or []:
        seen_on_page: set[str] = set()
        for item in page.get("content") or []:
            if item.get("type") not in ("paragraph", "heading"):
                continue
            text = (item.get("text") or "").strip()
            if not text or len(text) > 60:
                continue
            m = _WATERMARK_PATTERNS.match(text)
            if m:
                key = m.group(0).upper()
                if key not in seen_on_page:
                    seen_on_page.add(key)
                    watermark_hits[key] = watermark_hits.get(key, 0) + 1

    for wm_text, count in watermark_hits.items():
        if count >= 2:
            concerns.append(AuditConcern(
                description=(
                    f"Watermark/stamp text \"{wm_text}\" appears on "
                    f"{count}/{total_pages} pages"
                ),
                source="programmatic",
                severity="minor",
                analysis=(
                    f"watermark=\"{wm_text}\", page_count={count}/{total_pages}. "
                    "Repeated watermark/stamp text indicates a document overlay "
                    "that may be duplicated in the HTML output."
                ),
            ))


def _detect_flat_lists(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect lists with mixed numbering types at the same depth level.

    When a list has both numeric (1, 2, 3) and letter (a, b, c) markers at
    depth 0, it likely means sub-points were flattened into the parent list.
    """
    flat_list_count = 0
    examples: list[str] = []

    for page in extraction_data.get("pages") or []:
        for item in page.get("content") or []:
            if item.get("type") != "list":
                continue
            items = item.get("items") or []
            if len(items) < 2:
                continue

            # Check for mixed marker patterns at root level
            marker_types: set[str] = set()
            for li in items:
                text = li.get("text") or "" if isinstance(li, dict) else str(li)
                text = text.strip()
                if re.match(r"^\d+[.)]\s", text):
                    marker_types.add("numeric")
                elif re.match(r"^[a-z][.)]\s", text, re.IGNORECASE):
                    marker_types.add("letter")
                elif re.match(r"^(?:i{1,3}|iv|vi{0,3})[.)]\s", text, re.IGNORECASE):
                    marker_types.add("roman")

            if len(marker_types) >= 2:
                flat_list_count += 1
                if len(examples) < 3:
                    markers = "+".join(sorted(marker_types))
                    examples.append(f"p{page.get('page_number','?')}:{markers}")

    if flat_list_count > 0:
        concerns.append(AuditConcern(
            description=(
                f"Mixed numbering types at same list level ({flat_list_count} list(s)): "
                f"{', '.join(examples)} — possible sub-point flattening"
            ),
            source="programmatic",
            severity="minor",
            check_id="S5",
            analysis=(
                f"flat_lists={flat_list_count}. "
                "Lists with mixed numbering types (numeric+letter or numeric+roman) "
                "at the same depth level suggest sub-points were flattened."
            ),
        ))


def _detect_list_numbering_issues(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect list numbering restarts, sub-list flattening, and numbered items as headings.

    Pattern 1: Consecutive ordered lists where the second restarts numbering at 1.
    Pattern 2: Consecutive paragraphs with numbered text (list missed by extraction).
    Pattern 3: Heading elements whose text looks like numbered list items.
    """
    restart_count = 0
    para_list_count = 0
    heading_list_count = 0
    examples: list[str] = []

    def _extract_leading_number(text: str) -> int | None:
        """Extract the leading number from a list item or paragraph."""
        m = re.match(r"^\s*(\d+)\s*[.)]\s", text)
        return int(m.group(1)) if m else None

    # Track the last ordered list's ending number across pages for cross-page
    # restart detection. This persists between pages.
    cross_page_last_num: int | None = None
    cross_page_last_page: str | int | None = None

    for page in extraction_data.get("pages") or []:
        content = page.get("content") or []
        page_num = page.get("page_number", "?")

        # --- Pattern 1: Consecutive ordered lists with numbering restart ---
        # Also detects restarts across pages with intervening content.
        prev_list_last_num = None
        page_had_ordered_list = False
        page_last_ordered_num: int | None = None
        for item in content:
            if item.get("type") == "list" and item.get("list_type") == "ordered":
                items = item.get("items") or []
                if not items:
                    prev_list_last_num = None
                    continue
                # Get first item number
                first_text = items[0].get("text", "") if isinstance(items[0], dict) else str(items[0])
                first_num = _extract_leading_number(first_text)
                # Get last item number
                last_text = items[-1].get("text", "") if isinstance(items[-1], dict) else str(items[-1])
                last_num = _extract_leading_number(last_text)

                # Check for restart: consecutive on same page
                if (
                    first_num is not None
                    and first_num == 1
                    and prev_list_last_num is not None
                    and prev_list_last_num > 1
                ):
                    restart_count += 1
                    if len(examples) < 5:
                        examples.append(
                            f"p{page_num}: list restarts at 1 after ending at {prev_list_last_num}"
                        )
                # Check for restart: across pages with intervening content
                elif (
                    first_num is not None
                    and first_num == 1
                    and prev_list_last_num is None
                    and cross_page_last_num is not None
                    and cross_page_last_num > 1
                    and not page_had_ordered_list
                ):
                    restart_count += 1
                    if len(examples) < 5:
                        examples.append(
                            f"p{page_num}: list restarts at 1 (prev page {cross_page_last_page} ended at {cross_page_last_num})"
                        )

                prev_list_last_num = last_num
                page_had_ordered_list = True
                if last_num is not None:
                    page_last_ordered_num = last_num
            else:
                # Non-list element breaks the consecutive chain on this page
                prev_list_last_num = None

        # Update cross-page tracking
        if page_last_ordered_num is not None:
            cross_page_last_num = page_last_ordered_num
            cross_page_last_page = page_num
        # Don't reset cross_page if page had no lists — the restart could span
        # multiple pages of non-list content

        # --- Pattern 2: Consecutive paragraphs with numbered text ---
        numbered_run = 0
        run_restarts = 0
        prev_para_num = None
        for item in content:
            if item.get("type") == "paragraph":
                text = (item.get("text") or "").strip()
                num = _extract_leading_number(text)
                if num is not None:
                    numbered_run += 1
                    if prev_para_num is not None and num <= prev_para_num and num == 1:
                        run_restarts += 1
                    prev_para_num = num
                    continue
            # Non-match or different type — check if we had a run
            if numbered_run >= 3 and run_restarts > 0:
                para_list_count += run_restarts
                if len(examples) < 5:
                    examples.append(
                        f"p{page_num}: {numbered_run} numbered paragraphs with {run_restarts} restart(s)"
                    )
            numbered_run = 0
            run_restarts = 0
            prev_para_num = None
        # End-of-page flush
        if numbered_run >= 3 and run_restarts > 0:
            para_list_count += run_restarts
            if len(examples) < 5:
                examples.append(
                    f"p{page_num}: {numbered_run} numbered paragraphs with {run_restarts} restart(s)"
                )

        # --- Pattern 3: Heading elements that look like numbered list items ---
        for item in content:
            if item.get("type") == "heading":
                text = (item.get("text") or "").strip()
                # "1. Item text" or "3) Item text" as a heading is suspicious
                if re.match(r"^\d+\s*[.)]\s+\S", text):
                    heading_list_count += 1
                    if len(examples) < 5:
                        examples.append(f"p{page_num}: heading \"{text[:40]}\"")

        # --- Pattern 4: Single-item ordered lists (fragmented list) ---
        single_item_lists = 0
        for item in content:
            if (
                item.get("type") == "list"
                and item.get("list_type") == "ordered"
                and len(item.get("items") or []) == 1
            ):
                single_item_lists += 1
        if single_item_lists >= 2:
            restart_count += single_item_lists - 1
            if len(examples) < 5:
                examples.append(
                    f"p{page_num}: {single_item_lists} single-item ordered lists (fragmented)"
                )

    # Emit concerns
    total_issues = restart_count + para_list_count + heading_list_count
    if restart_count > 0:
        concerns.append(AuditConcern(
            description=(
                f"List numbering restart(s): {restart_count} ordered list(s) restart "
                f"numbering at 1 instead of continuing sequence"
            ),
            source="programmatic",
            severity="major",
            check_id="S5",
            analysis=(
                f"restart_count={restart_count}. "
                f"Examples: {'; '.join(examples[:3])}. "
                "Ordered lists that restart numbering suggest content was split "
                "across page breaks and not reassembled."
            ),
        ))

    if para_list_count > 0:
        concerns.append(AuditConcern(
            description=(
                f"Numbered paragraphs: {para_list_count} numbering restart(s) in "
                f"paragraph sequences that should be ordered lists"
            ),
            source="programmatic",
            severity="minor",
            check_id="S5",
            analysis=(
                f"para_list_restarts={para_list_count}. "
                "Consecutive paragraphs with numbered text (1. 2. 3.) suggest "
                "the extraction missed list structure."
            ),
        ))

    if heading_list_count >= 3:
        concerns.append(AuditConcern(
            description=(
                f"Numbered items as headings: {heading_list_count} heading(s) "
                f"look like numbered list items"
            ),
            source="programmatic",
            severity="minor",
            check_id="S5",
            analysis=(
                f"heading_list_count={heading_list_count}. "
                "Headings starting with '1.', '2.' etc. suggest numbered list "
                "items were misclassified as headings during extraction."
            ),
        ))


def _detect_flattened_sublists(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect nested list items flattened into text instead of children arrays.

    When a PDF has sub-items (a, b, c or bullets) under a parent item,
    the extraction may embed them as newline-delimited text within the
    parent's text field rather than structuring them as a children array.
    This loses semantic nesting in HTML output.
    """
    # Patterns that indicate sub-items embedded in text
    _SUBLIST_PATTERNS = [
        re.compile(r"\n\s*[a-z]\)\s", re.IGNORECASE),    # a) b) c)
        re.compile(r"\n\s*[a-z]\.\s", re.IGNORECASE),    # a. b. c.
        re.compile(r"\n\s*\d+\)\s"),                       # 1) 2) 3)
        re.compile(r"\n\s*[-•–]\s"),                       # - or • bullets
    ]

    hit_count = 0
    examples: list[str] = []

    for page in extraction_data.get("pages") or []:
        page_num = page.get("page_number", "?")
        for item in page.get("content") or []:
            itype = item.get("type")
            texts_to_check: list[tuple[str, str]] = []

            if itype == "list":
                for li in item.get("items") or []:
                    if not isinstance(li, dict):
                        continue
                    # Skip items that already have children — they're properly nested
                    if li.get("children"):
                        continue
                    text = li.get("text", "")
                    if text:
                        texts_to_check.append(("list", text))
            elif itype == "table":
                # Check both formats
                rows = item.get("rows") or []
                if rows:
                    for row in rows:
                        for cell in row.get("cells") or []:
                            text = cell.get("text", "") if isinstance(cell, dict) else str(cell)
                            if text:
                                texts_to_check.append(("table", text))
                else:
                    for cell in item.get("cells") or []:
                        text = cell.get("text", "") if isinstance(cell, dict) else str(cell)
                        if text:
                            texts_to_check.append(("table", text))

            for source, text in texts_to_check:
                if len(text) < 20:
                    continue
                for pat in _SUBLIST_PATTERNS:
                    matches = pat.findall(text)
                    if len(matches) >= 2:
                        hit_count += 1
                        if len(examples) < 3:
                            # Extract the first few sub-items for the example
                            lines = text.split("\n")
                            sub_items = [
                                ln.strip()[:40]
                                for ln in lines[1:]
                                if ln.strip() and pat.match("\n" + ln)
                            ][:3]
                            examples.append(
                                f"p{page_num} {source}: {', '.join(sub_items)}"
                            )
                        break  # One pattern match per text block is enough

    if hit_count >= 2:
        concerns.append(AuditConcern(
            description=(
                f"Flattened sub-lists: {hit_count} item(s) contain nested "
                f"sub-items (a/b/c or bullets) as plain text instead of "
                f"structured children"
            ),
            source="programmatic",
            severity="minor",
            check_id="S5",
            analysis=(
                f"flattened_sublists={hit_count}. "
                f"Examples: {'; '.join(examples)}. "
                "Sub-items embedded as newline-delimited text lose semantic "
                "nesting in HTML — screen readers can't navigate the hierarchy."
            ),
        ))


def _detect_form_elements(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect form elements that lose interactivity in HTML conversion.

    PDF forms with fillable fields, checkboxes, or signature lines cannot
    be faithfully represented in static HTML. Documents with significant
    form content should be flagged for review.
    """
    form_count = 0
    total_fields = 0
    for page in extraction_data.get("pages") or []:
        for item in page.get("content") or []:
            if item.get("type") == "form":
                form_count += 1
                total_fields += len(item.get("fields") or [])

    if form_count > 0:
        concerns.append(AuditConcern(
            description=(
                f"Form elements detected: {form_count} form(s) with "
                f"{total_fields} field(s) — forms lose functionality in HTML"
            ),
            source="programmatic",
            severity="major",
            analysis=(
                f"form_count={form_count}, total_fields={total_fields}. "
                "PDF form fields (checkboxes, text inputs, signatures) cannot "
                "be faithfully rendered as static HTML."
            ),
        ))


def _detect_table_structure_issues(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect structural issues in tables: inconsistent columns, etc.

    NOTE: This detector is DISABLED (returns immediately) because it has a
    62.5% false positive rate on clean documents.  Normal colspan/rowspan
    variation in well-formed tables triggers the column-count check, producing
    noise that misleads the Final Decider.  The LLM S3 fidelity check already
    handles table structure validation far more reliably because it can see
    the actual rendered output and compare against the source PDF.
    """
    # Disabled — LLM S3 check is sufficient.  See Iteration 10 research:
    # table_structure fired on 5/8 auto_approve docs in DHHS 100 while
    # LLM structural scores were 1.0 on all of them.
    return


# Markdown-in-text patterns: **bold**, *italic*, __bold__, _italic_
# Matches markdown emphasis that should have been rendered as HTML tags.
# Excludes: standalone asterisks (bullets), single * at line start, footnote refs
_MARKDOWN_BOLD_PATTERN = re.compile(r"\*\*[^*\n]{2,60}\*\*")
_MARKDOWN_ITALIC_PATTERN = re.compile(r"(?<!\*)\*(?!\*)[^*\n]{2,60}\*(?!\*)")


def _detect_markdown_as_text(
    extraction_data: dict,
    html_content: str | None,
    concerns: list[AuditConcern],
) -> None:
    """Detect unrendered Gemini markdown left as literal text.

    When Gemini outputs **bold** or *italic* markdown syntax and the pipeline
    doesn't convert it to <strong>/<em> tags, users see literal asterisks.
    Human reviewers flagged this on ~30+ docs as the most common issue.
    """
    bold_count = 0
    italic_count = 0
    examples: list[str] = []

    for page in extraction_data.get("pages") or []:
        for item in page.get("content") or []:
            texts_to_scan: list[str] = []
            item_type = item.get("type", "")

            if item_type in ("paragraph", "heading"):
                texts_to_scan.append(item.get("text") or "")
            elif item_type == "list":
                for li in item.get("items") or []:
                    if isinstance(li, dict):
                        texts_to_scan.append(li.get("text") or "")
                    elif isinstance(li, str):
                        texts_to_scan.append(li)

            for text in texts_to_scan:
                for m in _MARKDOWN_BOLD_PATTERN.finditer(text):
                    bold_count += 1
                    if len(examples) < 3:
                        examples.append(m.group(0)[:40])
                for m in _MARKDOWN_ITALIC_PATTERN.finditer(text):
                    italic_count += 1
                    if len(examples) < 3:
                        examples.append(m.group(0)[:40])

    total = bold_count + italic_count
    if total >= 3:
        concerns.append(AuditConcern(
            description=(
                f"Unrendered markdown formatting: {bold_count} bold (**text**) "
                f"and {italic_count} italic (*text*) instances — "
                f"e.g. {'; '.join(examples)}"
            ),
            source="programmatic",
            severity="minor",
            check_id="V5",
            analysis=(
                f"markdown_bold={bold_count}, markdown_italic={italic_count}. "
                "Gemini markdown syntax left as literal asterisks instead of "
                "being converted to HTML <strong>/<em> tags."
            ),
        ))


def _detect_image_alt_as_text(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect [Image: description] alt text leaked into visible paragraph text.

    A pipeline rendering bug where image alt text placeholders appear as
    visible <p> elements (e.g., [Image: NCGICC logo]). This is always a bug
    — legitimate content never contains [Image: ...] syntax.
    """
    instances: list[str] = []

    for page in extraction_data.get("pages") or []:
        for item in page.get("content") or []:
            if item.get("type") not in ("paragraph", "heading"):
                continue
            text = (item.get("text") or "").strip()
            if re.search(r"\[Image:\s*[^\]]+\]", text):
                instances.append(text[:60])

    if instances:
        examples = instances[:3]
        concerns.append(AuditConcern(
            description=(
                f"Image alt text leaked as visible text: {len(instances)} "
                f"instance(s) of [Image: ...] in paragraphs — "
                f"e.g. {'; '.join(examples)}"
            ),
            source="programmatic",
            severity="major",
            analysis=(
                f"image_alt_as_text_count={len(instances)}. "
                "Alt text placeholder syntax appearing as visible paragraph "
                "text indicates a rendering bug."
            ),
        ))


def _detect_link_displacement(
    extraction_data: dict,
    concerns: list[AuditConcern],
) -> None:
    """Detect links displaced from inline to end of page/section.

    In many docs, inline hyperlinks from the source PDF are extracted and
    placed at the bottom of the page as standalone link elements instead of
    being inline in the paragraph text. Human reviewers flag this as
    "links pulled to bottom of page" or "duplicated at bottom."
    """
    pages_with_displaced_links = 0
    total_displaced = 0

    for page in extraction_data.get("pages") or []:
        content = page.get("content") or []
        if len(content) < 3:
            continue

        # Find the last non-link content index
        last_non_link_idx = -1
        link_count_at_end = 0
        for i in range(len(content) - 1, -1, -1):
            if content[i].get("type") == "link":
                link_count_at_end += 1
            else:
                last_non_link_idx = i
                break

        # Links clustered at the end of a page (2+ links after all content)
        if link_count_at_end >= 2 and last_non_link_idx >= 0:
            # Verify these aren't just normal link elements — check if the
            # links' URLs also appear in the text content above (displaced copy)
            link_urls: set[str] = set()
            for i in range(last_non_link_idx + 1, len(content)):
                item = content[i]
                if item.get("type") == "link":
                    url = (item.get("url") or item.get("text") or "").strip().lower()
                    if url:
                        link_urls.add(url)

            if link_urls:
                pages_with_displaced_links += 1
                total_displaced += link_count_at_end

    if pages_with_displaced_links >= 2:
        severity = "major" if pages_with_displaced_links >= 3 else "minor"
        concerns.append(AuditConcern(
            description=(
                f"Links displaced to end of page: {total_displaced} link(s) "
                f"clustered at bottom of {pages_with_displaced_links} page(s) "
                f"instead of inline in text"
            ),
            source="programmatic",
            severity=severity,
            check_id="S7",
            analysis=(
                f"displaced_link_pages={pages_with_displaced_links}, "
                f"total_displaced={total_displaced}. "
                "Links appear as standalone elements at the end of pages "
                "rather than being inline in the paragraph text."
            ),
        ))


# ============================================================================
# Agreement Assessment & Concern Deduplication
# ============================================================================

def _assess_agreement(
    signals: SignalBreakdown,
    concerns: list[AuditConcern],
) -> str:
    if not signals.fidelity_available:
        return "unknown"

    # Always run corroboration first — it must happen before severity
    # downgrading so that corroborated concerns keep their severity.
    _mark_corroboration(concerns)

    # When fidelity overall approved the doc, downgrade non-corroborated
    # major fidelity concerns to minor — BUT only for content-level concerns.
    # Structural/visual concerns (S2/S4/S5/V4/V5) should keep their severity
    # because these affect accessibility navigation and are frequently
    # underrated by the fidelity evaluator.
    KEEP_SEVERITY_CHECK_IDS = {"S2", "S4", "S5", "S7", "V4", "V5"}
    fidelity_approved = signals.fidelity_routing == "AUTO_APPROVE"
    if fidelity_approved:
        for c in concerns:
            if (
                c.source == "fidelity_llm"
                and c.severity == "major"
                and not c.corroborated
                and c.check_id not in KEEP_SEVERITY_CHECK_IDS
            ):
                c.severity = "minor"

    llm_concerns = [c for c in concerns if c.source == "fidelity_llm"]
    prog_concerns = [c for c in concerns if c.source == "programmatic"]

    llm_significant = [c for c in llm_concerns if c.severity in ("major", "critical")]
    llm_has_significant = len(llm_significant) > 0
    llm_has_minor_only = len(llm_concerns) > 0 and not llm_has_significant

    prog_significant = [c for c in prog_concerns if c.severity in ("major", "critical")]
    prog_has_significant = len(prog_significant) > 0
    prog_has_minor_only = len(prog_concerns) > 0 and not prog_has_significant

    if llm_has_significant and prog_has_significant:
        return "converge"

    if not llm_has_significant and not prog_has_significant:
        if llm_has_minor_only or prog_has_minor_only:
            return "converge_with_minors"
        return "converge"

    llm_critical = any(c.severity == "critical" for c in llm_concerns)
    prog_critical = any(c.severity == "critical" for c in prog_concerns)
    if llm_critical or prog_critical:
        return "major_disagree"

    return "partial_disagree"


def _mark_corroboration(concerns: list[AuditConcern]) -> None:
    llm_check_ids = {c.check_id for c in concerns if c.source == "fidelity_llm" and c.check_id}

    # Hallucination: C1 + low precision
    has_c1 = "C1" in llm_check_ids
    has_low_precision = any(
        c.source == "programmatic" and "precision" in c.description.lower()
        for c in concerns
    )
    if has_c1 and has_low_precision:
        for c in concerns:
            if (c.check_id == "C1") or (c.source == "programmatic" and "precision" in c.description.lower()):
                c.corroborated = True

    # Truncation: C3/C5 + low recall
    has_truncation = bool({"C3", "C5"} & llm_check_ids)
    has_low_recall = any(
        c.source == "programmatic" and "recall" in c.description.lower()
        for c in concerns
    )
    if has_truncation and has_low_recall:
        for c in concerns:
            if c.check_id in ("C3", "C5") or (c.source == "programmatic" and "recall" in c.description.lower()):
                c.corroborated = True

    # Duplication: S1 + high duplication ratio
    has_s1 = "S1" in llm_check_ids
    has_duplication = any(
        c.source == "programmatic" and "duplication" in c.description.lower()
        for c in concerns
    )
    if has_s1 and has_duplication:
        for c in concerns:
            if c.check_id == "S1" or (c.source == "programmatic" and "duplication" in c.description.lower()):
                c.corroborated = True

    # Table truncation: C4 + low table count ratio
    has_c4 = "C4" in llm_check_ids
    has_low_table_ratio = any(
        c.source == "programmatic" and "table" in c.description.lower()
        for c in concerns
    )
    if has_c4 and has_low_table_ratio:
        for c in concerns:
            if c.check_id == "C4" or (c.source == "programmatic" and "table" in c.description.lower()):
                c.corroborated = True

    # Image placeholder: V2 + high placeholder ratio (V2 = images missing/broken)
    has_v2 = "V2" in llm_check_ids
    has_placeholder_concern = any(
        c.source == "programmatic" and "placeholder" in c.description.lower()
        for c in concerns
    )
    if has_v2 and has_placeholder_concern:
        for c in concerns:
            if c.check_id == "V2" or (c.source == "programmatic" and "placeholder" in c.description.lower()):
                c.corroborated = True

    # Fabrication: C1 + per-element fabrication
    has_fabrication = any(
        c.source == "programmatic" and "fabricated" in c.description.lower()
        for c in concerns
    )
    if has_c1 and has_fabrication:
        for c in concerns:
            if c.check_id == "C1" or (c.source == "programmatic" and "fabricated" in c.description.lower()):
                c.corroborated = True

    # Link loss: S7 + C5/C3 (missing links = missing content)
    has_s7 = "S7" in {c.check_id for c in concerns if c.check_id}
    has_missing_content = bool({"C3", "C5"} & llm_check_ids)
    if has_s7 and has_missing_content:
        for c in concerns:
            if c.check_id == "S7" or c.check_id in ("C3", "C5"):
                c.corroborated = True

    # Link loss: S7 + bare URL detection (both programmatic)
    has_bare_urls = any(
        c.source == "programmatic" and "bare url" in c.description.lower()
        for c in concerns
    )
    if has_s7 and has_bare_urls:
        for c in concerns:
            if c.check_id == "S7" or (c.source == "programmatic" and "bare url" in c.description.lower()):
                c.corroborated = True


_CROSS_DIMENSION_EQUIVALENCES: list[tuple[str, str]] = [
    ("S1", "V3"),   # content duplication — V4 renumbered to V3
    ("S3", "V1"),   # table structure — V2 renumbered to V1
]
# Cross-call check equivalences — same root cause appearing in two different calls.
# When both fire, keep the higher-severity one and mark as corroborated.
_CROSS_CALL_EQUIVALENCES: list[tuple[str, str]] = [
    ("C5", "S5"),  # Missing list content: C5 is text omission, S5 is list structure
]
_LLM_PROGRAMMATIC_EQUIVALENCES: list[tuple[str, str]] = [
    ("C1", "precision"),
    ("S1", "duplication"),
    ("V2", "placeholder"),  # V2 = images missing/broken (was V3 before renumber)
]
_SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}


def _check_id_to_call_type(check_id: str) -> EvaluationCallType:
    """Map check ID prefix to its call type."""
    prefix = check_id[0]
    return {
        "C": EvaluationCallType.CONTENT,
        "S": EvaluationCallType.STRUCTURAL,
        "V": EvaluationCallType.VISUAL,
    }[prefix]


def _recompute_composite(fidelity_report: FidelityReport) -> None:
    """Recompute composite score from individual call scores."""
    weighted_sum = 0.0
    weight_sum = 0.0
    for call_type, weight in CALL_WEIGHTS.items():
        call_result = getattr(fidelity_report, f"{call_type.value}", None)
        if call_result and call_result.evaluation_status == "complete":
            weighted_sum += call_result.score * weight
            weight_sum += weight
    if weight_sum > 0:
        fidelity_report.composite_score = weighted_sum / weight_sum


def _adjust_composite_for_dedup(
    fidelity_report: FidelityReport,
    removed_checks: list[tuple[str, DefectSeverity]],
) -> None:
    """Retroactively remove double-counted penalties from composite score."""
    if not removed_checks or fidelity_report is None:
        return
    for check_id, severity in removed_checks:
        try:
            call_type = _check_id_to_call_type(check_id)
        except KeyError:
            continue
        call_result = getattr(fidelity_report, call_type.value, None)
        if call_result and call_result.evaluation_status == "complete":
            penalty = SEVERITY_WEIGHTS[severity]
            call_result.penalty_sum = max(0, call_result.penalty_sum - penalty)
            normalization = NORMALIZATION_FACTORS[call_type]
            call_result.score = max(0.0, 1.0 - call_result.penalty_sum / normalization)
            call_result.defect_count = max(0, call_result.defect_count - 1)
            if severity == DefectSeverity.CRITICAL:
                call_result.critical_count = max(0, call_result.critical_count - 1)
            elif severity == DefectSeverity.MAJOR:
                call_result.major_count = max(0, call_result.major_count - 1)
            else:
                call_result.minor_count = max(0, call_result.minor_count - 1)
    _recompute_composite(fidelity_report)


def _deduplicate_concerns(
    concerns: list[AuditConcern],
    signals: SignalBreakdown,
) -> tuple[list[AuditConcern], list[tuple[str, DefectSeverity]]]:
    if len(concerns) <= 1:
        return list(concerns), []

    by_check_id: dict[str, list[int]] = {}
    for i, c in enumerate(concerns):
        if c.check_id and c.source == "fidelity_llm":
            by_check_id.setdefault(c.check_id.strip().upper(), []).append(i)

    prog_indices: list[int] = [
        i for i, c in enumerate(concerns) if c.source == "programmatic"
    ]
    remove: set[int] = set()

    for check_a, check_b in _CROSS_DIMENSION_EQUIVALENCES:
        indices_a = by_check_id.get(check_a, [])
        indices_b = by_check_id.get(check_b, [])
        if not indices_a or not indices_b:
            continue
        best_a = min(indices_a, key=lambda i: _SEVERITY_ORDER.get(concerns[i].severity, 99))
        best_b = min(indices_b, key=lambda i: _SEVERITY_ORDER.get(concerns[i].severity, 99))
        sev_a = _SEVERITY_ORDER.get(concerns[best_a].severity, 99)
        sev_b = _SEVERITY_ORDER.get(concerns[best_b].severity, 99)
        if sev_a <= sev_b:
            winner, loser = best_a, best_b
        else:
            winner, loser = best_b, best_a
        concerns[winner].corroborated = True
        remove.add(loser)

    # Cross-call equivalences (e.g., C5 + S5 for same missing content)
    for check_a, check_b in _CROSS_CALL_EQUIVALENCES:
        indices_a = by_check_id.get(check_a, [])
        indices_b = by_check_id.get(check_b, [])
        if not indices_a or not indices_b:
            continue
        best_a = min(indices_a, key=lambda i: _SEVERITY_ORDER.get(concerns[i].severity, 99))
        best_b = min(indices_b, key=lambda i: _SEVERITY_ORDER.get(concerns[i].severity, 99))
        sev_a = _SEVERITY_ORDER.get(concerns[best_a].severity, 99)
        sev_b = _SEVERITY_ORDER.get(concerns[best_b].severity, 99)
        if sev_a <= sev_b:
            winner, loser = best_a, best_b
        else:
            winner, loser = best_b, best_a
        concerns[winner].corroborated = True
        remove.add(loser)

    for check_id, keyword in _LLM_PROGRAMMATIC_EQUIVALENCES:
        llm_indices = by_check_id.get(check_id, [])
        if not llm_indices:
            continue
        matching_prog = [
            i for i in prog_indices
            if keyword in concerns[i].description.lower() and i not in remove
        ]
        if not matching_prog:
            continue
        best_llm = min(
            (i for i in llm_indices if i not in remove),
            key=lambda i: _SEVERITY_ORDER.get(concerns[i].severity, 99),
            default=None,
        )
        if best_llm is None:
            continue
        best_prog = min(
            matching_prog,
            key=lambda i: _SEVERITY_ORDER.get(concerns[i].severity, 99),
        )
        sev_llm = _SEVERITY_ORDER.get(concerns[best_llm].severity, 99)
        sev_prog = _SEVERITY_ORDER.get(concerns[best_prog].severity, 99)
        if sev_llm <= sev_prog:
            winner, loser = best_llm, best_prog
        else:
            winner, loser = best_prog, best_llm
        concerns[winner].corroborated = True
        remove.add(loser)

    # Build list of removed check IDs + severities for composite adjustment
    removed_checks: list[tuple[str, DefectSeverity]] = []
    for i in remove:
        c = concerns[i]
        if c.check_id and c.source == "fidelity_llm":
            sev = _CHECK_SEVERITY_MAP.get(c.check_id.strip().upper())
            if sev is not None:
                removed_checks.append((c.check_id.strip().upper(), sev))

    deduped = [c for i, c in enumerate(concerns) if i not in remove]
    signals.deduplicated_count = len(remove)
    return deduped, removed_checks


# ============================================================================
# Hard Vetoes — 8 deterministic rules
# ============================================================================

class HardVetoResult:
    __slots__ = ("fired", "rule_id", "routing", "reasoning")

    def __init__(
        self,
        fired: bool = False,
        rule_id: str = "",
        routing: AuditRouting = AuditRouting.HUMAN_REVIEW,
        reasoning: str = "",
    ):
        self.fired = fired
        self.rule_id = rule_id
        self.routing = routing
        self.reasoning = reasoning


def check_hard_vetoes(signals: SignalBreakdown) -> HardVetoResult:
    # LLM-only vetoes: only fidelity-based checks trigger hard vetoes.
    # Programmatic vetoes V2 (source PDF missing), V3 (word count loss),
    # V4 (fabrication), V5 (axe-core) removed.

    # V1: Pipeline failed
    if not signals.pipeline_success:
        error_detail = signals.pipeline_error or "unknown error"
        if len(error_detail) > 500:
            error_detail = error_detail[:500] + "…[truncated]"
        return HardVetoResult(
            fired=True, rule_id="V1", routing=AuditRouting.REJECT,
            reasoning=f"Pipeline failed: {error_detail}",
        )

    # V6: Fidelity auto-reject
    if signals.fidelity_routing == "AUTO_REJECT":
        return HardVetoResult(
            fired=True, rule_id="V6", routing=AuditRouting.REJECT,
            reasoning=f"LLM fidelity routed to AUTO_REJECT (composite={signals.fidelity_composite or 0:.2f}).",
        )

    # V7: Fidelity composite critically low
    if (
        signals.fidelity_available
        and signals.fidelity_composite is not None
        and signals.fidelity_composite <= 0.40
    ):
        dimensions_evaluated = sum(
            1 for score in (signals.fidelity_content, signals.fidelity_structural, signals.fidelity_visual)
            if score is not None
        )
        if dimensions_evaluated >= 2:
            return HardVetoResult(
                fired=True, rule_id="V7", routing=AuditRouting.HUMAN_REVIEW,
                reasoning=f"Fidelity composite critically low: {signals.fidelity_composite:.2f} ({dimensions_evaluated}/3 dims). Manual review required.",
            )

    # V8: Fidelity evaluation failure
    if (
        signals.fidelity_available
        and signals.fidelity_composite is not None
        and signals.fidelity_composite == 0.0
    ):
        dimensions_evaluated = sum(
            1 for score in (signals.fidelity_content, signals.fidelity_structural, signals.fidelity_visual)
            if score is not None
        )
        if dimensions_evaluated == 0:
            return HardVetoResult(
                fired=True, rule_id="V8", routing=AuditRouting.HUMAN_REVIEW,
                reasoning="Fidelity evaluation failure: all 3 calls failed. Manual review required.",
            )

    return HardVetoResult(fired=False)


# ============================================================================
# Fidelity Scoring — 3 LLM calls with MQM penalty scoring
# ============================================================================

def _get_check_name_map() -> dict[str, str]:
    return {
        "C1": "Fabricated paragraphs/sentences",
        "C2": "Fabricated contact information",
        "C3": "Complete section/page missing",
        "C4": "Table with significantly fewer rows",
        "C5": "Significant text omission within section",
        "C6": "Minor text differences beyond reformatting",
        "S1": "Content duplicated",
        "S2": "Heading hierarchy broken",
        "S3": "Table structure corrupted",
        "S4": "Reading order significantly different",
        "S5": "Lists converted incorrectly",
        "S6": "Footnotes not preserved",
        "S7": "Link injection issues",
        "V1": "Tables visually broken or misrendered",
        "V2": "Images missing or broken",
        "V3": "Content visually duplicated on page",
        "V4": "Images placed next to wrong text",
        "V5": "Text formatting not preserved",
    }


def _compute_mqm_score(
    checks: list[FidelityDefect], call_type: EvaluationCallType
) -> tuple[float, int]:
    penalty = sum(
        SEVERITY_WEIGHTS[d.d_severity]
        for d in checks
        if d.c_verdict == "FAIL" and d.d_severity is not None
    )
    norm = NORMALIZATION_FACTORS[call_type]
    score = max(0.0, 1.0 - penalty / norm)
    return score, penalty


def _compute_composite(
    available: dict[EvaluationCallType, CallResult],
) -> tuple[float, str | None]:
    if not available:
        return 0.0, "all_calls_failed"

    all_defects: list[FidelityDefect] = []
    for result in available.values():
        all_defects.extend(d for d in result.checks if d.c_verdict == "FAIL")

    # Knockout Rule 1: Critical veto
    criticals = [d for d in all_defects if d.d_severity == DefectSeverity.CRITICAL]
    if criticals:
        composite = _weighted_mean(available)
        return min(composite, 0.40), "critical_veto"

    # Knockout Rule 2: Dimension floor
    for result in available.values():
        if result.score == 0.0:
            composite = _weighted_mean(available)
            return min(composite, 0.25), "dimension_zero"

    # Knockout Rule 3: Content evaluation missing
    if EvaluationCallType.CONTENT not in available:
        composite = _weighted_mean(available)
        return min(composite, 0.60), "content_evaluation_missing"

    # Knockout Rule 4: Partial evaluation
    all_call_types = {EvaluationCallType.CONTENT, EvaluationCallType.STRUCTURAL, EvaluationCallType.VISUAL}
    missing = all_call_types - set(available.keys())
    if len(available) == 2 and missing:
        missing_name = ", ".join(ct.value for ct in missing)
        composite = _weighted_mean(available)
        return min(composite, 0.75), f"partial_evaluation:{missing_name}"

    # Knockout Rule 5: Insufficient evaluation
    if len(available) < 2:
        composite = _weighted_mean(available)
        return min(composite, 0.50), "insufficient_evaluation"

    return _weighted_mean(available), None


def _weighted_mean(available: dict[EvaluationCallType, CallResult]) -> float:
    total_weight = sum(CALL_WEIGHTS[k] for k in available)
    if total_weight == 0:
        return 0.0
    return sum(CALL_WEIGHTS[k] * available[k].score for k in available) / total_weight


def _route_fidelity(
    composite: float, knockout: str | None
) -> tuple[str, str, str]:
    if knockout:
        reason = f"Knockout rule: {knockout}"
        if knockout == "critical_veto":
            return "HIGH", "FLAG_FOR_REVIEW", reason
        if knockout == "dimension_zero":
            return "CRITICAL", "FLAG_FOR_REVIEW", reason
        if knockout == "all_calls_failed":
            return "CRITICAL", "FLAG_FOR_REVIEW", "Fidelity scoring unavailable"
        if knockout == "content_evaluation_missing":
            return "HIGH", "FLAG_FOR_REVIEW", reason
        if knockout.startswith("partial_evaluation:"):
            return "MEDIUM", "FLAG_FOR_REVIEW", reason
        if knockout == "insufficient_evaluation":
            return "HIGH", "MANDATORY_REVIEW", reason
        return "CRITICAL", "FLAG_FOR_REVIEW", reason

    if composite >= 0.85:
        return "LOW", "AUTO_APPROVE", f"Score {composite:.2f} >= 0.85"
    if composite >= 0.60:
        return "MEDIUM", "FLAG_FOR_REVIEW", f"Score {composite:.2f} in review range"
    if composite >= 0.35:
        return "HIGH", "MANDATORY_REVIEW", f"Score {composite:.2f} requires thorough review"
    return "CRITICAL", "AUTO_REJECT", f"Score {composite:.2f} < 0.35, reprocess"


def _parse_json_response(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def _validate_fidelity_response(response: dict, expected_check_ids: list[str]) -> bool:
    if "checks" not in response:
        return False
    response_ids = {c.get("check_id") for c in response["checks"]}
    if not set(expected_check_ids).issubset(response_ids):
        return False
    for check in response["checks"]:
        verdict = check.get("c_verdict")
        if verdict not in ("PASS", "FAIL"):
            return False
        if verdict == "FAIL":
            severity = check.get("d_severity")
            if severity not in ("MINOR", "MAJOR", "CRITICAL"):
                return False
            if not check.get("b_evidence_source") or not check.get("b_evidence_output"):
                return False
    return True


def _build_call_result(
    call_type: EvaluationCallType,
    parsed: dict,
    raw_text: str,
) -> CallResult:
    check_name_map = _get_check_name_map()
    checks: list[FidelityDefect] = []
    for check_data in parsed["checks"]:
        check_id = check_data["check_id"]
        severity = None
        if check_data["c_verdict"] == "FAIL" and check_data.get("d_severity"):
            llm_severity = DefectSeverity(check_data["d_severity"].lower())
            defined_severity = _CHECK_SEVERITY_MAP.get(check_id)
            if defined_severity and llm_severity != defined_severity:
                logger.info(
                    "Severity enforcement: %s LLM returned %s, using defined %s",
                    check_id, llm_severity.value, defined_severity.value,
                )
            severity = defined_severity or llm_severity
        checks.append(
            FidelityDefect(
                check_id=check_id,
                check_name=check_data.get("check_name", check_name_map.get(check_id, check_id)),
                a_analysis=check_data.get("a_analysis", ""),
                b_evidence_source=check_data.get("b_evidence_source", ""),
                b_evidence_output=check_data.get("b_evidence_output", ""),
                b_evidence_location=check_data.get("b_evidence_location", ""),
                c_verdict=check_data["c_verdict"],
                d_severity=severity,
            )
        )
    score, penalty_sum = _compute_mqm_score(checks, call_type)
    defects = [c for c in checks if c.c_verdict == "FAIL"]
    return CallResult(
        call_type=call_type,
        checks=checks,
        score=score,
        penalty_sum=penalty_sum,
        defect_count=len(defects),
        critical_count=sum(1 for d in defects if d.d_severity == DefectSeverity.CRITICAL),
        major_count=sum(1 for d in defects if d.d_severity == DefectSeverity.MAJOR),
        minor_count=sum(1 for d in defects if d.d_severity == DefectSeverity.MINOR),
        raw_response=raw_text,
    )


def _build_unknown_result(call_type: EvaluationCallType) -> CallResult:
    return CallResult(
        call_type=call_type,
        checks=[],
        score=0.0,
        penalty_sum=0,
        defect_count=0,
        critical_count=0,
        major_count=0,
        minor_count=0,
        raw_response=None,
        evaluation_status="failed",
    )


def _is_429_error(exc: Exception) -> bool:
    """Check if an exception is a 429 rate limit error."""
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "rate limit" in msg


def _call_gemini_fidelity(
    prompt: str,
    pdf_bytes: bytes,
    model: str | None = None,
) -> str:
    """Make a fidelity evaluation Gemini call (PDF + text prompt).

    Uses client rotation — on 429, marks the key as rate-limited
    and raises so the outer retry loop can try the next key.
    """
    client = _get_client()
    model = model or FIDELITY_MODEL

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=1.0,
        max_output_tokens=FIDELITY_MAX_OUTPUT_TOKENS,
        safety_settings=_get_safety_settings(),
        thinking_config=types.ThinkingConfig(
            thinking_budget=8192,
        ),
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                prompt,
            ],
            config=config,
        )
    except Exception as e:
        if _is_429_error(e):
            _mark_key_rate_limited(client)
        raise

    # Extract text from response
    if response.candidates and response.candidates[0].content:
        parts = response.candidates[0].content.parts
        if parts:
            text_parts = [p.text for p in parts if isinstance(getattr(p, "text", None), str)]
            result = "".join(text_parts)
            if result:
                return result

    raise ValueError("Empty Gemini response for fidelity evaluation")


_VALID_CLASSIFICATIONS = {"STANDARD", "FORM", "SINGLE_GRAPHIC", "SLIDE_DECK"}


def _extract_document_classification(raw_response: str | None) -> tuple[str | None, str | None]:
    """Extract document_classification from Call 1 raw response JSON."""
    if not raw_response:
        return None, None
    try:
        # Strip code fences if Gemini wraps the JSON
        text = raw_response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        data = json.loads(text)
        classification = data.get("document_classification", "STANDARD")
        rationale = data.get("document_classification_rationale")
        if isinstance(classification, str):
            classification = classification.strip().upper()
            if classification in _VALID_CLASSIFICATIONS:
                return classification, rationale
        return "STANDARD", rationale
    except (json.JSONDecodeError, AttributeError):
        return None, None


def run_fidelity_scoring(
    extraction_json_str: str,
    pdf_bytes: bytes,
    rendered_html: str | None = None,
    document_id: str | None = None,
) -> FidelityReport:
    """Run all 3 fidelity evaluation calls and produce a FidelityReport.

    Runs calls sequentially (simpler than async for standalone script).
    """
    # Truncate extraction JSON for large docs
    if len(extraction_json_str) > 2_000_000:
        extraction_json_str = extraction_json_str[:2_000_000] + "\n... [truncated for token budget]"

    call_results: dict[EvaluationCallType, CallResult] = {}

    # Call 1: Content Fidelity (also classifies document type)
    call_results[EvaluationCallType.CONTENT] = _run_single_fidelity_call(
        EvaluationCallType.CONTENT,
        _build_content_fidelity_prompt(extraction_json_str),
        pdf_bytes,
        CONTENT_CHECK_IDS,
        "Content",
    )

    # Check for document exclusion — short-circuit if not STANDARD
    doc_classification, doc_rationale = _extract_document_classification(
        call_results[EvaluationCallType.CONTENT].raw_response
    )
    if doc_classification and doc_classification != "STANDARD":
        logger.info("Document classified as %s — skipping structural/visual calls", doc_classification)
        content_result = call_results[EvaluationCallType.CONTENT]
        return FidelityReport(
            document_id=document_id,
            composite_score=0.0,
            risk_level="excluded",
            routing_action="EXCLUDED",
            routing_reason=f"Document classified as {doc_classification}: {doc_rationale or 'no rationale'}",
            content_fidelity=content_result,
            total_defects=content_result.defect_count,
            total_critical=content_result.critical_count,
            total_major=content_result.major_count,
            total_minor=content_result.minor_count,
            document_classification=doc_classification,
            document_classification_rationale=doc_rationale,
        )

    # Call 2: Structural Fidelity
    call_results[EvaluationCallType.STRUCTURAL] = _run_single_fidelity_call(
        EvaluationCallType.STRUCTURAL,
        _build_structural_fidelity_prompt(extraction_json_str),
        pdf_bytes,
        STRUCTURAL_CHECK_IDS,
        "Structural",
    )

    # Call 3: Visual Fidelity — strip base64 from HTML to stay within token budget
    visual_html = _strip_base64_from_html(rendered_html) if rendered_html else None
    call_results[EvaluationCallType.VISUAL] = _run_single_fidelity_call(
        EvaluationCallType.VISUAL,
        _build_visual_fidelity_prompt(visual_html),
        pdf_bytes,
        VISUAL_CHECK_IDS,
        "Visual",
    )

    # Build report
    available = {
        k: v for k, v in call_results.items()
        if v.evaluation_status == "complete"
    }
    composite, knockout = _compute_composite(available)
    risk_level, routing_action, routing_reason = _route_fidelity(composite, knockout)

    total_defects = sum(r.defect_count for r in available.values())
    total_critical = sum(r.critical_count for r in available.values())
    total_major = sum(r.major_count for r in available.values())
    total_minor = sum(r.minor_count for r in available.values())

    return FidelityReport(
        document_id=document_id,
        composite_score=round(composite, 4),
        risk_level=risk_level,
        routing_action=routing_action,
        routing_reason=routing_reason,
        knockout_triggered=knockout,
        content_fidelity=call_results.get(EvaluationCallType.CONTENT),
        structural_fidelity=call_results.get(EvaluationCallType.STRUCTURAL),
        visual_fidelity=call_results.get(EvaluationCallType.VISUAL),
        total_defects=total_defects,
        total_critical=total_critical,
        total_major=total_major,
        total_minor=total_minor,
        document_classification=doc_classification or "STANDARD",
        document_classification_rationale=doc_rationale,
    )


def _run_single_fidelity_call(
    call_type: EvaluationCallType,
    prompt: str,
    pdf_bytes: bytes,
    expected_ids: list[str],
    label: str,
) -> CallResult:
    """Run a single fidelity evaluation call with retry and validation."""
    for attempt in range(1 + FIDELITY_MAX_RETRIES):
        try:
            raw_text = _call_gemini_fidelity(prompt, pdf_bytes)
            parsed = _parse_json_response(raw_text)
            if not _validate_fidelity_response(parsed, expected_ids):
                if attempt < FIDELITY_MAX_RETRIES:
                    logger.warning(
                        "Fidelity %s: response validation failed, retrying (%d/%d)",
                        label, attempt + 1, FIDELITY_MAX_RETRIES,
                    )
                    time.sleep(min(2 ** attempt * 1.0, 30.0) + random.uniform(0, 1))
                    continue
                logger.error("Fidelity %s: validation failed after retries", label)
                return _build_unknown_result(call_type)
            return _build_call_result(call_type, parsed, raw_text)
        except json.JSONDecodeError as e:
            if attempt < FIDELITY_MAX_RETRIES:
                logger.warning("Fidelity %s: JSON parse failed: %s, retrying", label, sanitize_error(str(e)))
                time.sleep(min(2 ** attempt * 1.0, 30.0) + random.uniform(0, 1))
                continue
            logger.error("Fidelity %s: JSON parse failed after retries", label)
            return _build_unknown_result(call_type)
        except Exception as e:
            if attempt < FIDELITY_MAX_RETRIES:
                logger.warning("Fidelity %s: attempt %d failed: %s, retrying", label, attempt + 1, sanitize_error(str(e)))
                time.sleep(min(2 ** attempt * 1.0, 30.0) + random.uniform(0, 1))
                continue
            logger.error("Fidelity %s: failed after retries: %s", label, sanitize_error(str(e)))
            return _build_unknown_result(call_type)

    return _build_unknown_result(call_type)


# ============================================================================
# Final Decider — optional 4th LLM call
# ============================================================================

def _call_gemini_decider(prompt: str) -> str:
    """Make the Final Decider Gemini call."""
    client = _get_client()
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=1.0,
        max_output_tokens=DECIDER_MAX_OUTPUT_TOKENS,
        safety_settings=_get_safety_settings(),
        thinking_config=types.ThinkingConfig(
            thinking_budget=4096,
        ),
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        if _is_429_error(e):
            _mark_key_rate_limited(client)
        raise
    if response.candidates and response.candidates[0].content:
        parts = response.candidates[0].content.parts
        if parts:
            text_parts = [p.text for p in parts if isinstance(getattr(p, "text", None), str)]
            result = "".join(text_parts)
            if result:
                return result
    raise ValueError("Empty Gemini response for Final Decider")


class DeciderResult:
    __slots__ = (
        "routing", "confidence", "reasoning",
        "critical_findings", "major_findings", "minor_findings",
        "signal_agreement", "action_items", "raw_response",
    )

    def __init__(
        self,
        routing: AuditRouting = AuditRouting.HUMAN_REVIEW,
        confidence: float = 0.5,
        reasoning: str = "",
        critical_findings: list[str] | None = None,
        major_findings: list[str] | None = None,
        minor_findings: list[str] | None = None,
        signal_agreement: str = "unknown",
        action_items: list[str] | None = None,
        raw_response: str | None = None,
    ):
        self.routing = routing
        self.confidence = confidence
        self.reasoning = reasoning
        self.critical_findings = critical_findings or []
        self.major_findings = major_findings or []
        self.minor_findings = minor_findings or []
        self.signal_agreement = signal_agreement
        self.action_items = action_items or []
        self.raw_response = raw_response


def run_final_decider(
    signals: SignalBreakdown,
    concerns: list[AuditConcern],
    document_id: str = "unknown",
) -> DeciderResult:
    """Run the Final Decider LLM call."""
    prompt = _build_final_decider_prompt(signals, concerns, document_id)

    for attempt in range(1 + DECIDER_MAX_RETRIES):
        try:
            raw_text = _call_gemini_decider(prompt)
            return _parse_decider_response(raw_text)
        except json.JSONDecodeError as e:
            if attempt < DECIDER_MAX_RETRIES:
                logger.warning("Final Decider: JSON parse failed: %s, retrying", sanitize_error(str(e)))
                time.sleep(min(2 ** attempt * 1.0, 15.0) + random.uniform(0, 1))
                continue
            return _fallback_decider_result("JSON parse failure after retries")
        except Exception as e:
            if attempt < DECIDER_MAX_RETRIES:
                logger.warning("Final Decider: attempt %d failed: %s, retrying", attempt + 1, sanitize_error(str(e)))
                time.sleep(min(2 ** attempt * 1.0, 15.0) + random.uniform(0, 1))
                continue
            return _fallback_decider_result(sanitize_error(f"LLM call failed: {e}"))

    return _fallback_decider_result("Exhausted retries")


def _parse_decider_response(raw_text: str) -> DeciderResult:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    data = json.loads(text.strip())

    def _get(prefixed: str, unprefixed: str, default=None):
        return data.get(prefixed, data.get(unprefixed, default))

    routing_str = (_get("c_routing", "routing") or "HUMAN_REVIEW")
    if not isinstance(routing_str, str):
        routing_str = "HUMAN_REVIEW"
    routing_str = routing_str.upper().replace(" ", "_")
    routing_map = {
        "AUTO_APPROVE": AuditRouting.AUTO_APPROVE,
        "HUMAN_REVIEW": AuditRouting.HUMAN_REVIEW,
        "FLAG_FOR_REVIEW": AuditRouting.HUMAN_REVIEW,
        "REJECT": AuditRouting.REJECT,
    }
    routing = routing_map.get(routing_str, AuditRouting.HUMAN_REVIEW)

    raw_confidence = _get("d_confidence", "confidence")
    try:
        confidence = max(0.0, min(1.0, float(raw_confidence or 0.5)))
    except (TypeError, ValueError):
        confidence = 0.5

    return DeciderResult(
        routing=routing,
        confidence=confidence,
        reasoning=_get("a_reasoning", "reasoning") or "",
        critical_findings=_get("e_critical_findings", "critical_findings") or [],
        major_findings=_get("f_major_findings", "major_findings") or [],
        minor_findings=_get("g_minor_findings", "minor_findings") or [],
        signal_agreement=_get("b_signal_agreement", "signal_agreement") or "unknown",
        action_items=_get("h_action_items", "action_items") or [],
        raw_response=raw_text,
    )


def _fallback_decider_result(reason: str) -> DeciderResult:
    return DeciderResult(
        routing=AuditRouting.HUMAN_REVIEW,
        confidence=0.5,
        reasoning=f"Final Decider unavailable ({reason}). Defaulting to HUMAN_REVIEW.",
        action_items=["Final Decider failed — manual review recommended."],
    )


# ============================================================================
# Quality Score Computation
# ============================================================================

_COMPONENT_WEIGHTS = {
    "fidelity_composite": 0.40,
    "text_match": 0.20,
    "content_completeness": 0.15,
    "accessibility": 0.10,
    "decision_confidence": 0.15,
}


def _text_match_score(signals: SignalBreakdown) -> float | None:
    recall = signals.shingling_recall
    precision = signals.shingling_precision
    if recall is None and precision is None:
        return None
    if recall is not None and precision is not None:
        if (recall + precision) > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0.0
    elif recall is not None:
        f1 = recall
    else:
        f1 = precision
    dup_penalty = min((signals.duplication_ratio or 0.0) * 0.66, 0.20)
    return max(0.0, f1 - dup_penalty)


def _completeness_score(signals: SignalBreakdown) -> float | None:
    wcr = signals.word_count_ratio
    tcr = signals.table_count_ratio
    if wcr is None and tcr is None:
        return None
    scores: list[float] = []
    if wcr is not None:
        scores.append(max(0.0, 1.0 - min(abs(wcr - 1.0), 1.0)))
    if tcr is not None and tcr < FABRICATED_TABLE_SENTINEL:
        scores.append(max(0.0, 1.0 - min(abs(tcr - 1.0), 1.0)))
    return sum(scores) / len(scores) if scores else None


# Severity weights for axe-core impact levels
_AXE_SEVERITY_WEIGHTS = {
    "critical": 10,
    "serious": 5,
    "moderate": 2,
    "minor": 1,
}


def _axe_accessibility_score(violations: list[dict]) -> float:
    """Compute accessibility score from axe-core violations using a decay function.

    score = 1 / (1 + weighted_sum / K) where K=20.
    Multiplies by nodes_count per rule — one color-contrast rule
    affecting 50 elements is much worse than affecting 1.

    Returns 1.0 for no violations, approaches 0.0 for many severe violations.
    """
    weighted = sum(
        _AXE_SEVERITY_WEIGHTS.get(v.get("impact", "minor"), 1)
        * v.get("nodes_count", 1)
        for v in violations
    )
    return 1.0 / (1.0 + weighted / 20.0)


def _build_axe_compliance(raw_axe_output: dict, scan_duration_ms: int) -> dict:
    """Build the axe_compliance dict for AuditReport from raw axe-core output.

    This is informational only — does NOT affect scoring or routing.
    """
    violations = raw_axe_output.get("violations", [])

    critical = sum(v.get("nodes_count", 1) for v in violations if v.get("impact") == "critical")
    serious = sum(v.get("nodes_count", 1) for v in violations if v.get("impact") == "serious")
    moderate = sum(v.get("nodes_count", 1) for v in violations if v.get("impact") == "moderate")
    minor = sum(v.get("nodes_count", 1) for v in violations if v.get("impact") == "minor")

    return {
        "available": True,
        "score": round(_axe_accessibility_score(violations), 4),
        "total_violations": len(violations),
        "total_affected_nodes": critical + serious + moderate + minor,
        "by_severity": {
            "critical": critical,
            "serious": serious,
            "moderate": moderate,
            "minor": minor,
        },
        "violations": [
            {
                "rule_id": v.get("id", "unknown"),
                "impact": v.get("impact", "unknown"),
                "description": v.get("description", ""),
                "help_url": v.get("help_url", ""),
                "nodes_count": v.get("nodes_count", 1),
                "wcag_tags": v.get("tags", []),
            }
            for v in violations
        ],
        "wcag_level": "WCAG 2.1 AA",
        "axe_version": raw_axe_output.get("axe_version"),
        "scan_duration_ms": scan_duration_ms,
        "error": None,
    }


def _decider_signal_score(routing: AuditRouting, decider_confidence: float) -> float:
    if routing == AuditRouting.AUTO_APPROVE:
        return decider_confidence
    elif routing == AuditRouting.REJECT:
        return max(0.0, 1.0 - decider_confidence)
    else:
        return 0.5


def _apply_knockout_ceiling(
    raw_score: float, signals: SignalBreakdown,
) -> tuple[float, str | None]:
    baseline_words = signals.baseline_words or 0

    _fidelity_eval_failed = (
        signals.fidelity_available
        and signals.fidelity_composite is not None
        and signals.fidelity_composite == 0.0
        and signals.fidelity_content is None
        and signals.fidelity_structural is None
        and signals.fidelity_visual is None
    )

    # LLM-only ceilings: only fidelity-based checks influence scoring.
    # Programmatic ceilings (extreme_word_loss, extreme_hallucination) removed.
    checks: list[tuple[bool, float, str]] = [
        (not signals.pipeline_success, 0.15, "pipeline_failed"),
        (_fidelity_eval_failed, 0.60, "fidelity_evaluation_failed"),
        (signals.fidelity_routing == "AUTO_REJECT", 0.30, "fidelity_auto_reject"),
        (
            signals.fidelity_composite is not None
            and signals.fidelity_composite <= 0.40,
            0.40, "fidelity_critical_veto",
        ),
        (
            signals.fidelity_composite is not None
            and signals.fidelity_composite <= 0.60,
            0.60, "fidelity_low_score",
        ),
        (not signals.fidelity_available, 0.75, "fidelity_unavailable"),
    ]

    for condition, ceiling, reason in checks:
        if condition:
            capped = min(raw_score, ceiling)
            return capped, reason

    return raw_score, None


def compute_quality_score(
    signals: SignalBreakdown,
    routing: AuditRouting,
    decider_confidence: float,
) -> tuple[int, ScoreBreakdown]:
    # LLM-only scoring: quality score = fidelity composite x 100
    # Programmatic signals (shingling, word count, axe) are collected for
    # informational reporting but do NOT influence scoring or routing.
    if signals.fidelity_available and signals.fidelity_composite is not None:
        raw_score = signals.fidelity_composite
    else:
        raw_score = 0.50  # no fidelity data available

    capped_score, ceiling_reason = _apply_knockout_ceiling(raw_score, signals)
    quality_score = max(0, min(100, round(capped_score * 100)))

    breakdown = ScoreBreakdown(
        fidelity_composite=signals.fidelity_composite if signals.fidelity_available else None,
        raw_score=round(raw_score, 4),
        ceiling=round(capped_score, 4) if ceiling_reason else None,
        ceiling_reason=ceiling_reason,
        available_components=1 if signals.fidelity_available else 0,
    )

    return quality_score, breakdown


# ============================================================================
# Routing Escalation Guard
# ============================================================================

_ROUTING_SEVERITY = {
    AuditRouting.AUTO_APPROVE: 0,
    AuditRouting.HUMAN_REVIEW: 1,
    AuditRouting.REJECT: 2,
    AuditRouting.EXCLUDED: -1,  # not comparable — exclusion is a separate track
}

_FIDELITY_ROUTING_MAP = {
    "AUTO_APPROVE": AuditRouting.AUTO_APPROVE,
    "FLAG_FOR_REVIEW": AuditRouting.AUTO_APPROVE,
    "MANDATORY_REVIEW": AuditRouting.HUMAN_REVIEW,
    "AUTO_REJECT": AuditRouting.REJECT,
}


def _infer_agreement(concerns: list[AuditConcern]) -> str:
    sources = {c.source for c in concerns}
    if not concerns:
        return "converge"
    if len(sources) >= 2:
        has_corroborated = any(c.corroborated for c in concerns)
        return "converge" if has_corroborated else "partial_disagree"
    return "partial_disagree"


# ============================================================================
# Programmatic Decision (skip_llm mode)
# ============================================================================

def _programmatic_decision(
    signals: SignalBreakdown,
    concerns: list[AuditConcern],
) -> tuple[AuditRouting, float, str, list[str]]:
    """Make a routing decision from programmatic signals only (no LLM)."""
    # Filter to only fidelity_llm and axe_core concerns (not programmatic)
    # for the final decider substitute
    fidelity_concerns = [c for c in concerns if c.source in ("fidelity_llm", "axe_core")]
    critical_concerns = [c for c in fidelity_concerns if c.severity == "critical"]
    major_concerns = [c for c in fidelity_concerns if c.severity == "major"]

    if critical_concerns:
        reasons = "; ".join(c.description for c in critical_concerns[:3])
        return (
            AuditRouting.REJECT, 0.8,
            f"Programmatic-only: {len(critical_concerns)} critical concern(s): {reasons}",
            [c.description for c in critical_concerns],
        )

    if major_concerns:
        reasons = "; ".join(c.description for c in major_concerns[:3])
        return (
            AuditRouting.HUMAN_REVIEW, 0.7,
            f"Programmatic-only: {len(major_concerns)} major concern(s): {reasons}",
            [c.description for c in major_concerns],
        )

    if (
        signals.fidelity_available
        and signals.fidelity_composite is not None
        and signals.fidelity_composite >= 0.85
        and signals.fidelity_routing == "AUTO_APPROVE"
    ):
        minor_count = len([c for c in concerns if c.severity == "minor"])
        qualifier = f" ({minor_count} minor concern(s) noted)" if minor_count else ""
        return (
            AuditRouting.AUTO_APPROVE, 0.7,
            f"Programmatic-only: no major/critical concerns, fidelity approved{qualifier}.",
            [],
        )

    # Fidelity unavailable but programmatic signals are excellent: auto-approve
    # when shingling and word count both show near-perfect text preservation
    # and there are no major/critical concerns from any source.
    if (
        not signals.fidelity_available
        and signals.shingling_recall is not None
        and signals.shingling_recall >= 0.90
        and signals.shingling_precision is not None
        and signals.shingling_precision >= 0.90
        and (signals.word_count_ratio or 0) >= 0.90
        and (signals.baseline_words or 0) >= 50
    ):
        all_major_critical = [
            c for c in concerns if c.severity in ("major", "critical")
        ]
        if not all_major_critical:
            minor_count = len([c for c in concerns if c.severity == "minor"])
            qualifier = (
                f" ({minor_count} minor concern(s) noted)" if minor_count else ""
            )
            return (
                AuditRouting.AUTO_APPROVE, 0.65,
                f"Programmatic-only: fidelity unavailable but programmatic signals "
                f"excellent (shingling={signals.shingling_recall:.3f}, "
                f"word_ratio={signals.word_count_ratio:.3f}){qualifier}.",
                [],
            )

    return (
        AuditRouting.HUMAN_REVIEW, 0.5,
        "Programmatic-only: no severe issues found but Final Decider not available.",
        ["Run with LLM enabled for higher-confidence routing."],
    )


# ============================================================================
# Main Audit Orchestrator
# ============================================================================

def _strip_base64_for_prompt(data: dict) -> dict:
    """Strip base64 image data from extraction JSON for LLM prompt inclusion.

    The source PDF is attached separately to fidelity calls, so embedding
    base64 images in the prompt wastes tokens and causes context overflow
    on image-heavy documents (e.g., 26MB JSON → 1M+ tokens).
    Preserves all other fields (type, description, bbox, format, caption).
    """
    import copy
    stripped = copy.deepcopy(data)
    for page in stripped.get("pages") or []:
        for item in page.get("content") or []:
            if item.get("type") == "image" and "base64_data" in item:
                b64_len = len(item["base64_data"])
                item["base64_data"] = f"[{b64_len} chars stripped for token budget]"
    return stripped


_BASE64_IMG_RE = re.compile(
    r'src="data:image/[^;]+;base64,[A-Za-z0-9+/=]+"',
)


def _strip_base64_from_html(html: str) -> str:
    """Strip base64 image data from HTML to reduce token count.

    Replaces data:image/...;base64,... src attributes with a placeholder.
    The source PDF is attached separately so the LLM can still evaluate images.
    """
    return _BASE64_IMG_RE.sub('src="[base64 image stripped for token budget]"', html)


def audit_document(
    extraction_json_path: Path,
    pdf_path: Path | None = None,
    html_path: Path | None = None,
    skip_llm: bool = False,
    skip_decider: bool = False,
    skip_axe: bool = False,
    document_id: str | None = None,
) -> AuditReport:
    """Run a comprehensive audit on one document.

    Args:
        extraction_json_path: Path to extraction-test JSON file.
        pdf_path: Path to source PDF (for shingling + fidelity).
        html_path: Path to rendered HTML (for visual fidelity).
        skip_llm: Skip all LLM calls (fidelity + decider).
        skip_decider: Skip only the Final Decider (still run fidelity).
        skip_axe: Skip axe-core WCAG accessibility scan.
        document_id: Optional document identifier.

    Returns:
        AuditReport with routing decision, reasoning, and signals.
    """
    start_ms = time.monotonic_ns() // 1_000_000
    doc_id = document_id or extraction_json_path.stem

    # Source file tracking
    _source_files = {
        "extraction_json": str(extraction_json_path),
        "source_pdf": str(pdf_path) if pdf_path else None,
        "rendered_html": str(html_path) if html_path else None,
    }

    # Load extraction JSON
    extraction_data = json.loads(extraction_json_path.read_text(encoding="utf-8"))
    # Strip base64 image data for LLM prompts — images are in the attached PDF
    extraction_json_str = json.dumps(_strip_base64_for_prompt(extraction_data), indent=2)

    # Load PDF bytes
    pdf_bytes: bytes | None = None
    if pdf_path and pdf_path.exists():
        pdf_bytes = pdf_path.read_bytes()

    # Load rendered HTML
    rendered_html: str | None = None
    if html_path and html_path.exists():
        rendered_html = html_path.read_text(encoding="utf-8")

    # Step 1: Run fidelity scoring (3 LLM calls)
    fidelity_report: FidelityReport | None = None
    if not skip_llm and pdf_bytes is not None:
        print(f"  FIDELITY  {doc_id}: Running 3 fidelity evaluation calls...")
        try:
            fidelity_report = run_fidelity_scoring(
                extraction_json_str=extraction_json_str,
                pdf_bytes=pdf_bytes,
                rendered_html=rendered_html,
                document_id=doc_id,
            )
            print(
                f"  FIDELITY  {doc_id}: composite={fidelity_report.composite_score:.2f}, "
                f"routing={fidelity_report.routing_action}, "
                f"defects={fidelity_report.total_defects} "
                f"(C={fidelity_report.total_critical}, M={fidelity_report.total_major}, m={fidelity_report.total_minor})"
            )
        except Exception as e:
            print(f"  FIDELITY  {doc_id}: FAILED — {sanitize_error(str(e))}", file=sys.stderr)

    # Step 1b: Check for document exclusion
    if fidelity_report and fidelity_report.document_classification != "STANDARD":
        elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms
        classification = fidelity_report.document_classification
        rationale = fidelity_report.document_classification_rationale or ""
        print(f"  EXCLUDED  {doc_id}: {classification} — {rationale}")
        return _build_audit_report(
            doc_id=doc_id,
            routing=AuditRouting.EXCLUDED,
            decider_confidence=0.95,
            reasoning=f"Document excluded: {classification}. {rationale}",
            decision_method="document_classification",
            concerns=[],
            action_items=[],
            signals=SignalBreakdown(pipeline_success=True),
            processing_ms=elapsed_ms,
            quality_score=0,
            score_breakdown=ScoreBreakdown(raw_score=0.0, available_components=0),
            fidelity_report=fidelity_report,
            source_files=_source_files,
            exclusion_reason=classification,
        )

    # Step 2: Collect signals
    signals, concerns = collect_signals(
        extraction_data=extraction_data,
        pdf_path=pdf_path,
        fidelity_report=fidelity_report,
        html_content=rendered_html,
    )

    # Step 2b: Adjust composite score for dedup
    if fidelity_report and hasattr(signals, 'dedup_removed_checks') and signals.dedup_removed_checks:
        _adjust_composite_for_dedup(fidelity_report, signals.dedup_removed_checks)

    # Step 2c: Run axe-core accessibility scan (informational only — does NOT affect scoring/routing)
    axe_compliance_dict: dict | None = None
    if not skip_axe and rendered_html is not None and html_path is not None:
        print(f"  AXE-CORE  {doc_id}: Running WCAG 2.1 AA scan...")
        axe_start_ms = time.monotonic_ns() // 1_000_000
        raw_axe = _run_axe_core(html_path)
        axe_elapsed_ms = (time.monotonic_ns() // 1_000_000) - axe_start_ms
        if raw_axe is not None:
            axe_compliance_dict = _build_axe_compliance(raw_axe, scan_duration_ms=axe_elapsed_ms)
            # Populate SignalBreakdown fields (informational only)
            signals.axe_available = True
            signals.axe_violations = axe_compliance_dict["total_violations"]
            signals.axe_critical = axe_compliance_dict["by_severity"]["critical"]
            signals.axe_serious = axe_compliance_dict["by_severity"]["serious"]
            signals.axe_moderate = axe_compliance_dict["by_severity"]["moderate"]
            signals.axe_minor = axe_compliance_dict["by_severity"]["minor"]
            print(
                f"  AXE-CORE  {doc_id}: score={axe_compliance_dict['score']:.2f}, "
                f"violations={axe_compliance_dict['total_violations']} "
                f"(C={signals.axe_critical}, S={signals.axe_serious}, "
                f"M={signals.axe_moderate}, m={signals.axe_minor})"
            )
        else:
            print(f"  AXE-CORE  {doc_id}: skipped (Node.js or axe-runner not available)")

    # Step 3: Check hard vetoes
    veto = check_hard_vetoes(signals)
    if veto.fired:
        elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms
        score, score_breakdown = compute_quality_score(signals, veto.routing, 0.95)
        report = _build_audit_report(
            doc_id=doc_id,
            routing=veto.routing,
            decider_confidence=0.95,
            reasoning=veto.reasoning,
            decision_method=f"hard_veto:{veto.rule_id}",
            concerns=concerns,
            action_items=[veto.reasoning],
            signals=signals,
            processing_ms=elapsed_ms,
            quality_score=score,
            score_breakdown=score_breakdown,
            fidelity_report=fidelity_report,
            source_files=_source_files,
            axe_compliance=axe_compliance_dict,
        )
        return report

    # Step 4: Final Decider
    if skip_llm or skip_decider:
        routing, confidence, reasoning, action_items = _programmatic_decision(signals, concerns)
        decision_method = "programmatic_only"
        signals.signal_agreement = _infer_agreement(concerns)
    else:
        print(f"  DECIDER   {doc_id}: Running Final Decider call...")
        result = run_final_decider(signals, concerns, doc_id)
        routing = result.routing
        confidence = result.confidence
        reasoning = result.reasoning
        action_items = result.action_items
        decision_method = "llm_final_decider" if result.raw_response else "fallback"

    # Step 5: Compute quality score BEFORE escalation guard
    score, score_breakdown = compute_quality_score(signals, routing, confidence)

    # Step 6: Escalation guard
    routing_escalated_from = None
    if signals.fidelity_routing:
        fidelity_audit_routing = _FIDELITY_ROUTING_MAP.get(
            signals.fidelity_routing, AuditRouting.HUMAN_REVIEW
        )
        if _ROUTING_SEVERITY.get(routing, 0) < _ROUTING_SEVERITY.get(fidelity_audit_routing, 0):
            routing_escalated_from = routing.value
            routing = fidelity_audit_routing
            reasoning += (
                f" [Escalation guard: routing elevated from {routing_escalated_from}"
                f" to {routing.value} because fidelity routing was {signals.fidelity_routing}]"
            )

    elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms

    # Step 7: Build report
    report = _build_audit_report(
        doc_id=doc_id,
        routing=routing,
        decider_confidence=confidence,
        reasoning=reasoning,
        decision_method=decision_method,
        concerns=concerns,
        action_items=action_items,
        signals=signals,
        processing_ms=elapsed_ms,
        quality_score=score,
        score_breakdown=score_breakdown,
        routing_escalated_from=routing_escalated_from,
        fidelity_report=fidelity_report,
        source_files=_source_files,
        axe_compliance=axe_compliance_dict,
    )

    return report


def _build_audit_report(
    doc_id: str,
    routing: AuditRouting,
    decider_confidence: float,
    reasoning: str,
    decision_method: str,
    concerns: list[AuditConcern],
    action_items: list[str],
    signals: SignalBreakdown,
    processing_ms: int,
    quality_score: int | None = None,
    score_breakdown: ScoreBreakdown | None = None,
    routing_escalated_from: str | None = None,
    fidelity_report: FidelityReport | None = None,
    source_files: dict[str, str | None] | None = None,
    exclusion_reason: str | None = None,
    axe_compliance: dict | None = None,
) -> AuditReport:
    severity_order = {"critical": 0, "major": 1, "minor": 2}
    sorted_concerns = sorted(concerns, key=lambda c: severity_order.get(c.severity, 3))

    if quality_score is None:
        quality_score, score_breakdown = compute_quality_score(
            signals, routing, decider_confidence,
        )

    return AuditReport(
        document_id=doc_id,
        audited_at=datetime.now(UTC),
        source_files=source_files or {},
        quality_score=quality_score,
        quality_grade=quality_grade(quality_score),
        fidelity_composite=signals.fidelity_composite if signals.fidelity_available else None,
        fidelity_content=signals.fidelity_content if signals.fidelity_available else None,
        fidelity_structural=signals.fidelity_structural if signals.fidelity_available else None,
        fidelity_visual=signals.fidelity_visual if signals.fidelity_available else None,
        routing=routing,
        routing_label=_ROUTING_LABELS.get(routing.value, routing.value),
        reasoning=reasoning,
        decision_method=decision_method,
        concerns=sorted_concerns,
        action_items=action_items,
        routing_changed=False,
        pipeline_original_routing=None,
        exclusion_reason=exclusion_reason,
        axe_compliance=axe_compliance,
        internal=InternalMetrics(
            decider_confidence=decider_confidence,
            score_breakdown=score_breakdown,
            signals=signals,
            processing_ms=processing_ms,
            fidelity_report_path="fidelity-report.json" if fidelity_report else None,
            routing_escalated_from=routing_escalated_from,
        ),
    )


# ============================================================================
# CLI
# ============================================================================

def _find_pdf_for_json(json_path: Path, pdf_dir: Path | None) -> Path | None:
    """Find the source PDF for a given JSON file."""
    # Try same directory as JSON
    same_dir = json_path.parent / "source.pdf"
    if same_dir.exists():
        return same_dir

    # Try pdf_dir with matching name
    if pdf_dir:
        stem = json_path.stem
        # Try pdf_dir/stem/source.pdf
        candidate = pdf_dir / stem / "source.pdf"
        if candidate.exists():
            return candidate
        # Try pdf_dir/stem.pdf
        candidate = pdf_dir / f"{stem}.pdf"
        if candidate.exists():
            return candidate

    return None


def _find_html_for_json(json_path: Path, html_dir: Path | None) -> Path | None:
    """Find the rendered HTML for a given JSON file."""
    # Try same directory as JSON
    same_dir = json_path.with_suffix(".html")
    if same_dir.exists():
        return same_dir

    # Try same folder with any .html file
    html_files = list(json_path.parent.glob("*.html"))
    if html_files:
        return html_files[0]

    # Try html_dir with matching name
    if html_dir:
        stem = json_path.stem
        candidate = html_dir / f"{stem}.html"
        if candidate.exists():
            return candidate

    return None


def process_one(
    json_path: Path,
    pdf_path: Path | None,
    html_path: Path | None,
    output_dir: Path | None,
    skip_llm: bool,
    skip_decider: bool,
    skip_axe: bool = False,
) -> dict:
    """Process a single document and return summary dict."""
    doc_id = json_path.stem
    try:
        start = time.time()
        audit_start_dt = datetime.now(UTC)  # for analytics
        report = audit_document(
            extraction_json_path=json_path,
            pdf_path=pdf_path,
            html_path=html_path,
            skip_llm=skip_llm,
            skip_decider=skip_decider,
            skip_axe=skip_axe,
            document_id=doc_id,
        )
        elapsed = time.time() - start
        audit_end_dt = datetime.now(UTC)  # for analytics

        # Write report
        out_dir = output_dir or json_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"{doc_id}-audit-report.json"
        report_path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )

        # Also write fidelity report if generated
        if report.internal.fidelity_report_path:
            # Fidelity report is embedded in the audit report signals
            pass

        # Optional analytics
        if _HAS_ANALYTICS:
            try:
                collector = _PipelineAnalyticsCollector(
                    tenant_id=TENANT_ID,
                    gemini_model=GEMINI_MODEL,
                )
                collector.record_stage(
                    document_id=doc_id,
                    stage_name="audit",
                    start_time=audit_start_dt,
                    end_time=audit_end_dt,
                )
                collector.record_audit(
                    document_id=doc_id,
                    audit_result={
                        "completeness_score": report.quality_score,
                        "is_complete": report.routing == AuditRouting.AUTO_APPROVE,
                        "missing_elements": report.concerns,
                    },
                    start_time=audit_start_dt,
                    end_time=audit_end_dt,
                )
                collector.flush()
            except Exception as _exc:
                logger.debug("Analytics failed for %s (non-fatal): %s", doc_id, _exc)

        return {
            "document_id": doc_id,
            "quality_score": report.quality_score,
            "quality_grade": report.quality_grade,
            "routing": report.routing.value,
            "routing_label": report.routing_label,
            "fidelity_composite": report.fidelity_composite,
            "decision_method": report.decision_method,
            "concerns": len(report.concerns),
            "elapsed_seconds": round(elapsed, 1),
            "report_path": str(report_path),
        }
    except Exception as e:
        return {"document_id": doc_id, "error": sanitize_error(str(e))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone quality auditor for the 3-step pipeline.\n"
                    "Reads extraction JSON + source PDF + rendered HTML,\n"
                    "runs fidelity scoring (3 LLM calls), and outputs audit-report.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", type=Path,
        help="Path to a .json file, a directory of .json files, or a parent "
             "directory with subdirectories containing source.pdf + .json + .html files",
    )
    parser.add_argument(
        "--pdf", type=Path, default=None,
        help="Path to source PDF (single file mode)",
    )
    parser.add_argument(
        "--html", type=Path, default=None,
        help="Path to rendered HTML (single file mode)",
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=None,
        help="Directory containing source PDFs (batch mode, looks for {stem}/source.pdf or {stem}.pdf)",
    )
    parser.add_argument(
        "--html-dir", type=Path, default=None,
        help="Directory containing rendered HTML files (batch mode, looks for {stem}.html)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output directory for audit reports (default: same as input)",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip all LLM calls (fidelity + decider). Programmatic signals only.",
    )
    parser.add_argument(
        "--skip-decider", action="store_true",
        help="Skip Final Decider (4th LLM call) but still run fidelity scoring.",
    )
    parser.add_argument(
        "--skip-axe", action="store_true",
        help="Skip axe-core WCAG accessibility scan.",
    )
    parser.add_argument(
        "--api-keys", type=str, default=None,
        help="Comma-separated Gemini API keys for rotation (Developer API mode). "
             "Also reads from GEMINI_API_KEYS env var. Enables --api-mode automatically.",
    )
    parser.add_argument(
        "--api-mode", action="store_true",
        help="Use Gemini Developer API with API keys instead of Vertex AI. "
             "Keys from --api-keys flag or GEMINI_API_KEYS env var.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override the Gemini model name (default: gemini-2.5-flash-preview-05-20)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=1,
        help="Number of parallel workers for batch mode (default: 1 = sequential). "
             "Set to number of API keys for max throughput (e.g., --workers 11).",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # Auto-load .env.local from project root (if not already in environment)
    _load_env_local()

    # Model override (CLI flag > env var > default)
    global GEMINI_MODEL, FIDELITY_MODEL
    model_override = args.model or os.environ.get("GEMINI_MODEL", "")
    if model_override:
        GEMINI_MODEL = model_override
        FIDELITY_MODEL = model_override
        print(f"  Model: {GEMINI_MODEL}")

    # API key initialization
    api_keys_str = args.api_keys or os.environ.get("GEMINI_API_KEYS", "")
    if api_keys_str or args.api_mode:
        keys = [k for k in api_keys_str.split(",") if k.strip()] if api_keys_str else []
        # Also check for individual GEMINI_API_KEY_1..N env vars
        if not keys:
            for i in range(1, 20):
                key = os.environ.get(f"GEMINI_API_KEY_{i}", "")
                if key:
                    keys.append(key)
        # Also check single GEMINI_API_KEY / GOOGLE_API_KEY
        if not keys:
            single_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
            if single_key:
                keys.append(single_key)
        if not keys:
            print("Error: --api-mode requires API keys via --api-keys, GEMINI_API_KEYS, "
                  "GEMINI_API_KEY_1..N, GEMINI_API_KEY, or GOOGLE_API_KEY env vars",
                  file=sys.stderr)
            sys.exit(1)
        _init_api_key_clients(keys)
    else:
        _init_vertex_client()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    # Single file mode
    if args.input.is_file():
        result = process_one(
            json_path=args.input,
            pdf_path=args.pdf,
            html_path=args.html or _find_html_for_json(args.input, args.html_dir),
            output_dir=args.output,
            skip_llm=args.skip_llm,
            skip_decider=args.skip_decider,
            skip_axe=args.skip_axe,
        )
        _print_result(result)
        return

    # Directory mode — detect layout
    input_dir = args.input

    # Check for subdirectory layout (json_to_html_to_auditor/ style)
    subdirs = [
        d for d in sorted(input_dir.iterdir())
        if d.is_dir() and (d / "source.pdf").exists()
    ]

    if subdirs:
        # Subdirectory layout: each subfolder has source.pdf + .json + .html
        print(f"Found {len(subdirs)} document folders in {input_dir}/")
        print(f"Workers: {args.workers}\n")

        # Build work items
        work_items = []
        for folder in subdirs:
            json_files_in_folder = list(folder.glob("*.json"))
            json_files_in_folder = [f for f in json_files_in_folder if "audit" not in f.stem.lower()]
            if not json_files_in_folder:
                print(f"  SKIP  {folder.name}: no extraction JSON found")
                continue
            json_path = json_files_in_folder[0]
            pdf_path = folder / "source.pdf"
            html_path = _find_html_for_json(json_path, None)
            work_items.append((json_path, pdf_path, html_path, args.output or folder))

        def _do_one(item):
            jp, pp, hp, od = item
            return process_one(
                json_path=jp, pdf_path=pp, html_path=hp,
                output_dir=od, skip_llm=args.skip_llm, skip_decider=args.skip_decider,
                skip_axe=args.skip_axe,
            )

        results = []
        if args.workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_do_one, item): item for item in work_items}
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    _print_result(result)
        else:
            for item in work_items:
                result = _do_one(item)
                results.append(result)
                _print_result(result)

        _print_summary(results)
        return

    # Flat directory: .json files in directory
    json_files = sorted(input_dir.glob("*.json"))
    json_files = [f for f in json_files if "audit" not in f.stem.lower()]

    if not json_files:
        # Try nested
        json_files = sorted(input_dir.glob("*/*.json"))
        json_files = [f for f in json_files if "audit" not in f.stem.lower()]

    if not json_files:
        print(f"No extraction JSON files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files in {input_dir}/")
    print(f"Workers: {args.workers}\n")

    work_items = []
    for jf in json_files:
        pdf_path = args.pdf or _find_pdf_for_json(jf, args.pdf_dir)
        html_path = args.html or _find_html_for_json(jf, args.html_dir)
        work_items.append((jf, pdf_path, html_path, args.output))

    def _do_one_flat(item):
        jp, pp, hp, od = item
        return process_one(
            json_path=jp, pdf_path=pp, html_path=hp,
            output_dir=od, skip_llm=args.skip_llm, skip_decider=args.skip_decider,
            skip_axe=args.skip_axe,
        )

    results = []
    if args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_do_one_flat, item): item for item in work_items}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                _print_result(result)
    else:
        for item in work_items:
            result = _do_one_flat(item)
            results.append(result)
            _print_result(result)

    _print_summary(results)


def _print_result(result: dict) -> None:
    if "error" in result:
        print(f"  ERROR  {result['document_id']}: {result['error']}")
    else:
        routing = result.get("routing", "unknown")
        score = result.get("quality_score", "?")
        grade = result.get("quality_grade", "?")
        fid = result.get("fidelity_composite")
        fid_str = f", fidelity={fid:.2f}" if fid is not None else ""
        method = result.get("decision_method", "?")
        elapsed = result.get("elapsed_seconds", "?")
        concerns_count = result.get("concerns", 0)

        status = ("PASS" if routing == "auto_approve"
                  else "FAIL" if routing == "reject"
                  else "EXCL" if routing == "excluded"
                  else "REVIEW")
        print(
            f"  {status:6s} {result['document_id']} "
            f"(score={score}/{grade}, routing={routing}{fid_str}, "
            f"method={method}, concerns={concerns_count}, {elapsed}s)"
        )


def _print_summary(results: list[dict]) -> None:
    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    approved = sum(1 for r in results if r.get("routing") == "auto_approve")
    review = sum(1 for r in results if r.get("routing") == "human_review")
    rejected = sum(1 for r in results if r.get("routing") == "reject")
    excluded = sum(1 for r in results if r.get("routing") == "excluded")

    scores = [r["quality_score"] for r in results if "quality_score" in r and r.get("routing") != "excluded"]
    avg_score = sum(scores) / len(scores) if scores else 0

    print(f"\n{'='*60}")
    print(f"Audit complete: {total} documents")
    print(f"  Auto-approve: {approved}")
    print(f"  Human review: {review}")
    print(f"  Reject:       {rejected}")
    print(f"  Excluded:     {excluded}")
    print(f"  Errors:       {errors}")
    if scores:
        print(f"  Avg score:    {avg_score:.0f}")


if __name__ == "__main__":
    main()

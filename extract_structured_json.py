"""
PDF Structured JSON Extraction Tool with Quality Verification

This script processes all PDFs in data/ folders, extracts structured JSON
(paragraphs, tables, images) using Gemini, and verifies extraction quality.

Usage:
    # Default (uses DATA_FOLDER constant or --data-dir):
    python extract_structured_json.py

    # Specify data directory:
    python extract_structured_json.py --data-dir /path/to/NCDIT-ADA-FILES/data/

    # API key mode (Developer API, no Vertex AI needed):
    python extract_structured_json.py --api-mode --data-dir /path/to/data/

    # Override model:
    python extract_structured_json.py --model gemini-3-flash-preview

Output:
    - Per-PDF JSON files: output/{pdf_id}.json
    - Aggregate reports: output/_reports/summary.json, quality_report.html
"""

import base64
import copy
import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import backoff
import logging
import os
import fitz  # PyMuPDF
import pypdfium2 as pdfium
from dotenv import load_dotenv
from sanitize import sanitize_error
from tqdm import tqdm
from google import genai
from google.genai import types

try:
    from ada_analytics import PipelineAnalyticsCollector
    _HAS_ANALYTICS = True
except ImportError:
    _HAS_ANALYTICS = False

# Load .env from pipeline root (one level up from src/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean from an env var (true/1/yes -> True, else default)."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes")


# Configuration — env vars override hardcoded defaults, CLI args override both
PROJECT_ID = os.environ.get("PROJECT_ID", "camp-ai-nc")
REGION = os.environ.get("GEMINI_LOCATION", "global")  # so we can call Gemini 3
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "65500"))
TEMPERATURE_EXTRACTION = float(os.environ.get("TEMPERATURE_EXTRACTION", "1.0"))
TEMPERATURE_VALIDATION = float(os.environ.get("TEMPERATURE_VALIDATION", "1.0"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
TOP_K = int(os.environ.get("TOP_K", "40"))

# Processing settings
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "30"))
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.80"))
RENDER_SCALE = int(os.environ.get("RENDER_SCALE", "3"))

# Feature flags
ENABLE_COHERENCE_CHECK = _env_bool("ENABLE_COHERENCE_CHECK", False)
ENABLE_IMAGE_EXTRACTION = _env_bool("ENABLE_IMAGE_EXTRACTION", True)
ENABLE_VIDEO_DETECTION = _env_bool("ENABLE_VIDEO_DETECTION", True)

# Video platform patterns
VIDEO_PATTERNS = [
    r'youtube\.com/watch',
    r'youtu\.be/',
    r'vimeo\.com/',
    r'sharepoint\.com.*video',
    r'sharepoint\.com.*:v:',
    r'stream\.microsoft\.com',
]

# Directories — env vars override defaults, --data-dir / --output-dir CLI args override both
DATA_FOLDER = Path(os.environ.get("DATA_FOLDER", "../../workspace/input"))
OUTPUT_FOLDER = Path(os.environ.get("OUTPUT_FOLDER", "../../workspace/output"))
REPORTS_FOLDER = OUTPUT_FOLDER / "_reports"

# Test file list - if set, only process these document IDs
TEST_FILE_LIST = os.environ.get("TEST_FILE_LIST") or None


# Suppress Google GenAI warning about non-text parts (thought_signature)
# See: https://github.com/googleapis/python-genai/issues/850
class _SuppressNonTextPartsWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "there are non-text parts in the response:" not in record.getMessage()


logging.getLogger("google_genai.types").addFilter(_SuppressNonTextPartsWarning())


# ---------------------------------------------------------------------------
# Gemini client initialization — lazy, supports both Vertex AI and API key mode
# ---------------------------------------------------------------------------

def _load_env_local() -> None:
    """Load .env.local from project root into os.environ (won't overwrite existing)."""
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
                print(f"  Loaded {loaded} env vars from {env_file}")
            return
        search = search.parent


client: genai.Client | None = None
_api_mode: bool = False


def _init_client(api_mode: bool = False) -> None:
    """Initialize the Gemini client (Vertex AI or API key mode).

    Safe to call multiple times — sets _api_mode flag so worker processes
    can auto-initialize with the same mode.
    """
    global client, _api_mode
    _api_mode = api_mode
    if api_mode:
        _load_env_local()
        # Try GEMINI_API_KEYS (comma-separated), then GEMINI_API_KEY, then GOOGLE_API_KEY
        api_key = os.environ.get("GEMINI_API_KEYS", "").split(",")[0].strip()
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "--api-mode requires API key via GEMINI_API_KEYS, GEMINI_API_KEY, "
                "or GOOGLE_API_KEY env var (or .env.local)"
            )
        client = genai.Client(api_key=api_key)
        print(f"  API key mode enabled")
    else:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=REGION,
        )


def _get_client() -> genai.Client:
    """Get the Gemini client, auto-initializing in worker processes if needed."""
    global client
    if client is None:
        _load_env_local()
        # In worker processes, check env for api mode signal
        api_key = os.environ.get("GEMINI_API_KEYS", "").split(",")[0].strip()
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client(
                vertexai=True,
                project=PROJECT_ID,
                location=REGION,
            )
    return client


def read_string_from_file(fname):
    with open(fname, "r") as f:
        return f.read()


EXTRACTION_PROMPT = read_string_from_file("PROMPT_FOR_EXTRACT.md")
COHERENCE_CHECK_PROMPT = read_string_from_file("PROMPT_FOR_VALIDATE.md")


def parse_test_file_list(md_path: Path) -> List[str]:
    """Parse document IDs from the test files list.

    Supports two formats:
    1. Plain text: one document ID per line
    2. Markdown table: document names in the second column (| # | Document | ...)

    Returns list of document IDs (folder names under data/).
    """
    if not md_path.exists():
        print(f"Warning: Test file list not found at {md_path}, processing all PDFs")
        return []

    text = md_path.read_text(encoding="utf-8")
    doc_ids = []

    # Detect format: if any non-empty line starts with |, treat as markdown table
    lines = text.splitlines()
    is_table = any(line.strip().startswith("|") for line in lines if line.strip())

    if is_table:
        for line in lines:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cols = [c.strip() for c in line.split("|")]
            if len(cols) < 3:
                continue
            num_col = cols[1]
            if not num_col or num_col.startswith("#") or num_col.startswith(":") or num_col.startswith("-"):
                continue
            try:
                int(num_col)
            except ValueError:
                continue
            doc_id = cols[2]
            if doc_id:
                doc_ids.append(doc_id)
    else:
        # Plain text: one document ID per line
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                doc_ids.append(line)

    return doc_ids

# Schema for structured output - uses anyOf for type-specific field validation
# Each content type only allows its specific fields to reduce token usage and prevent field confusion
# Note: Uses standard JSON Schema types (lowercase) for google-genai SDK
EXTRACTION_SCHEMA = {
    "type": "array",
    "items": {
        "anyOf": [
            # Heading: type, level, text
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["heading"]},
                    "level": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["type", "level", "text"],
            },
            # Paragraph: type, text
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["paragraph"]},
                    "text": {"type": "string"},
                },
                "required": ["type", "text"],
            },
            # Table: type, cells
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["table"]},
                    "cells": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "column_start": {"type": "integer"},
                                "row_start": {"type": "integer"},
                                "num_columns": {"type": "integer"},
                                "num_rows": {"type": "integer"},
                            },
                            "required": ["text", "column_start", "row_start"],
                        },
                    },
                },
                "required": ["type", "cells"],
            },
            # Image: type, description, caption, position
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["image"]},
                    "description": {"type": "string"},
                    "caption": {"type": "string"},
                    "position": {"type": "string"},
                },
                "required": ["type", "description"],
            },
            # Video: type, url, description
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["video"]},
                    "url": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["type", "url"],
            },
            # Form: type, title, fields
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["form"]},
                    "title": {"type": "string"},
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "field_type": {
                                    "type": "string",
                                    "enum": [
                                        "text", "textarea", "checkbox", "radio",
                                        "dropdown", "date", "signature", "number",
                                        "email", "phone", "unknown"
                                    ],
                                },
                                "value": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "required": {"type": "boolean"},
                                "position": {"type": "string"},
                            },
                            "required": ["label", "field_type"],
                        },
                    },
                },
                "required": ["type", "fields"],
            },
            # List: type, list_type, items (supports nested lists)
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["list"]},
                    "list_type": {
                        "type": "string",
                        "enum": ["ordered", "unordered"],
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "children": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"},
                                        },
                                        "required": ["text"],
                                    },
                                },
                            },
                            "required": ["text"],
                        },
                    },
                },
                "required": ["type", "list_type", "items"],
            },
            # Header/Footer: type, subtype, text
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["header_footer"]},
                    "subtype": {
                        "type": "string",
                        "enum": ["header", "footer"],
                    },
                    "text": {"type": "string"},
                },
                "required": ["type", "subtype", "text"],
            },
            # Link: type, text, url
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["link"]},
                    "text": {"type": "string"},
                    "url": {"type": "string"},
                },
                "required": ["type", "text", "url"],
            },
        ],
    },
}


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_content_item_html(item: Dict) -> str:
    """Render a single extracted content item as an HTML fragment.

    Handles all schema types: heading, paragraph, table, image, video,
    form, list, header_footer, link.
    """
    item_type = item.get("type", "unknown")

    if item_type == "heading":
        level = item.get("level", 1)
        text = _esc(item.get("text", ""))
        return (
            f'<div class="content-item">'
            f'<div class="content-type">Heading (H{level})</div>'
            f'<div><strong>{text}</strong></div>'
            f'</div>'
        )

    if item_type == "paragraph":
        text = _esc(item.get("text", ""))
        return (
            f'<div class="content-item">'
            f'<div class="content-type">Paragraph</div>'
            f'<div>{text}</div>'
            f'</div>'
        )

    if item_type == "table":
        cells = item.get("cells", [])
        if not cells:
            return (
                '<div class="content-item">'
                '<div class="content-type">Table (empty)</div>'
                '</div>'
            )
        max_row = max((c.get("row_start", 0) + c.get("num_rows", 1)) for c in cells)
        max_col = max((c.get("column_start", 0) + c.get("num_columns", 1)) for c in cells)
        tbl = '<table class="extracted-table">'
        for r in range(max_row):
            tbl += "<tr>"
            for c in range(max_col):
                cell_text = ""
                for cell in cells:
                    if cell.get("row_start") == r and cell.get("column_start") == c:
                        cell_text = _esc(cell.get("text", ""))
                        break
                tbl += f"<td>{cell_text}</td>"
            tbl += "</tr>"
        tbl += "</table>"
        return (
            f'<div class="content-item">'
            f'<div class="content-type">Table ({len(cells)} cells)</div>'
            f'{tbl}'
            f'</div>'
        )

    if item_type == "image":
        desc = _esc(item.get("description", "No description"))
        caption = item.get("caption", "")
        caption_html = f"<div><em>{_esc(caption)}</em></div>" if caption else ""
        return (
            f'<div class="content-item">'
            f'<div class="content-type">Image</div>'
            f'<div>{desc}</div>'
            f'{caption_html}'
            f'</div>'
        )

    if item_type == "video":
        url = _esc(item.get("url", ""))
        return (
            f'<div class="content-item">'
            f'<div class="content-type">Video</div>'
            f'<div><a href="{url}" target="_blank">{url}</a></div>'
            f'</div>'
        )

    if item_type == "form":
        form_title = _esc(item.get("title", "Untitled Form"))
        fields = item.get("fields", [])
        ftbl = '<table class="extracted-table"><tr><th>Label</th><th>Type</th><th>Value</th></tr>'
        for field in fields:
            label = _esc(field.get("label", ""))
            ftype = _esc(field.get("field_type", "unknown"))
            value = _esc(str(field.get("value") or "[empty]"))
            ftbl += f"<tr><td>{label}</td><td>{ftype}</td><td>{value}</td></tr>"
        ftbl += "</table>"
        return (
            f'<div class="content-item">'
            f'<div class="content-type">Form: {form_title}</div>'
            f'{ftbl}'
            f'</div>'
        )

    if item_type == "list":
        list_type = item.get("list_type", "unordered")
        tag = "ol" if list_type == "ordered" else "ul"
        items_html = ""
        for li in item.get("items", []):
            children_html = ""
            children = li.get("children", [])
            if children:
                children_html = "<ul>"
                for child in children:
                    children_html += f"<li>{_esc(child.get('text', ''))}</li>"
                children_html += "</ul>"
            items_html += f"<li>{_esc(li.get('text', ''))}{children_html}</li>"
        return (
            f'<div class="content-item">'
            f'<div class="content-type">List ({list_type})</div>'
            f'<{tag}>{items_html}</{tag}>'
            f'</div>'
        )

    if item_type == "header_footer":
        subtype = item.get("subtype", "header")
        label = "Header" if subtype == "header" else "Footer"
        text = _esc(item.get("text", ""))
        color = "#607D8B"
        return (
            f'<div class="content-item" style="border-left-color: {color}; background: #eceff1;">'
            f'<div class="content-type">{label}</div>'
            f'<div style="color: #546E7A; font-size: 0.9em;">{text}</div>'
            f'</div>'
        )

    if item_type == "link":
        text = _esc(item.get("text", ""))
        url = item.get("url", "")
        url_esc = _esc(url)
        return (
            f'<div class="content-item" style="border-left-color: #1976D2;">'
            f'<div class="content-type">Link</div>'
            f'<div><a href="{url_esc}" target="_blank">{text}</a></div>'
            f'<div style="font-size: 0.8em; color: #999;">{url_esc}</div>'
            f'</div>'
        )

    # Fallback for unknown types
    return (
        f'<div class="content-item">'
        f'<div class="content-type">{_esc(item_type)}</div>'
        f'<div>{_esc(json.dumps(item, default=str)[:500])}</div>'
        f'</div>'
    )


# Shared CSS used by both per-document HTML and sample review HTML
CONTENT_CSS = """
    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.5; }
    h1 { color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }
    h2 { color: #555; margin-top: 30px; }
    .page-section { margin: 20px 0; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafafa; }
    .page-header { font-size: 1.1em; font-weight: bold; color: #333; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #ddd; }
    .content-item { margin-bottom: 10px; padding: 8px; border-left: 3px solid #2196F3; background: #f5f5f5; }
    .content-type { font-weight: bold; color: #666; font-size: 0.85em; text-transform: uppercase; margin-bottom: 4px; }
    table.extracted-table { border-collapse: collapse; width: 100%; font-size: 0.9em; }
    table.extracted-table th, table.extracted-table td { border: 1px solid #ddd; padding: 4px 8px; }
    table.extracted-table th { background: #e0e0e0; }
    .meta { color: #888; font-size: 0.9em; margin-bottom: 15px; }
    ol, ul { margin: 4px 0; padding-left: 24px; }
    a { color: #1976D2; text-decoration: none; }
    a:hover { text-decoration: underline; }
"""


def generate_document_html(result: Dict, output_path: Path):
    """Generate a full HTML document from a single PDF's extracted JSON.

    Args:
        result: The full extraction result dict for one PDF (with pages array).
        output_path: Path to write the .html file.
    """
    pdf_id = result.get("pdf_id", "unknown")
    total_pages = result.get("total_pages", 0)
    timestamp = result.get("extraction_timestamp", "")
    metrics = result.get("quality_metrics", {})
    coherence = metrics.get("avg_coherence_score", "N/A")

    pages_html = ""
    for page in result.get("pages", []):
        page_num = page.get("page_number", "?")
        error = page.get("error")
        content = page.get("content", [])
        val = page.get("validation", {})
        page_coherence = val.get("coherence_score", "N/A")

        if error:
            content_html = f'<p style="color: #c62828;">Extraction error: {_esc(error)}</p>'
        elif not content:
            content_html = '<p style="color: #999;">No content extracted</p>'
        else:
            content_html = "\n".join(render_content_item_html(item) for item in content)

        pages_html += f"""
    <div class="page-section">
        <div class="page-header">Page {page_num} <span style="font-weight: normal; color: #888;">(coherence: {page_coherence}/10)</span></div>
        {content_html}
    </div>
"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{_esc(pdf_id)} - Extracted Content</title>
    <style>{CONTENT_CSS}</style>
</head>
<body>
    <h1>{_esc(pdf_id)}</h1>
    <div class="meta">
        Pages: {total_pages} | Avg Coherence: {coherence}/10 | Extracted: {timestamp}
    </div>
{pages_html}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _render_single_page(args: Tuple[str, int]) -> bytes:
    """Render a single PDF page to PNG bytes. Module-level for ProcessPoolExecutor."""
    import io
    pdf_path_str, page_num = args
    doc = pdfium.PdfDocument(pdf_path_str)
    page = doc.get_page(page_num)
    bitmap = page.render(scale=RENDER_SCALE)
    pil_image = bitmap.to_pil()
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _process_page_worker(args: Tuple[str, int, str]) -> Tuple[str, int, Dict]:
    """Process a single PDF page in a worker process.

    Module-level function for multiprocessing.Pool. Each worker process
    re-imports this module and gets its own Gemini client.

    Args:
        args: Tuple of (pdf_path_str, page_num, doc_id)

    Returns:
        Tuple of (doc_id, page_num, result_dict)
    """
    pdf_path_str, page_num, doc_id = args
    extractor = PDFExtractor()
    result = extractor.process_single_page(Path(pdf_path_str), page_num)
    return doc_id, page_num, result


def get_safety_settings():
    """Return safety settings that allow all content."""
    return [
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE"
        ),
    ]


class PDFExtractor:
    """Main class for PDF extraction with quality verification."""

    def __init__(self):
        self.stats = {
            "total_pdfs": 0,
            "total_pages": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "pages_by_confidence": {"high": 0, "medium": 0, "low": 0},
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
        self._analytics = None
        if _HAS_ANALYTICS:
            try:
                self._analytics = PipelineAnalyticsCollector(
                    tenant_id=PROJECT_ID,
                    gemini_model=GEMINI_MODEL,
                )
                logging.getLogger(__name__).info("Pipeline analytics collector initialised")
            except Exception as exc:
                logging.getLogger(__name__).debug("Analytics init failed (non-fatal): %s", exc)

    def render_page_to_image(self, pdf_path: Path, page_num: int) -> bytes:
        """Render a PDF page to PNG image bytes."""
        doc = pdfium.PdfDocument(str(pdf_path))
        page = doc.get_page(page_num)
        bitmap = page.render(scale=RENDER_SCALE)
        pil_image = bitmap.to_pil()

        # Convert to bytes
        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    @backoff.on_exception(backoff.expo, Exception, max_tries=2)
    def call_gemini_for_extraction(self, image_bytes: bytes, temperature: float = TEMPERATURE_EXTRACTION) -> Tuple[str, int, int]:
        """Call Gemini API with structured output schema (synchronous).

        Args:
            image_bytes: PNG image data
            temperature: Model temperature (default TEMPERATURE_EXTRACTION)

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        model = GEMINI_MODEL

        config = types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=temperature,
            top_p=TOP_P,
            top_k=TOP_K,
            response_mime_type="application/json",
            response_schema=EXTRACTION_SCHEMA,
            safety_settings=get_safety_settings(),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH
        )
        response = _get_client().models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                EXTRACTION_PROMPT,
            ],
            config=config,
        )

        # Extract token usage from response metadata
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

        return response.text, input_tokens, output_tokens

    @backoff.on_exception(backoff.expo, Exception, max_tries=2)
    def call_gemini_text_sync(self, prompt: str, temperature: float = TEMPERATURE_VALIDATION) -> Tuple[str, int, int]:
        """Call Gemini API with text-only prompt (no image).

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        config = types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=temperature,
            top_p=TOP_P,
            top_k=TOP_K,
            safety_settings=get_safety_settings(),
        )

        response = _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )

        # Extract token usage from response metadata
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

        return response.text, input_tokens, output_tokens
 
    ALT_TEXT_PROMPT = (
        "Describe this image for use as alt text on a web page. "
        "Write a concise description (1-3 sentences) of what the image visually shows. "
        "Be specific: identify people, objects, logos, charts, maps, diagrams, or scenes. "
        "If there is text visible in the image (e.g., a title, label, or caption), include the key text. "
        "Do NOT say 'image of' or 'picture of' — just describe the content directly. "
        "Do NOT transcribe all text in the image — just summarize what it shows. "
        "Return ONLY the alt text description, no JSON, no quotes, no extra formatting."
    )

    def generate_alt_text_for_image(self, image_bytes: bytes, image_format: str = "png") -> Tuple[str, int, int]:
        """Generate alt text for a single image by sending it to Gemini.

        This produces more accurate alt text than the full-page extraction approach
        because Gemini sees only the specific image, not the whole page.

        Args:
            image_bytes: Raw image data (PNG, JPEG, etc.)
            image_format: Image format string (e.g., "png", "jpeg")

        Returns:
            Tuple of (alt_text, input_tokens, output_tokens)
        """
        return self._call_gemini_for_alt_text(image_bytes, image_format, self.ALT_TEXT_PROMPT)

    FULL_PAGE_ALT_TEXT_PROMPT = (
        "This is a full-page rendering of a PDF page that contains multiple overlapping images "
        "(such as a presentation slide, infographic, or layered design). "
        "Describe the overall visual content of this page for use as alt text on a web page. "
        "Write a concise description (2-4 sentences) of the key visual elements: images, diagrams, "
        "charts, logos, photographs, and their spatial arrangement. "
        "Include any visible titles or key labels, but do NOT transcribe all text. "
        "Do NOT say 'image of' or 'picture of'. "
        "Return ONLY the alt text description, no JSON, no quotes, no extra formatting."
    )

    def regenerate_alt_text_for_images(
        self, images: List[Dict], pdf_name: str, page_num: int
    ) -> Tuple[int, int]:
        """Regenerate alt text for all images that have base64 data.

        Sends each individual image to Gemini for accurate per-image alt text,
        replacing the position-based descriptions from the full-page extraction.

        For full-page composite renders (5+ images merged into one), uses a
        specialized prompt and makes only ONE Gemini call for the entire composite.

        Args:
            images: List of image dicts (modified in-place)
            pdf_name: PDF filename for logging
            page_num: 0-indexed page number for logging

        Returns:
            Tuple of (total_input_tokens, total_output_tokens)
        """
        total_input = 0
        total_output = 0

        for i, img in enumerate(images):
            # Skip images without binary data
            if "base64_data" not in img:
                continue

            # Decode base64 to bytes
            try:
                image_bytes = base64.b64decode(img["base64_data"])
            except Exception:
                continue

            # Skip tiny images (likely icons/spacers, < 1KB)
            if len(image_bytes) < 1024:
                continue

            img_format = img.get("format", "png")
            old_desc = img.get("description", "")

            # Choose prompt based on whether this is a full-page composite render
            is_composite = img.get("_full_page_render", False)

            try:
                if is_composite:
                    # Full-page composite: single Gemini call with specialized prompt
                    alt_text, inp_tok, out_tok = self._call_gemini_for_alt_text(
                        image_bytes, img_format, self.FULL_PAGE_ALT_TEXT_PROMPT
                    )
                else:
                    alt_text, inp_tok, out_tok = self.generate_alt_text_for_image(
                        image_bytes, img_format
                    )
                total_input += inp_tok
                total_output += out_tok

                if alt_text:
                    img["description"] = alt_text
                    label = "composite" if is_composite else f"image {i+1}"
                    print(
                        f"  [{pdf_name}] Page {page_num + 1}: {label} alt text: "
                        f"\"{alt_text[:60]}...\" (was: \"{old_desc[:40]}...\")"
                    )
            except Exception as e:
                print(
                    f"  [{pdf_name}] Page {page_num + 1}: Image {i+1} alt text failed: {e}"
                )
                # Keep the original description on failure

        return total_input, total_output

    @backoff.on_exception(backoff.expo, Exception, max_tries=2)
    def _call_gemini_for_alt_text(self, image_bytes: bytes, image_format: str, prompt: str) -> Tuple[str, int, int]:
        """Call Gemini with an image and a custom prompt to generate alt text.

        Low-level helper used by both generate_alt_text_for_image() and
        the full-page composite handler.
        """
        mime_map = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
        }
        mime_type = mime_map.get(image_format.lower(), "image/png")

        config = types.GenerateContentConfig(
            max_output_tokens=500,
            temperature=TEMPERATURE_EXTRACTION,
            top_p=TOP_P,
            top_k=TOP_K,
            safety_settings=get_safety_settings(),
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=config,
        )

        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

        alt_text = (response.text or "").strip()
        # Strip any surrounding quotes Gemini may add
        if alt_text.startswith('"') and alt_text.endswith('"'):
            alt_text = alt_text[1:-1]
        if alt_text.startswith("'") and alt_text.endswith("'"):
            alt_text = alt_text[1:-1]

        return alt_text, input_tokens, output_tokens

    def check_coherence(self, content: List[Dict]) -> Tuple[Dict, int, int]:
        """
        Use LLM to evaluate the coherence and completeness of extracted content.

        Returns:
            Tuple of (result dict with coherence_score and issues, input_tokens, output_tokens)
        """
        if not ENABLE_COHERENCE_CHECK:
            return {"coherence_score": None, "issues": []}, 0, 0

        # Format content for review
        content_text = self._format_content_for_review(content)

        # Limit content length to avoid token limits
        if len(content_text) > 8000:
            content_text = content_text[:8000] + "\n... [truncated for review]"

        prompt = COHERENCE_CHECK_PROMPT.format(content=content_text)

        try:
            response, input_tokens, output_tokens = self.call_gemini_text_sync(prompt, temperature=TEMPERATURE_VALIDATION)

            # Parse the response
            text = response.strip()
            if text.startswith("```"):
                text = re.sub(r'^```(?:json)?\s*\n', '', text)
                text = re.sub(r'\n```\s*$', '', text)

            result = json.loads(text)
            return {
                "coherence_score": result.get("coherence_score"),
                "issues": result.get("issues", []),
            }, input_tokens, output_tokens
        except Exception as e:
            return {
                "coherence_score": None,

                "issues": [f"Coherence check failed: {sanitize_error(str(e))}"],
            }, 0, 0

    def _format_content_for_review(self, content: List[Dict]) -> str:
        """Format extracted content as readable text for coherence review."""
        parts = []
        for item in content:
            item_type = item.get("type")
            if item_type == "heading":
                level = item.get("level", 1)
                parts.append(f"[H{level}] {item.get('text', '')}\n")
            elif item_type == "paragraph":
                parts.append(f"[PARAGRAPH]\n{item.get('text', '')}\n")
            elif item_type == "table":
                parts.append("[TABLE]")
                cells = item.get("cells", [])
                if cells:
                    # Group cells by row
                    rows = {}
                    for cell in cells:
                        row = cell.get("row_start", 0)
                        if row not in rows:
                            rows[row] = []
                        rows[row].append(cell.get("text", ""))
                    for row_num in sorted(rows.keys()):
                        parts.append(f"  Row {row_num}: {' | '.join(rows[row_num])}")
                parts.append("")
            elif item_type == "image":
                desc = item.get("description", "No description")
                parts.append(f"[IMAGE: {desc}]\n")
            elif item_type == "video":
                url = item.get("url", "")
                parts.append(f"[VIDEO: {url}]\n")
            elif item_type == "list":
                list_type = item.get("list_type", "unordered")
                parts.append(f"[LIST ({list_type})]")
                for idx, li in enumerate(item.get("items", []), 1):
                    prefix = f"  {idx}." if list_type == "ordered" else "  -"
                    parts.append(f"{prefix} {li.get('text', '')}")
                    for child in li.get("children", []):
                        parts.append(f"    - {child.get('text', '')}")
                parts.append("")
            elif item_type == "form":
                form_title = item.get("title", "Untitled Form")
                parts.append(f"[FORM: {form_title}]")
                for field in item.get("fields", []):
                    label = field.get("label", "Unknown")
                    field_type = field.get("field_type", "unknown")
                    value = field.get("value", "")
                    parts.append(f"  - {label} ({field_type}): {value or '[empty]'}")
                parts.append("")
            elif item_type == "header_footer":
                subtype = item.get("subtype", "header").upper()
                parts.append(f"[{subtype}] {item.get('text', '')}\n")
            elif item_type == "link":
                text = item.get("text", "")
                url = item.get("url", "")
                parts.append(f"[LINK: {text} -> {url}]\n")
        return "\n".join(parts)

    def parse_json_response(self, response_text: str) -> Tuple[List[Dict], bool, str]:
        """Parse JSON from Gemini response, handling markdown code blocks.

        Returns:
            Tuple of (data, success, error_detail)
            - data: parsed list of content items (empty list on failure)
            - success: True if parsing succeeded
            - error_detail: description of failure (empty string on success)
        """
        # Handle None response (can happen when Gemini returns empty/blocked response)
        if response_text is None:
            return [], False, "Empty response from Gemini (response.text was None)"

        text = response_text.strip()

        # Check for empty response
        if not text:
            return [], False, "Empty response from Gemini"

        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*\n', '', text)
            text = re.sub(r'\n```\s*$', '', text)

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data, True, ""
            # Valid JSON but not a list
            return [], False, f"Response is {type(data).__name__}, not list. Content: {text[:200]}"
        except json.JSONDecodeError as e:
            # Truncate response for error message
            snippet = text[:200] + "..." if len(text) > 200 else text
            return [], False, f"JSON parse error: {e}. Response: {snippet}"

    def extract_video_links_from_page(self, pdf_path: Path, page_num: int) -> List[Dict]:
        """Extract video links from a PDF page using PyMuPDF."""
        if not ENABLE_VIDEO_DETECTION:
            return []

        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        links = page.get_links()

        video_links = []
        for link in links:
            uri = link.get("uri", "")
            if not uri:
                continue

            # Check if URL matches any video platform pattern
            is_video = False
            platform = None
            for pattern in VIDEO_PATTERNS:
                if re.search(pattern, uri, re.IGNORECASE):
                    is_video = True
                    if "youtube" in pattern or "youtu.be" in pattern:
                        platform = "youtube"
                    elif "vimeo" in pattern:
                        platform = "vimeo"
                    elif "sharepoint" in pattern or "stream.microsoft" in pattern:
                        platform = "microsoft"
                    break

            if is_video:
                # Get link bounding box
                link_rect = link.get("from")
                bbox = None
                if link_rect:
                    bbox = {
                        "x0": link_rect.x0,
                        "y0": link_rect.y0,
                        "x1": link_rect.x1,
                        "y1": link_rect.y1,
                    }

                video_links.append({
                    "url": uri,
                    "platform": platform,
                    "bbox": bbox,
                })

        doc.close()
        return video_links

    def extract_hyperlinks_from_page(self, pdf_path: Path, page_num: int) -> List[Dict]:
        """Extract all non-video hyperlinks from a PDF page using PyMuPDF.

        Returns a list of dicts with keys: url, text, bbox.
        The 'text' is extracted from the text under the link's bounding box.
        Video links are excluded (handled separately by extract_video_links_from_page).
        """
        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        links = page.get_links()

        hyperlinks = []
        for link in links:
            uri = link.get("uri", "")
            if not uri:
                continue

            # Skip video links - those are handled by extract_video_links_from_page
            is_video = any(
                re.search(pattern, uri, re.IGNORECASE)
                for pattern in VIDEO_PATTERNS
            )
            if is_video:
                continue

            # Get the display text under the link's bounding box
            link_rect = link.get("from")
            display_text = ""
            bbox = None
            if link_rect:
                display_text = page.get_text("text", clip=link_rect).strip()
                bbox = {
                    "x0": link_rect.x0,
                    "y0": link_rect.y0,
                    "x1": link_rect.x1,
                    "y1": link_rect.y1,
                }

            # Use URL as fallback display text if none found
            if not display_text:
                display_text = uri

            hyperlinks.append({
                "url": uri,
                "text": display_text,
                "bbox": bbox,
            })

        doc.close()
        return hyperlinks

    def extract_images_from_pdf_page(self, pdf_path: Path, page_num: int) -> List[Dict]:
        """Extract embedded images from a PDF page using PyMuPDF.

        Uses a hybrid approach depending on image characteristics:

        1. Full-page background layers (bbox covers ≥90% page width and ≥50%
           page height): rendered via page.get_pixmap(clip=rect) to composite
           all overlapping layers correctly. PDFs often use multiple stacked
           full-page images (photo + grunge texture + gradient) that only look
           correct when composited together.

        2. Images with SMask (transparency mask): extracted via
           doc.extract_image(xref) then composited with the mask using PIL.
           The SMask is stored as a separate PDF object and must be applied
           as an alpha channel.

        3. All other images: extracted via doc.extract_image(xref) with zero
           post-processing. This preserves indexed/palette colorspaces and
           original compression (JPEG stays JPEG).

        When ENABLE_IMAGE_EXTRACTION is True, includes base64 image data.
        When False, still returns image metadata (bbox, format) but no base64 data.
        """
        import io
        from PIL import Image as PILImage

        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        page_rect = page.rect
        image_list = page.get_images(full=True)

        extracted_images = []
        for img_index, img in enumerate(image_list):
            xref = img[0]
            smask_xref = img[1]  # SMask xref (0 if none)
            try:
                base_image = doc.extract_image(xref)
                image_ext = base_image["ext"]

                # Get bounding box
                image_rects = page.get_image_rects(xref)
                bbox = None
                rect = None
                if image_rects:
                    rect = image_rects[0]
                    bbox = {
                        "x0": rect.x0,
                        "y0": rect.y0,
                        "x1": rect.x1,
                        "y1": rect.y1,
                    }

                image_entry = {
                    "index": img_index,
                    "format": image_ext,
                    "bbox": bbox,
                }

                if ENABLE_IMAGE_EXTRACTION:
                    image_bytes = base_image["image"]

                    # Check if this is a full-page background layer
                    is_full_page = (
                        rect is not None
                        and rect.width >= page_rect.width * 0.9
                        and rect.height >= page_rect.height * 0.5
                    )

                    if is_full_page:
                        # Check if this page has MULTIPLE stacked full-page
                        # images sharing the same bbox (layered backgrounds).
                        # Only these need compositing via get_pixmap.
                        # Single full-page images use raw extract_image.
                        fp_count = 0
                        for other_img in image_list:
                            other_rects = page.get_image_rects(other_img[0])
                            if other_rects:
                                or_ = other_rects[0]
                                if (or_.width >= page_rect.width * 0.9
                                        and or_.height >= page_rect.height * 0.5):
                                    fp_count += 1

                        if fp_count >= 2:
                            # Multiple stacked layers — render composited page
                            mat = fitz.Matrix(3, 3)
                            pix = page.get_pixmap(matrix=mat, clip=rect)
                            image_bytes = pix.tobytes("png")
                            image_entry["format"] = "png"
                        # else: single full-page image, raw bytes are fine
                    elif smask_xref:
                        # Composite SMask transparency using PIL
                        smask_data = doc.extract_image(smask_xref)
                        img_pil = PILImage.open(io.BytesIO(image_bytes))
                        smask_pil = PILImage.open(io.BytesIO(smask_data["image"]))
                        if img_pil.size != smask_pil.size:
                            smask_pil = smask_pil.resize(img_pil.size, PILImage.LANCZOS)
                        img_rgba = img_pil.convert("RGBA")
                        if smask_pil.mode != "L":
                            smask_pil = smask_pil.convert("L")
                        img_rgba.putalpha(smask_pil)
                        buf = io.BytesIO()
                        img_rgba.save(buf, format="PNG")
                        image_bytes = buf.getvalue()
                        image_entry["format"] = "png"
                    # else: raw extract_image bytes — no post-processing

                    image_entry["base64_data"] = base64.b64encode(image_bytes).decode("utf-8")

                extracted_images.append(image_entry)
            except Exception:
                continue

        # Full-page render: when a page has many embedded images (5+),
        # they are almost certainly layered/composited elements (presentation
        # slides, diagrams, infographics) that only look correct together.
        # Replace ALL individual images with a single full-page screenshot
        # rendered via get_pixmap to preserve exact spatial relationships.
        if ENABLE_IMAGE_EXTRACTION and len(extracted_images) >= 5:
            mat = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
            pix = page.get_pixmap(matrix=mat)
            page_png = pix.tobytes("png")
            extracted_images = [{
                "index": 0,
                "format": "png",
                "bbox": {
                    "x0": page_rect.x0, "y0": page_rect.y0,
                    "x1": page_rect.x1, "y1": page_rect.y1,
                },
                "base64_data": base64.b64encode(page_png).decode("utf-8"),
                "_full_page_render": True,
            }]
        else:
            # Deduplicate images with heavily overlapping bboxes.
            # When multiple images share the same area (>80% overlap),
            # keep only the largest one. This prevents the same content
            # being shown multiple times (e.g., page images captured twice
            # at different sizes).
            extracted_images = self._deduplicate_overlapping_images(extracted_images)


        doc.close()
        return extracted_images

    def _deduplicate_overlapping_images(self, images: List[Dict]) -> List[Dict]:
        """Remove duplicate images that cover the same area on the page.

        When multiple images have bounding boxes that overlap by >80%,
        keep only the one with the largest area (highest resolution).
        This prevents the same content appearing multiple times
        (e.g., page screenshots captured at different sizes).
        """
        if len(images) <= 1:
            return images

        # Mark images to remove
        to_remove = set()
        for i in range(len(images)):
            if i in to_remove:
                continue
            bbox_i = images[i].get("bbox")
            if not bbox_i:
                continue
            area_i = (bbox_i["x1"] - bbox_i["x0"]) * (bbox_i["y1"] - bbox_i["y0"])

            for j in range(i + 1, len(images)):
                if j in to_remove:
                    continue
                bbox_j = images[j].get("bbox")
                if not bbox_j:
                    continue

                if self._bboxes_overlap(bbox_i, bbox_j, threshold=0.8):
                    # Keep the larger image, remove the smaller one
                    area_j = (bbox_j["x1"] - bbox_j["x0"]) * (bbox_j["y1"] - bbox_j["y0"])
                    if area_i >= area_j:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break  # i is removed, stop comparing it

        return [img for idx, img in enumerate(images) if idx not in to_remove]

    def _render_image_region_fallback(
        self, pdf_path: Path, page_num: int, position: str,
        page_height: float, page_width: float
    ) -> Optional[str]:
        """Render the image region of a page, excluding surrounding text blocks.

        Uses PyMuPDF text block analysis to detect where body text starts/ends
        around the image, then crops to just the image region.  This prevents
        duplicating OCR'd text that Gemini will also extract as structured text.

        Returns None when the candidate crop is dominated by text (>60% text
        coverage), which means the "image" Gemini identified is really a region
        of ordinary text — rendering it would duplicate the OCR output.

        NOTE: The ideal architecture would extract/mask known image regions
        BEFORE passing the page to Gemini, so Gemini never OCRs content inside
        image bounding boxes.  That requires two-pass extraction and is tracked
        as a future improvement.  For now this method provides the best
        single-pass approximation: crop to the tightest non-text region, and
        skip rendering entirely when the region is mostly text.

        Algorithm:
        1. Extract all text blocks from the page and cluster into regions
           (merge gap ≤ 60 pt).
        2. Derive a vertical center (v_center) from the position label.
        3. y_crop_start = y1 of the last region with max_block_chars ≥
           SUBSTANTIAL_START (100) whose y0 is before v_center.
           (Always 0 when v_pos == "top".)
        4. y_crop_end = y0 of the first region with max_block_chars ≥
           SUBSTANTIAL_END (120) whose y0 is after v_center AND at least
           MIN_DIST (10 % of page height) away from v_center.
           The MIN_DIST guard prevents diagram labels that sit just below
           v_center from being mistaken for body-text boundaries.
        5. If the resulting crop is shorter than MIN_CROP_HEIGHT (50 pt),
           fall back to position-based bounds (top/bottom/middle thirds).
        6. Compute text coverage = sum of text-block heights overlapping the
           crop / crop height.  If coverage > OVERLAP_THRESHOLD (60 %),
           return None — the region is too text-heavy to render as an image.
        7. Render full-page-width crop at RENDER_SCALE and return as PNG b64.

        Returns:
            Base64-encoded PNG string, or None if rendering fails or the crop
            would duplicate OCR'd text.
        """
        MERGE_GAP = 60           # pt: cluster text blocks within this gap
        SUBSTANTIAL_START = 100  # chars: min block length for y_crop_start boundary
        SUBSTANTIAL_END = 120    # chars: min block length for y_crop_end boundary
        MIN_DIST_FRAC = 0.10     # y_crop_end region must be ≥ this fraction of
                                  # page height away from v_center
        MIN_CROP_HEIGHT = 50     # pt: crop shorter than this → use position fallback
        OVERLAP_THRESHOLD = 0.60 # text coverage fraction above which → return None

        try:
            doc = fitz.open(str(pdf_path))
            page = doc[page_num]
            page_rect = page.rect  # fitz.Rect with .y0, .y1, .x0, .x1
            ph = float(page_rect.y1)   # page height in pts
            pw = float(page_rect.x1)   # page width in pts
            min_dist = ph * MIN_DIST_FRAC

            # --- Parse position label into vertical position hint ---
            pos_lower = position.lower()
            if "top" in pos_lower:
                v_pos = "top"
                v_center = ph * 0.20
            elif "bottom" in pos_lower:
                v_pos = "bottom"
                v_center = ph * 0.80
            else:
                v_pos = "middle"
                v_center = ph * 0.50

            # --- Collect text blocks: (y0, y1, text) ---
            raw_blocks = []
            for blk in page.get_text("blocks"):
                # blocks: (x0, y0, x1, y1, text, block_no, block_type)
                # block_type 0 = text, 1 = image
                if len(blk) >= 7 and blk[6] == 0:
                    y0, y1, text = blk[1], blk[3], blk[4]
                    if text.strip():
                        raw_blocks.append((y0, y1, text))
            raw_blocks.sort(key=lambda b: b[0])

            # --- Cluster into regions ---
            regions = []  # list of [y0, y1, max_single_block_chars]
            for y0, y1, text in raw_blocks:
                nch = len(text.strip())
                if regions and y0 - regions[-1][1] <= MERGE_GAP:
                    regions[-1][1] = max(regions[-1][1], y1)
                    regions[-1][2] = max(regions[-1][2], nch)
                else:
                    regions.append([y0, y1, nch])

            # --- Determine crop boundaries ---
            if not regions:
                # No extractable PDF text at all (fully scanned page).
                # Start with position-based thirds, then try to tighten using
                # small overlay images embedded in the PDF (seals, stamps,
                # signature lines, logos).  These overlays often mark exactly
                # where the visual element lives, giving us a much tighter crop
                # than a generic one-third of the page.
                OVERLAY_PAD_ABOVE = 20.0   # pt above the topmost overlay image
                OVERLAY_PAD_BELOW = 50.0   # pt below the bottommost overlay image
                OVERLAY_MIN_HT   = 80.0    # minimum crop height when using overlays

                if v_pos == "top":
                    y_crop_start, y_crop_end = 0.0, ph * 0.35
                elif v_pos == "bottom":
                    y_crop_start, y_crop_end = ph * 0.65, ph
                else:
                    y_crop_start, y_crop_end = ph * 0.20, ph * 0.80

                # Quadrant boundaries for filtering overlay images
                quad_y_lo = 0.0 if v_pos == "top" else (ph * 0.50 if v_pos == "bottom" else 0.0)
                quad_y_hi = (ph * 0.50 if v_pos == "top" else ph)
                quad_x_lo = 0.0 if "right" not in pos_lower else pw * 0.40
                quad_x_hi = pw if "left" not in pos_lower else pw * 0.60

                ov_y0s, ov_y1s = [], []
                for ov_img in page.get_images(full=True):
                    ov_xref = ov_img[0]
                    ov_rects = page.get_image_rects(ov_xref)
                    if not ov_rects:
                        continue
                    r = ov_rects[0]
                    wf = r.width / pw
                    hf = r.height / ph
                    # Exclude full-page backgrounds and large text templates
                    if wf >= 0.90 and hf >= 0.85:
                        continue
                    if wf * hf > 0.30:
                        continue
                    # Include only overlays whose centre falls inside the quadrant
                    cx = (r.x0 + r.x1) / 2.0
                    cy = (r.y0 + r.y1) / 2.0
                    if quad_y_lo <= cy <= quad_y_hi and quad_x_lo <= cx <= quad_x_hi:
                        ov_y0s.append(r.y0)
                        ov_y1s.append(r.y1)

                if ov_y0s:
                    tight_y0 = max(min(ov_y0s) - OVERLAY_PAD_ABOVE, 0.0)
                    tight_y1 = min(max(ov_y1s) + OVERLAY_PAD_BELOW, ph)
                    if tight_y1 - tight_y0 < OVERLAY_MIN_HT:
                        mid = (tight_y0 + tight_y1) / 2.0
                        tight_y0 = max(0.0, mid - OVERLAY_MIN_HT / 2.0)
                        tight_y1 = min(ph, mid + OVERLAY_MIN_HT / 2.0)
                    y_crop_start, y_crop_end = tight_y0, tight_y1
            else:
                # y_crop_start: bottom of the last "body text" region above
                # v_center.  Uses the more lenient SUBSTANTIAL_START threshold
                # so pages where every line is short (e.g. numbered legislation)
                # still get a boundary.
                y_crop_start = 0.0
                if v_pos != "top":
                    for (ry0, ry1, max_nch) in regions:
                        if ry0 < v_center and max_nch >= SUBSTANTIAL_START:
                            y_crop_start = ry1

                # y_crop_end: top of the first "body text" region below v_center
                # that is also far enough from v_center to not be a diagram label.
                # MIN_DIST prevents short in-diagram captions (77 pt below center
                # on page 37) from cutting off the diagram; distant body text
                # (240 pt below center) is used correctly.
                y_crop_end = ph
                for (ry0, ry1, max_nch) in regions:
                    if ry0 > v_center and max_nch >= SUBSTANTIAL_END and (ry0 - v_center) >= min_dist:
                        y_crop_end = ry0
                        break

                # Clamp to page bounds
                y_crop_start = max(0.0, y_crop_start)
                y_crop_end = min(ph, y_crop_end)

                # If the resulting crop is too narrow, fall back to position-based
                if y_crop_end - y_crop_start < MIN_CROP_HEIGHT:
                    if v_pos == "top":
                        y_crop_start, y_crop_end = 0.0, ph * 0.35
                    elif v_pos == "bottom":
                        y_crop_start, y_crop_end = ph * 0.65, ph
                    else:
                        y_crop_start, y_crop_end = ph * 0.20, ph * 0.80

            # --- Horizontal bounds from position label ---
            # "left" / "right" sub-positions halve the page horizontally so that
            # e.g. "bottom-left" and "bottom-right" signatures each get their own
            # crop rather than the same full-width strip.
            pw = float(page_rect.x1)
            if "left" in pos_lower:
                x_crop_start, x_crop_end = 0.0, pw * 0.55  # slight overlap at centre
            elif "right" in pos_lower:
                x_crop_start, x_crop_end = pw * 0.45, pw
            else:
                x_crop_start, x_crop_end = 0.0, pw

            # --- Text-coverage guard ---
            # If the crop region is mostly covered by text, the "image" Gemini
            # identified is really a text region.  Rendering it would duplicate
            # OCR'd content, so skip it entirely.
            #
            # Coverage is computed on *clustered regions* rather than raw blocks
            # to avoid false positives on diagrams whose visual boxes have PDF
            # text layers: those overlapping short blocks merge into one small
            # region, giving a low coverage fraction even though raw-block sums
            # can exceed 100 %.
            crop_height = y_crop_end - y_crop_start
            text_coverage = 0.0
            if crop_height > 0:
                for ry0, ry1, _ in regions:
                    overlap = min(ry1, y_crop_end) - max(ry0, y_crop_start)
                    if overlap > 0:
                        text_coverage += overlap
                text_coverage = min(text_coverage / crop_height, 1.0)

            if text_coverage > OVERLAP_THRESHOLD:
                doc.close()
                return None  # "image" is a text-heavy region — omit to avoid duplication

            # --- Render the crop ---
            clip = fitz.Rect(x_crop_start, y_crop_start, x_crop_end, y_crop_end)
            mat = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
            pix = page.get_pixmap(matrix=mat, clip=clip)
            doc.close()
            png_bytes = pix.tobytes("png")
            return base64.b64encode(png_bytes).decode("utf-8")
        except Exception:
            return None

    def extract_text_with_pymupdf(self, pdf_path: Path, page_num: int) -> str:
        """Extract text from a PDF page using PyMuPDF for cross-validation."""
        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        text = page.get_text("text")
        doc.close()
        return text

    def _bboxes_overlap(self, bbox1: Optional[Dict], bbox2: Optional[Dict], threshold: float = 0.5) -> bool:
        """Check if two bounding boxes overlap significantly."""
        if not bbox1 or not bbox2:
            return False

        # Calculate intersection
        x_left = max(bbox1["x0"], bbox2["x0"])
        y_top = max(bbox1["y0"], bbox2["y0"])
        x_right = min(bbox1["x1"], bbox2["x1"])
        y_bottom = min(bbox1["y1"], bbox2["y1"])

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate areas
        bbox1_area = (bbox1["x1"] - bbox1["x0"]) * (bbox1["y1"] - bbox1["y0"])
        bbox2_area = (bbox2["x1"] - bbox2["x0"]) * (bbox2["y1"] - bbox2["y0"])

        # Check if intersection is significant relative to either bbox
        min_area = min(bbox1_area, bbox2_area)
        if min_area == 0:
            return False

        overlap_ratio = intersection_area / min_area
        return overlap_ratio >= threshold

    def match_images_to_descriptions(
        self, gemini_images: List[Dict], pymupdf_images: List[Dict],
        video_links: List[Dict], page_height: float, page_width: float = 612.0
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Match Gemini image descriptions to PyMuPDF extracted images by position.
        Also identifies which images are actually video thumbnails.

        Returns:
            Tuple of (matched_images, video_items)
        """
        matched_images = []
        video_items = []

        # First, identify images that are video thumbnails by checking bbox overlap
        video_image_indices = set()
        for video_link in video_links:
            video_bbox = video_link.get("bbox")
            if not video_bbox:
                continue

            # Find image that overlaps with this video link
            for i, pymupdf_img in enumerate(pymupdf_images):
                if self._bboxes_overlap(video_bbox, pymupdf_img.get("bbox")):
                    video_image_indices.add(i)

                    # Create video item with thumbnail
                    video_item = {
                        "type": "video",
                        "url": video_link["url"],
                        "platform": video_link["platform"],
                        "thumbnail_format": pymupdf_img["format"],
                        "bbox": video_bbox,
                        "description": None,  # Will be filled from Gemini if available
                    }
                    if "base64_data" in pymupdf_img:
                        video_item["thumbnail_base64"] = pymupdf_img["base64_data"]
                    video_items.append(video_item)
                    break
            else:
                # No matching image found, still create video item without thumbnail
                video_items.append({
                    "type": "video",
                    "url": video_link["url"],
                    "platform": video_link["platform"],
                    "bbox": video_bbox,
                    "description": None,
                })

        # Remove video thumbnails from the image list
        remaining_images = [
            img for i, img in enumerate(pymupdf_images)
            if i not in video_image_indices
        ]

        if not remaining_images and not gemini_images:
            return [], video_items


        position_ranges = {}
        # Map position strings to vertical and horizontal ranges
        # Uses 2D matching to better distinguish images at same vertical level
        v_ranges = {
            "top": (0, page_height / 3),
            "middle": (page_height / 3, 2 * page_height / 3),
            "bottom": (2 * page_height / 3, page_height),
        }
        h_ranges = {
            "left": (0, page_width / 3),
            "center": (page_width / 3, 2 * page_width / 3),
            "right": (2 * page_width / 3, page_width),
        }

        def _parse_position(pos_str: str):
            """Parse 'top-left' into vertical and horizontal components."""
            parts = pos_str.split("-") if "-" in pos_str else [pos_str]
            v_pos = parts[0] if parts[0] in v_ranges else "middle"
            h_pos = parts[1] if len(parts) > 1 and parts[1] in h_ranges else "center"
            return v_pos, h_pos

        def _position_to_xy(v_pos: str, h_pos: str):
            """Convert position labels to target (x, y) coordinates."""
            v_range = v_ranges.get(v_pos, v_ranges["middle"])
            h_range = h_ranges.get(h_pos, h_ranges["center"])
            return (h_range[0] + h_range[1]) / 2, (v_range[0] + v_range[1]) / 2

        for gemini_img in gemini_images:
            position = gemini_img.get("position", "middle-center")
            v_pos, h_pos = _parse_position(position)
            # Check if this Gemini description matches a video (by position)
            matched_video = False
            for video_item in video_items:
                if video_item.get("bbox"):
                    video_y = (video_item["bbox"]["y0"] + video_item["bbox"]["y1"]) / 2
                    v_range = v_ranges.get(v_pos, v_ranges["middle"])
                    if v_range[0] <= video_y <= v_range[1]:

                        video_item["description"] = gemini_img.get("description", "")
                        matched_video = True
                        break

            if matched_video:
                continue

            # Find best matching PyMuPDF image by 2D position distance
            best_match = None
            best_distance = float("inf")

            target_x, target_y = _position_to_xy(v_pos, h_pos)

            for pymupdf_img in remaining_images:
                if pymupdf_img.get("bbox"):
                    bbox = pymupdf_img["bbox"]
                    img_x = (bbox["x0"] + bbox["x1"]) / 2
                    img_y = (bbox["y0"] + bbox["y1"]) / 2
                    # 2D Euclidean distance, normalized by page dimensions
                    dx = (img_x - target_x) / page_width
                    dy = (img_y - target_y) / page_height
                    distance = (dx ** 2 + dy ** 2) ** 0.5
                    if distance < best_distance:
                        best_distance = distance
                        best_match = pymupdf_img

            # Combine Gemini description with PyMuPDF data
            combined = {
                "type": "image",
                "description": gemini_img.get("description", ""),
                "caption": gemini_img.get("caption"),
                "position": position,
            }

            if best_match:
                if "base64_data" in best_match:
                    combined["base64_data"] = best_match["base64_data"]
                combined["format"] = best_match["format"]
                combined["bbox"] = best_match["bbox"]
                # Remove used image to prevent double-matching
                remaining_images.remove(best_match)

            matched_images.append(combined)

        # Add remaining PyMuPDF images that weren't matched to any Gemini description.
        # Filter out large images (>40% of page area) that are likely full-page
        # screenshots or background images — these duplicate the text content
        # that Gemini has already extracted and produce confusing output.
        page_area = page_height * page_width
        for remaining in remaining_images:
            bbox = remaining.get("bbox")
            if bbox:
                img_area = (bbox["x1"] - bbox["x0"]) * (bbox["y1"] - bbox["y0"])
                if page_area > 0 and img_area / page_area > 0.40:
                    # Skip large page-screenshot images — they just duplicate
                    # the text content Gemini already extracted
                    continue
            unmatched_image = {
                "type": "image",
                "description": "Unidentified image",
                "format": remaining["format"],
                "bbox": remaining["bbox"],
            }
            if "base64_data" in remaining:
                unmatched_image["base64_data"] = remaining["base64_data"]
            matched_images.append(unmatched_image)

        return matched_images, video_items

    def process_single_page(
        self, pdf_path: Path, page_num: int
    ) -> Dict:
        """Process a single page with just-in-time rendering and extraction."""
        pdf_name = pdf_path.name
        print(f"  [{pdf_name}] Page {page_num + 1}: Starting...")

        result = {
            "page_number": page_num + 1,  # 1-indexed for output
            "content": [],
            "validation": {
                "coherence_score": None,
                "coherence_issues": [],
            },
            "error": None,
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        }

        try:
            # Render page to PNG
            print(f"  [{pdf_name}] Page {page_num + 1}: Rendering PNG...")
            image_bytes = _render_single_page((str(pdf_path), page_num))
            print(f"  [{pdf_name}] Page {page_num + 1}: Rendered {len(image_bytes):,} bytes")

            # Extraction with structured output schema (guarantees valid JSON)
            # Try Flash first, fall back to Gemini 3 on exception or invalid JSON
            primary_content = []
            primary_valid = False
            flash_error = None

            try:
                print(f"  [{pdf_name}] Page {page_num + 1}: Calling Gemini for extraction...")
                primary_response, input_tokens, output_tokens = self.call_gemini_for_extraction(image_bytes)
                print(f"  [{pdf_name}] Page {page_num + 1}: Gemini returned ({input_tokens} in, {output_tokens} out tokens)")
                result["token_usage"]["input_tokens"] += input_tokens
                result["token_usage"]["output_tokens"] += output_tokens
                primary_content, primary_valid, parse_error = self.parse_json_response(primary_response)
                if not primary_valid:
                    flash_error = f"[{GEMINI_MODEL}] Invalid JSON: {parse_error}"
            except Exception as e:
                flash_error = sanitize_error(f"[{GEMINI_MODEL}] {type(e).__name__}: {e}")
                print(f"  [{pdf_name}] Page {page_num + 1}: Gemini error: {flash_error}")

            if not primary_valid:
                result["error"] = flash_error or "Unknown extraction error"
                print(f"  [{pdf_name}] Page {page_num + 1}: Failed - {result['error']}")
                return result

            # Separate images from other content (preserve indices for reinsertion)
            gemini_images = [item for item in primary_content if item.get("type") == "image"]
            other_content = [item for item in primary_content if item.get("type") != "image"]

            # Extract actual images from PDF with PyMuPDF
            pymupdf_images = self.extract_images_from_pdf_page(pdf_path, page_num)

            # Extract video links from PDF
            video_links = self.extract_video_links_from_page(pdf_path, page_num)

            # Extract hyperlinks from PDF using PyMuPDF (non-video links)
            pymupdf_hyperlinks = self.extract_hyperlinks_from_page(pdf_path, page_num)

            # Get page dimensions for position matching
            doc = fitz.open(str(pdf_path))
            page_obj = doc[page_num]
            page_rect = page_obj.rect
            page_height = page_rect.height
            page_width = page_rect.width
            page_has_text = bool(page_obj.get_text("blocks"))
            doc.close()

            # --- Filter out background / text-content images before matching ---
            # These images duplicate OCR'd content and must not appear as <img>
            # elements in the output.
            #
            # Rule 1 — full-page backgrounds: an image that covers ≥90 % of the
            # page width AND ≥85 % of the page height is a scanned-page background.
            # Including it alongside OCR'd text would show the whole document twice.
            #
            # Rule 2 — large images on fully-scanned pages: on pages where
            # PyMuPDF finds zero text blocks (the page is a pure raster scan),
            # any embedded image whose bounding box exceeds 30 % of the page area
            # is assumed to be a pre-printed text template (like a resolution body
            # or letterhead form) rather than a stand-alone visual.  Including it
            # would again duplicate the Gemini OCR output.
            page_area = page_height * page_width
            filtered_pymupdf = []
            for img in pymupdf_images:
                bbox = img.get("bbox")
                if bbox and page_area > 0:
                    w_frac = (bbox["x1"] - bbox["x0"]) / page_width
                    h_frac = (bbox["y1"] - bbox["y0"]) / page_height
                    # Rule 1
                    if w_frac >= 0.90 and h_frac >= 0.85:
                        continue
                    # Rule 2
                    if not page_has_text and (w_frac * h_frac) > 0.30:
                        continue
                filtered_pymupdf.append(img)
            pymupdf_images = filtered_pymupdf

            # Match and merge image data, separating out videos
            merged_images, video_items = self.match_images_to_descriptions(
                gemini_images, pymupdf_images, video_links, page_height, page_width
            )

            # Fallback rendering: for Gemini-described images that have no base64
            # data (PyMuPDF couldn't extract the binary), render the approximate
            # page region using pypdfium2.
            if ENABLE_IMAGE_EXTRACTION:
                for img in merged_images:
                    if "base64_data" not in img and img.get("description", "").lower() not in (
                        "unidentified image", ""
                    ):
                        # Render the region where this image should be
                        rendered = self._render_image_region_fallback(
                            pdf_path, page_num, img.get("position", "middle-center"),
                            page_height, page_width
                        )
                        if rendered:
                            img["base64_data"] = rendered
                            img["format"] = "png"
                            img["_fallback_render"] = True

            # Per-image alt text generation: send each extracted image individually
            # to Gemini for accurate descriptions. This fixes swapped/wrong alt text
            # that occurs when Gemini generates descriptions from the full page image.
            if True:
                print(f"  [{pdf_name}] Page {page_num + 1}: Generating per-image alt text...")
                alt_inp, alt_out = self.regenerate_alt_text_for_images(
                    merged_images, pdf_name, page_num
                )
                result["token_usage"]["input_tokens"] += alt_inp
                result["token_usage"]["output_tokens"] += alt_out

            # Combine content preserving Gemini's reading order for images.
            # Replace Gemini image placeholders in-place with enriched versions,
            # then append any unmatched PyMuPDF images and videos.

            merged_iter = iter(merged_images)
            combined_content = []
            for item in primary_content:
                if item.get("type") == "image":
                    # Replace with enriched version (same order as Gemini output)
                    enriched = next(merged_iter, None)
                    if enriched:
                        combined_content.append(enriched)
                else:
                    combined_content.append(item)
            # Append any remaining enriched images (unmatched PyMuPDF images)
            for remaining in merged_iter:
                combined_content.append(remaining)

            # Append videos
            combined_content.extend(video_items)

            # Merge PyMuPDF hyperlinks INTO existing content instead of appending.
            # This enriches Gemini's link objects with correct URLs and avoids
            # duplicating links already present in the content.
            combined_content = self._merge_pymupdf_links(combined_content, pymupdf_hyperlinks)


            # Post-process content (deduplication, character normalization)
            result["content"], post_process_stats = self._post_process_content(combined_content)
            result["validation"]["post_processing"] = post_process_stats

            # Coherence check (LLM-based quality assessment)
            if ENABLE_COHERENCE_CHECK:
                print(f"  [{pdf_name}] Page {page_num + 1}: Running coherence check...")
                coherence_result, coh_input_tokens, coh_output_tokens = self.check_coherence(result["content"])
                print(f"  [{pdf_name}] Page {page_num + 1}: Coherence check done (score: {coherence_result.get('coherence_score')})")
                result["token_usage"]["input_tokens"] += coh_input_tokens
                result["token_usage"]["output_tokens"] += coh_output_tokens
                result["validation"]["coherence_score"] = coherence_result.get("coherence_score")
                result["validation"]["coherence_issues"] = coherence_result.get("issues", [])

            print(f"  [{pdf_name}] Page {page_num + 1}: Complete")

        except Exception as e:
            result["error"] = str(e)
            print(f"  [{pdf_name}] Page {page_num + 1}: Exception - {e}")

        return result

    def _merge_pymupdf_links(self, content: List[Dict], pymupdf_hyperlinks: List[Dict]) -> List[Dict]:
        """Merge PyMuPDF-extracted hyperlinks into Gemini content instead of appending.

        Strategy:
        1. Enrich Gemini link objects that have broken URLs (url == text) with
           correct URLs from PyMuPDF by matching on display text.
        2. Collect all URLs already present in Gemini content (in link objects,
           paragraphs with markdown links, table cells, etc.).
        3. Only add PyMuPDF links that are truly NEW (URL not already in content).
        4. Clean garbled text from PyMuPDF link text (newline artifacts).

        This addresses:
        - Duplicate links at bottom of page
        - Broken Gemini URLs replaced with correct PyMuPDF URLs
        - Links inside tables being duplicated below
        """
        if not pymupdf_hyperlinks:
            return content

        # Build a lookup from PyMuPDF: normalized display text -> actual URL
        pymupdf_by_text = {}
        pymupdf_urls = set()
        for h in pymupdf_hyperlinks:
            url = h.get("url", "").strip()
            text = h.get("text", "").strip()
            if url:
                pymupdf_urls.add(url)
                # Normalize text for matching (collapse whitespace, lowercase)
                norm_text = " ".join(text.split()).lower()
                if norm_text:
                    pymupdf_by_text[norm_text] = url

        # Pass 1: Enrich Gemini link objects with correct URLs from PyMuPDF
        # Also collect all URLs already in the content
        urls_in_content = set()
        for item in content:
            item_type = item.get("type")

            if item_type == "link":
                link_text = item.get("text", "").strip()
                link_url = item.get("url", "").strip()
                urls_in_content.add(link_url)

                # Check if this link has a broken URL (url == text or url is not a valid URL)
                url_looks_broken = (
                    link_url == link_text
                    or (not link_url.startswith(("http://", "https://", "mailto:", "ftp://", "/"))
                        and "." not in link_url)
                )

                if url_looks_broken:
                    # Try to find correct URL from PyMuPDF
                    norm_text = " ".join(link_text.split()).lower()
                    if norm_text in pymupdf_by_text:
                        corrected_url = pymupdf_by_text[norm_text]
                        item["url"] = corrected_url
                        urls_in_content.add(corrected_url)

            elif item_type == "paragraph":
                text = item.get("text", "")
                # Extract URLs from markdown links in paragraph text
                for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', text):
                    urls_in_content.add(m.group(2))
                # Extract bare URLs
                for m in re.finditer(r'https?://\S+', text):
                    urls_in_content.add(m.group(0))

            elif item_type == "table":
                for cell in item.get("cells", []):
                    cell_text = cell.get("text", "")
                    for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', cell_text):
                        urls_in_content.add(m.group(2))
                    for m in re.finditer(r'https?://\S+', cell_text):
                        urls_in_content.add(m.group(0))

            elif item_type == "list":
                for li in item.get("items", []):
                    li_text = li.get("text", "")
                    for m in re.finditer(r'https?://\S+', li_text):
                        urls_in_content.add(m.group(0))
                    for child in li.get("children", []):
                        child_text = child.get("text", "")
                        for m in re.finditer(r'https?://\S+', child_text):
                            urls_in_content.add(m.group(0))

        # Pass 2: Only add PyMuPDF links whose URL is NOT already in content
        new_links = []
        for h in pymupdf_hyperlinks:
            url = h.get("url", "").strip()
            if not url:
                continue
            if url in urls_in_content:
                continue  # Already present, skip

            # Clean garbled text (newline artifacts from PyMuPDF extraction)
            text = h.get("text", "").strip()
            text = re.sub(r'[\n\r]+', ' ', text)  # Replace newlines with space
            text = " ".join(text.split())  # Collapse whitespace

            # Skip if text is empty or very short garbage
            if not text or len(text) < 2:
                text = url  # Use URL as display text

            new_links.append({"type": "link", "text": text, "url": url})
            urls_in_content.add(url)  # Prevent adding same URL twice

        # Append only genuinely new links
        if new_links:
            content.extend(new_links)

        return content

    def _flatten_to_text(self, content: List[Dict]) -> str:
        """Flatten extracted content to plain text for comparison."""
        texts = []
        for item in content:
            item_type = item.get("type")
            if item_type == "paragraph":
                texts.append(item.get("text", ""))
            elif item_type == "table":
                for cell in item.get("cells", []):
                    texts.append(cell.get("text", ""))
            elif item_type == "list":
                for li in item.get("items", []):
                    texts.append(li.get("text", ""))
                    for child in li.get("children", []):
                        texts.append(child.get("text", ""))
            elif item_type == "header_footer":
                texts.append(item.get("text", ""))
            elif item_type == "link":
                texts.append(item.get("text", ""))
        return " ".join(texts)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        # Normalize whitespace and case
        t1 = " ".join(text1.split()).lower()
        t2 = " ".join(text2.split()).lower()

        if not t1 and not t2:
            return 1.0
        if not t1 or not t2:
            return 0.0

        return SequenceMatcher(None, t1, t2).ratio()

    def _compare_extractions(self, primary: List[Dict], secondary: List[Dict]) -> float:
        """Compare two extractions for consistency."""
        primary_text = self._flatten_to_text(primary)
        secondary_text = self._flatten_to_text(secondary)
        return self._calculate_text_similarity(primary_text, secondary_text)

    def _normalize_ocr_characters(self, text: str) -> str:
        """
        Normalize common OCR errors and character artifacts.

        Fixes:
        - Doubled trademark symbols (™™ → ™)
        - Doubled registered symbols (®® → ®)
        - Doubled copyright symbols (©© → ©)
        - Common character confusions in specific contexts
        """
        if not text:
            return text

        # Fix doubled special symbols
        replacements = [
            ('™™', '™'),
            ('®®', '®'),
            ('©©', '©'),
            ('™ ™', '™'),  # With space between
            ('® ®', '®'),
            ('© ©', '©'),
            ('™™™', '™'),  # Triple
            ('®®®', '®'),
            ('©©©', '©'),
        ]

        result = text
        for old, new in replacements:
            result = result.replace(old, new)

        return result

    def _merge_consecutive_lists(self, content: List[Dict]) -> List[Dict]:
        """Merge consecutive lists of the same type into a single list.

        When Gemini splits a logical list into multiple single-item (or few-item)
        list objects with the same list_type (e.g., multiple consecutive ordered
        lists), merge them into one list. This commonly happens when page breaks
        or extraction artifacts fragment lists.

        Only merges lists that are directly adjacent (no content between them)
        and have the same list_type (both ordered or both unordered).
        """
        if len(content) < 2:
            return content

        result = []
        i = 0
        while i < len(content):
            item = content[i]

            if item.get("type") != "list":
                result.append(item)
                i += 1
                continue

            # Found a list — check if next items are also lists of same type
            merged_items = list(item.get("items", []))
            list_type = item.get("list_type", "unordered")

            j = i + 1
            while j < len(content):
                next_item = content[j]
                if (next_item.get("type") == "list"
                        and next_item.get("list_type") == list_type):
                    # Same type list, merge items
                    merged_items.extend(next_item.get("items", []))
                    j += 1
                else:
                    break

            if j > i + 1:
                # Merged multiple lists — create combined list
                merged_list = {
                    "type": "list",
                    "list_type": list_type,
                    "items": merged_items,
                }
                result.append(merged_list)
            else:
                result.append(item)

            i = j

        return result

    def _deduplicate_consecutive_paragraphs(self, content: List[Dict], similarity_threshold: float = 0.95) -> List[Dict]:
        """
        Remove consecutive duplicate or near-duplicate paragraphs.

        This addresses the issue where the same paragraph is extracted multiple times
        in a row (e.g., from track changes or reading order issues).

        Args:
            content: List of content items
            similarity_threshold: How similar two paragraphs must be to be considered duplicates (0-1)

        Returns:
            Deduplicated content list
        """
        if not content:
            return content

        result = []
        prev_text = None
        prev_type = None
        duplicate_count = 0

        for item in content:
            item_type = item.get("type")

            # Only deduplicate paragraphs and headings
            if item_type in ("paragraph", "heading"):
                current_text = item.get("text", "")

                # Check if this is a duplicate of the previous item
                is_duplicate = False
                if prev_type == item_type and prev_text and current_text:
                    # Normalize for comparison
                    prev_normalized = " ".join(prev_text.split()).lower()
                    curr_normalized = " ".join(current_text.split()).lower()

                    # Exact match or very high similarity
                    if prev_normalized == curr_normalized:
                        is_duplicate = True
                    elif len(prev_normalized) > 20 and len(curr_normalized) > 20:
                        # Only check similarity for longer texts to avoid false positives
                        similarity = self._calculate_text_similarity(prev_text, current_text)
                        if similarity >= similarity_threshold:
                            is_duplicate = True

                if is_duplicate:
                    duplicate_count += 1
                    continue  # Skip this duplicate

                prev_text = current_text
                prev_type = item_type
            else:
                # Non-text items reset the duplicate tracking
                prev_text = None
                prev_type = None

            result.append(item)

        if duplicate_count > 0:
            # Log for diagnostics (could be captured in validation)
            pass

        return result

    def _strip_list_number_prefix(self, text: str) -> str:
        """Strip leading NUMERIC prefixes from ordered list item text.

        When Gemini marks content as an ordered list AND includes the number
        in the text (e.g., "1. text"), the rendered HTML shows double numbering
        since <ol> auto-generates "1. 2. 3." automatically. This strips only
        numeric prefixes.

        Alphabetic ("a. text") and roman numeral ("i. text") prefixes are
        intentionally NOT stripped here, because render_json.py's
        _detect_list_style() needs them to set the correct <ol type="a"> or
        <ol type="i"> attribute. render_json.py's _strip_list_prefix() will
        remove them during rendering.

        Patterns stripped:
        - "1. text" or "1) text" or "(1) text" (numeric only)

        Only strips if the prefix is followed by a space and more text.
        """
        if not text:
            return text

        # Numeric only: "1.", "1)", "(1)"
        stripped = re.sub(r'^\s*\(?\d{1,3}\)?[\.\)]\s+', '', text)
        if stripped != text:
            return stripped

        return text

    def _strip_spurious_markdown(self, text: str) -> str:
        """Strip markdown bold/italic markers from text where they shouldn't be.

        Used for table cells and headings where Gemini adds ** or * around text
        that should be plain. The renderer (render_json.py) handles formatting
        separately, so having markdown in the JSON just causes duplicate
        formatting or literal asterisks in output.

        Examples:
            "**Members Present**" -> "Members Present"
            "**Page No.** 1 of 12" -> "Page No. 1 of 12"
            "***Bold Italic***" -> "Bold Italic"
        """
        if not text or '**' not in text and '*' not in text:
            return text

        # Remove bold+italic: ***text***
        result = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text, flags=re.DOTALL)
        # Remove bold: **text**
        result = re.sub(r'\*{2}(.+?)\*{2}', r'\1', result, flags=re.DOTALL)
        # Remove italic: *text* (but not ** which was already handled)
        result = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', result, flags=re.DOTALL)
        # Strip any remaining orphan asterisks at start/end
        result = re.sub(r'^\s*\*{1,3}\s+', '', result)
        result = re.sub(r'\s+\*{1,3}\s*$', '', result)
        return result

    def _post_process_content(self, content: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Apply all post-processing steps to extracted content.

        Returns:
            Tuple of (processed_content, post_processing_stats)
        """
        stats = {
            "duplicates_removed": 0,
            "characters_normalized": 0,
        }

        if not content:
            return content, stats

        # Count items before deduplication
        original_count = len(content)

        # Step 1: Normalize OCR characters in all text fields
        for item in content:
            item_type = item.get("type")

            if item_type in ("paragraph", "heading"):
                original_text = item.get("text", "")
                normalized_text = self._normalize_ocr_characters(original_text)
                if normalized_text != original_text:
                    item["text"] = normalized_text
                    stats["characters_normalized"] += 1

            elif item_type == "table":
                for cell in item.get("cells", []):
                    original_text = cell.get("text", "")
                    normalized_text = self._normalize_ocr_characters(original_text)
                    if normalized_text != original_text:
                        cell["text"] = normalized_text
                        stats["characters_normalized"] += 1

            elif item_type == "image":
                for field in ["description", "caption"]:
                    if field in item and item[field]:
                        original_text = item[field]
                        normalized_text = self._normalize_ocr_characters(original_text)
                        if normalized_text != original_text:
                            item[field] = normalized_text
                            stats["characters_normalized"] += 1

            elif item_type == "list":
                for li in item.get("items", []):
                    original_text = li.get("text", "")
                    normalized_text = self._normalize_ocr_characters(original_text)
                    if normalized_text != original_text:
                        li["text"] = normalized_text
                        stats["characters_normalized"] += 1
                    for child in li.get("children", []):
                        original_text = child.get("text", "")
                        normalized_text = self._normalize_ocr_characters(original_text)
                        if normalized_text != original_text:
                            child["text"] = normalized_text
                            stats["characters_normalized"] += 1

            elif item_type == "header_footer":
                original_text = item.get("text", "")
                normalized_text = self._normalize_ocr_characters(original_text)
                if normalized_text != original_text:
                    item["text"] = normalized_text
                    stats["characters_normalized"] += 1

            elif item_type == "link":
                original_text = item.get("text", "")
                normalized_text = self._normalize_ocr_characters(original_text)
                if normalized_text != original_text:
                    item["text"] = normalized_text
                    stats["characters_normalized"] += 1

        # Step 2: Strip spurious markdown from table cells, headings, and list items
        for item in content:
            item_type = item.get("type")
            if item_type == "table":
                for cell in item.get("cells", []):
                    cell["text"] = self._strip_spurious_markdown(cell.get("text", ""))
            elif item_type == "heading":
                item["text"] = self._strip_spurious_markdown(item.get("text", ""))
            elif item_type == "list":
                for li in item.get("items", []):
                    li["text"] = self._strip_spurious_markdown(li.get("text", ""))
                    for child in li.get("children", []):
                        child["text"] = self._strip_spurious_markdown(child.get("text", ""))

        # Step 3: Strip duplicate numbering from ordered list items
        # Gemini often includes the list number in the text (e.g., "1. text")
        # while also marking the list as ordered — causing double numbering.
        for item in content:
            if item.get("type") == "list" and item.get("list_type") == "ordered":
                for li in item.get("items", []):
                    li["text"] = self._strip_list_number_prefix(li.get("text", ""))

        # Step 4: Merge consecutive single-item lists of the same type
        content = self._merge_consecutive_lists(content)

        # Step 5: Remove "page intentionally left blank" boilerplate
        _blank_page_re = re.compile(
            r'^\s*\*{0,3}\s*(?:this\s+)?page\s+(?:(?:was\s+)?(?:left\s+)?intentionally\s+(?:left\s+)?blank'
            r'|(?:left\s+)?(?:blank\s+)?intentionally)\s*\*{0,3}\s*\.?\s*$',
            re.IGNORECASE
        )
        content = [
            item for item in content
            if not (
                item.get("type") in ("paragraph", "heading")
                and _blank_page_re.match(item.get("text", ""))
            )
        ]

        # Step 6: Fix broken hyperlinks
        content, links_fixed = self._fix_broken_hyperlinks(content)
        stats["broken_links_fixed"] = links_fixed

        # Step 7: Deduplicate consecutive paragraphs
        content = self._deduplicate_consecutive_paragraphs(content)
        stats["duplicates_removed"] = original_count - len(content)

        # Step 8: Convert asterisk-bullet paragraphs to unordered list items
        content = self._convert_asterisk_bullet_paragraphs(content)

        return content, stats

    def _convert_asterisk_bullet_paragraphs(self, content: list) -> list:
        """Convert paragraphs starting with '* ' (asterisk bullet) to unordered list items.

        When Gemini uses asterisk markers instead of proper list objects (e.g., a
        paragraph with text "* Item text"), convert to {type: list, list_type: unordered}.
        Consecutive asterisk-bullet paragraphs are merged into a single list.

        Only converts paragraphs where text starts with "* " (asterisk + space).
        Does NOT convert paragraphs starting with "**" (markdown bold).
        """
        result = []
        i = 0
        while i < len(content):
            item = content[i]
            if item.get("type") == "paragraph":
                text = item.get("text", "")
                # Check for "* " bullet (not "**" bold)
                if text.startswith("* ") and not text.startswith("**"):
                    # Collect consecutive asterisk-bullet paragraphs
                    list_items = []
                    while i < len(content):
                        curr = content[i]
                        if curr.get("type") == "paragraph":
                            t = curr.get("text", "")
                            if t.startswith("* ") and not t.startswith("**"):
                                list_items.append({"text": t[2:].strip()})
                                i += 1
                                continue
                        break
                    result.append({
                        "type": "list",
                        "list_type": "unordered",
                        "items": list_items,
                    })
                    continue
            result.append(item)
            i += 1
        return result

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if a string looks like a valid URL or URL-like reference.

        Returns True for:
        - http:// or https:// URLs
        - mailto: links
        - tel: links
        - ftp:// links
        - Bare domains with dots (e.g., www.example.com, example.gov)
        - Paths starting with / (root-relative)
        """
        if not url:
            return False
        url = url.strip()
        # Protocol-prefixed URLs
        if re.match(r'^(?:https?://|mailto:|tel:|ftp://)', url, re.IGNORECASE):
            return True
        # Root-relative paths
        if url.startswith('/') and len(url) > 1:
            return True
        # Bare domains (must have a dot and a valid-looking TLD)
        if re.match(r'^[\w.-]+\.\w{2,}(?:/\S*)?$', url):
            return True
        return False

    @staticmethod
    def _fix_url_protocol(url: str) -> str:
        """Add missing protocol to bare domain URLs.

        www.example.com -> https://www.example.com
        example.gov/path -> https://example.gov/path
        """
        url = url.strip()
        if not url:
            return url
        # Already has protocol
        if re.match(r'^(?:https?://|mailto:|tel:|ftp://|/)', url, re.IGNORECASE):
            return url
        # Bare domain with dot — add https://
        if re.match(r'^[\w.-]+\.\w{2,}(?:/\S*)?$', url):
            return 'https://' + url
        return url

    def _fix_broken_hyperlinks(self, content: List[Dict]) -> tuple:
        """Fix or remove broken hyperlinks in extracted content.

        Handles:
        1. Link items where URL == text and URL is not valid: convert to paragraph
        2. Link items with clearly invalid URLs (no dots, contains spaces, etc.):
           convert to paragraph
        3. Add missing protocol to bare domain URLs (www.x.com -> https://www.x.com)
        4. Merge consecutive link items with the same URL
        5. Fix bare-domain URLs in markdown links within paragraph text

        Returns:
            Tuple of (processed_content, count_of_links_fixed)
        """
        if not content:
            return content, 0

        fixed_count = 0
        result = []

        for item in content:
            if item.get("type") != "link":
                result.append(item)
                continue

            link_text = item.get("text", "").strip()
            link_url = item.get("url", "").strip()

            # Fix 1: Add missing protocol to bare domain URLs
            if link_url and not re.match(r'^(?:https?://|mailto:|tel:|ftp://|/)', link_url, re.IGNORECASE):
                fixed_url = self._fix_url_protocol(link_url)
                if fixed_url != link_url:
                    item["url"] = fixed_url
                    link_url = fixed_url
                    fixed_count += 1

            # Fix 2: Check if URL is valid after potential protocol fix
            url_is_valid = self._is_valid_url(link_url)

            # Fix 3: If URL == text and URL is not valid, convert to paragraph
            url_equals_text = (
                link_url == link_text
                or link_url.rstrip('.') == link_text.rstrip('.')
                or link_url.lower().strip() == link_text.lower().strip()
            )

            if url_equals_text and not url_is_valid:
                # Convert broken link to paragraph (preserve the text)
                result.append({"type": "paragraph", "text": link_text})
                fixed_count += 1
                continue

            # Fix 4: If URL is clearly not valid (no protocol, no dots, has spaces), convert to paragraph
            if not url_is_valid:
                result.append({"type": "paragraph", "text": link_text})
                fixed_count += 1
                continue

            # Link is valid — keep it
            result.append(item)

        # Fix 5: Merge consecutive link items with the same URL
        merged = []
        i = 0
        while i < len(result):
            item = result[i]
            if item.get("type") == "link":
                # Look ahead for consecutive links with the same URL
                link_url = item.get("url", "")
                texts = [item.get("text", "")]
                j = i + 1
                while j < len(result) and result[j].get("type") == "link" and result[j].get("url", "") == link_url:
                    texts.append(result[j].get("text", ""))
                    j += 1
                if j > i + 1:
                    # Merge texts
                    item["text"] = " ".join(t for t in texts if t)
                    fixed_count += (j - i - 1)
                    merged.append(item)
                    i = j
                else:
                    merged.append(item)
                    i += 1
            else:
                merged.append(item)
                i += 1

        # Fix 6: Fix bare domain URLs in markdown links within paragraph text
        for item in merged:
            if item.get("type") in ("paragraph", "heading"):
                text = item.get("text", "")
                # Find markdown links [text](url) where url is a bare domain
                def _fix_md_link_url(m):
                    md_text = m.group(1)
                    md_url = m.group(2)
                    if not re.match(r'^(?:https?://|mailto:|tel:|ftp://|/)', md_url, re.IGNORECASE):
                        if re.match(r'^[\w.-]+\.\w{2,}(?:/\S*)?$', md_url):
                            return f'[{md_text}](https://{md_url})'
                    return m.group(0)

                new_text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _fix_md_link_url, text)
                if new_text != text:
                    item["text"] = new_text
                    fixed_count += 1

        return merged, fixed_count

    def _deduplicate_cross_page_content(self, pages: List[Dict]) -> List[Dict]:
        """Remove content that repeats identically across multiple pages.

        Targets:
        - Header/footer tables that appear on every page (e.g., document metadata tables
          with page numbers). Keeps first occurrence, removes subsequent copies.
        - Repeated header_footer items across pages. Keeps first occurrence.

        Does NOT remove content that differs between pages (different page numbers
        are tolerated — we normalize "Page X of Y" before comparison).
        """
        if len(pages) < 2:
            return pages

        def _normalize_for_dedup(text: str) -> str:
            """Normalize text for dedup comparison, ignoring page numbers and markdown."""
            t = " ".join(text.split()).lower()
            # Strip markdown bold/italic
            t = re.sub(r'\*{1,3}', '', t)
            # Normalize page numbers: "page 3 of 12" -> "page N of N"
            t = re.sub(r'page\s*(?:no\.?)?\s*\d+\s*(?:of\s*\d+)?', 'page N', t)
            # Also normalize standalone numbers that look like page numbers
            t = re.sub(r'\b\d+\s*of\s*\d+\b', 'N of N', t)
            return t.strip()

        def _table_fingerprint(item: Dict) -> str:
            """Create a fingerprint for a table, ignoring page number variations."""
            cells = item.get("cells", [])
            parts = []
            for cell in sorted(cells, key=lambda c: (c.get("row_start", 0), c.get("column_start", 0))):
                parts.append(_normalize_for_dedup(cell.get("text", "")))
            return "|".join(parts)

        def _header_footer_fingerprint(item: Dict) -> str:
            """Create a fingerprint for a header/footer item."""
            return _normalize_for_dedup(item.get("text", ""))

        # Collect fingerprints from page 1 to identify repeating elements
        first_page_content = pages[0].get("content", []) if not pages[0].get("error") else []

        seen_table_fps = set()
        seen_hf_fps = set()
        for item in first_page_content:
            if item.get("type") == "table":
                seen_table_fps.add(_table_fingerprint(item))
            elif item.get("type") == "header_footer":
                seen_hf_fps.add(_header_footer_fingerprint(item))

        # Check which fingerprints repeat on page 2 (if exists)
        repeating_table_fps = set()
        repeating_hf_fps = set()
        if len(pages) > 1 and not pages[1].get("error"):
            page2_content = pages[1].get("content", [])
            for item in page2_content:
                if item.get("type") == "table":
                    fp = _table_fingerprint(item)
                    if fp in seen_table_fps:
                        repeating_table_fps.add(fp)
                elif item.get("type") == "header_footer":
                    fp = _header_footer_fingerprint(item)
                    if fp in seen_hf_fps:
                        repeating_hf_fps.add(fp)

        if not repeating_table_fps and not repeating_hf_fps:
            return pages  # Nothing repeats, no changes needed

        # Remove repeating items from pages 2+ (keep page 1 intact)
        for page_idx in range(1, len(pages)):
            page = pages[page_idx]
            if page.get("error"):
                continue
            content = page.get("content", [])
            filtered = []
            for item in content:
                if item.get("type") == "table":
                    fp = _table_fingerprint(item)
                    if fp in repeating_table_fps:
                        continue  # Skip repeated table
                elif item.get("type") == "header_footer":
                    fp = _header_footer_fingerprint(item)
                    if fp in repeating_hf_fps:
                        continue  # Skip repeated header/footer
                filtered.append(item)
            page["content"] = filtered

        return pages

    def _normalize_heading_hierarchy(self, pages: List[Dict]) -> List[Dict]:
        """Normalize heading hierarchy across all pages of a document.

        Fixes common Gemini extraction issues where:
        1. All headings are the same level (e.g., all H2) — infers hierarchy
        2. Heading levels reset across page boundaries
        3. Levels skip (H1 → H4 with no H2/H3)
        4. Series headings at inconsistent levels (e.g., Priority #1 at H4,
           Priority #2 at H2)
        """
        if len(pages) < 1:
            return pages

        # Collect all headings with their page/index references
        all_headings = []
        for page_idx, page in enumerate(pages):
            if page.get("error"):
                continue
            for item_idx, item in enumerate(page.get("content", [])):
                if item.get("type") == "heading":
                    all_headings.append({
                        "page_idx": page_idx,
                        "item_idx": item_idx,
                        "level": item.get("level", 2),
                        "text": item.get("text", ""),
                    })

        if len(all_headings) < 2:
            return pages

        # Step 1: Fix series headings — headings that follow a numbered/lettered
        # pattern should all be at the same level as the first in the series.
        # E.g., "Priority #1" at H4, "Priority #2" at H2 → make #2 also H4
        self._fix_series_heading_levels(pages, all_headings)

        # Refresh heading data after series fix
        all_headings = []
        for page_idx, page in enumerate(pages):
            if page.get("error"):
                continue
            for item_idx, item in enumerate(page.get("content", [])):
                if item.get("type") == "heading":
                    all_headings.append({
                        "page_idx": page_idx,
                        "item_idx": item_idx,
                        "level": item.get("level", 2),
                        "text": item.get("text", ""),
                    })

        # Step 2: Check if headings are "flat" — most at the same level
        level_counts = {}
        for h in all_headings:
            level_counts[h["level"]] = level_counts.get(h["level"], 0) + 1

        total_headings = len(all_headings)
        max_level_count = max(level_counts.values())
        dominant_level = max(level_counts, key=level_counts.get)

        # Only normalize flat hierarchies if >70% same level and not H1-dominant
        is_flat = (max_level_count / total_headings) > 0.70

        if is_flat and dominant_level >= 2:
            has_h1 = any(h["level"] == 1 for h in all_headings)

            if not has_h1:
                # Promote the very first heading to H1
                first = all_headings[0]
                pages[first["page_idx"]]["content"][first["item_idx"]]["level"] = 1

            # Step 2a: Track heading context across pages for demotion.
            # If a page establishes a deeper heading level (e.g., H3 under H2),
            # subsequent pages with only dominant-level headings should continue
            # at the deeper level until a clear "new section" heading appears.
            deepest_child_level = None  # Deepest non-dominant level seen so far
            parent_level = None  # Level of the heading that "owns" the child

            for page_idx, page in enumerate(pages):
                if page.get("error"):
                    continue
                content = page.get("content", [])
                page_headings = [
                    (idx, item) for idx, item in enumerate(content)
                    if item.get("type") == "heading"
                ]
                if not page_headings:
                    continue

                # Check if this page established a parent→child relationship
                for i, (h_idx, h) in enumerate(page_headings):
                    h_level = h.get("level", 2)
                    if h_level == dominant_level and i + 1 < len(page_headings):
                        next_level = page_headings[i + 1][1].get("level", 2)
                        if next_level > dominant_level:
                            # Found parent (dominant) → child relationship
                            parent_level = dominant_level
                            deepest_child_level = next_level

                # If we have an active child context and this page has only
                # dominant-level headings, check if they should be demoted
                if deepest_child_level and parent_level:
                    all_at_dominant = all(
                        h.get("level") == dominant_level
                        for _, h in page_headings
                    )
                    if all_at_dominant and len(page_headings) >= 1:
                        # All headings at dominant level — these may be
                        # continuations of the child context. Demote them.
                        for h_idx, h in page_headings:
                            content[h_idx]["level"] = deepest_child_level

                # Check if any heading at dominant level appears with
                # non-heading content before and after it (section boundary)
                # If so, reset the child context
                for i, (h_idx, h) in enumerate(page_headings):
                    h_level = h.get("level", 2)
                    if h_level < dominant_level:
                        # Found a heading shallower than dominant — reset context
                        deepest_child_level = None
                        parent_level = None

            # Step 2b: Per-page demotion for remaining same-level headings
            for page_idx, page in enumerate(pages):
                if page.get("error"):
                    continue
                content = page.get("content", [])
                page_headings = [
                    (idx, item) for idx, item in enumerate(content)
                    if item.get("type") == "heading"
                ]
                if len(page_headings) < 2:
                    continue

                first_h_idx, first_h = page_headings[0]
                first_level = first_h.get("level", 2)

                same_level_count = sum(
                    1 for _, h in page_headings
                    if h.get("level") == first_level
                )
                if same_level_count == len(page_headings) and same_level_count >= 3:
                    # Many headings at same level on one page — demote all but first
                    for h_idx, h in page_headings[1:]:
                        if h.get("level") == first_level:
                            content[h_idx]["level"] = first_level + 1

        # Step 3: Fix skipped levels across the document
        # Re-collect headings after modifications
        all_headings = []
        for page_idx, page in enumerate(pages):
            if page.get("error"):
                continue
            for item_idx, item in enumerate(page.get("content", [])):
                if item.get("type") == "heading":
                    all_headings.append({
                        "page_idx": page_idx,
                        "item_idx": item_idx,
                        "level": item.get("level", 2),
                        "text": item.get("text", ""),
                    })

        prev_level = 0
        for h in all_headings:
            page = pages[h["page_idx"]]
            item = page["content"][h["item_idx"]]
            current_level = item.get("level", 2)

            if prev_level > 0 and current_level > prev_level + 1:
                item["level"] = prev_level + 1
                current_level = prev_level + 1

            prev_level = current_level

        return pages

    def _fix_series_heading_levels(self, pages: List[Dict], all_headings: List[Dict]):
        """Fix heading levels for series headings (e.g., Priority #1, #2, #3).

        When headings follow a numbered/lettered pattern, they should all be at
        the same level as the first one in the series. This fixes the common
        issue where Gemini extracts the first item at the correct level but
        resets to H2 on subsequent pages.
        """
        # Build series patterns: extract numbering from heading text
        # Match patterns like "Priority #1", "1. Introduction", "Section A", etc.
        series_pattern = re.compile(
            r'^(.*?)\s*'  # prefix text
            r'(?:'
            r'#?\s*(\d+)'  # numbered: #1, 1, etc.
            r'|([a-zA-Z])(?=\s*[.:\)])'  # lettered: a., A), etc.
            r')'
        )

        # Group headings by their "series prefix" (text without the number/letter)
        series_groups = {}
        for h in all_headings:
            text = h["text"].strip()
            m = series_pattern.match(text)
            if m:
                prefix = m.group(1).strip().lower()
                if prefix and len(prefix) > 2:
                    if prefix not in series_groups:
                        series_groups[prefix] = []
                    series_groups[prefix].append(h)

        # For each series with 2+ members, normalize levels to match the first
        for prefix, group in series_groups.items():
            if len(group) < 2:
                continue

            # Use the level of the first heading in the series
            target_level = group[0]["level"]

            for h in group[1:]:
                if h["level"] != target_level:
                    page = pages[h["page_idx"]]
                    page["content"][h["item_idx"]]["level"] = target_level

    def _merge_cross_page_content(self, pages: List[Dict]) -> List[Dict]:
        """Merge content that was split across page boundaries.

        Fixes:
        1. Split paragraphs: last paragraph on page N doesn't end with
           sentence-ending punctuation → merge with first paragraph on page N+1
        2. Split lists: list at end of page N + list of same type at start
           of page N+1 → merge into single list
        3. Split tables: table at end of page N + table with matching column
           count at start of page N+1 → merge into single table

        Only merges when there's strong evidence of a split (not just any
        consecutive same-type content).
        """
        if len(pages) < 2:
            return pages

        for page_idx in range(len(pages) - 1):
            curr_page = pages[page_idx]
            next_page = pages[page_idx + 1]

            if curr_page.get("error") or next_page.get("error"):
                continue

            curr_content = curr_page.get("content", [])
            next_content = next_page.get("content", [])

            if not curr_content or not next_content:
                continue

            # Get last non-header_footer item on current page
            last_item = None
            last_idx = None
            for i in range(len(curr_content) - 1, -1, -1):
                if curr_content[i].get("type") != "header_footer":
                    last_item = curr_content[i]
                    last_idx = i
                    break

            # Get first non-header_footer item on next page
            first_item = None
            first_idx = None
            for i in range(len(next_content)):
                if next_content[i].get("type") != "header_footer":
                    first_item = next_content[i]
                    first_idx = i
                    break

            if last_item is None or first_item is None:
                continue

            # Case 1: Split paragraphs
            if (last_item.get("type") == "paragraph"
                    and first_item.get("type") == "paragraph"):
                last_text = last_item.get("text", "").rstrip()
                first_text = first_item.get("text", "").lstrip()

                if last_text and first_text:
                    # Check if the paragraph was likely split:
                    # - Last text doesn't end with sentence-ending punctuation
                    # - First text doesn't start with a capital letter (continuation)
                    #   OR first text starts with lowercase
                    ends_mid_sentence = (
                        last_text
                        and not last_text[-1] in '.!?:;"\u201d'
                        and not last_text.endswith('...')
                    )
                    starts_continuation = (
                        first_text
                        and (first_text[0].islower()
                             or first_text[0] in ',-;')
                    )

                    if ends_mid_sentence or starts_continuation:
                        # Merge: append first paragraph text to last paragraph
                        # Add a space if last doesn't end with hyphen (word break)
                        if last_text.endswith('-'):
                            # Word break — join without space
                            merged_text = last_text[:-1] + first_text
                        else:
                            merged_text = last_text + " " + first_text
                        curr_content[last_idx]["text"] = merged_text
                        # Remove the first item from next page
                        next_content.pop(first_idx)

            # Case 2: Split lists
            elif (last_item.get("type") == "list"
                  and first_item.get("type") == "list"
                  and last_item.get("list_type") == first_item.get("list_type")):
                # Merge list items from next page into current page's list
                last_items = last_item.get("items", [])
                first_items = first_item.get("items", [])
                if last_items and first_items:
                    last_item["items"] = last_items + first_items
                    next_content.pop(first_idx)

            # Case 3: Split tables
            elif (last_item.get("type") == "table"
                  and first_item.get("type") == "table"):
                last_cells = last_item.get("cells", [])
                first_cells = first_item.get("cells", [])
                if last_cells and first_cells:
                    # Check if tables have matching column counts
                    last_max_col = max(
                        (c.get("column_start", 0) + c.get("num_columns", 1))
                        for c in last_cells
                    )
                    first_max_col = max(
                        (c.get("column_start", 0) + c.get("num_columns", 1))
                        for c in first_cells
                    )

                    if last_max_col == first_max_col:
                        # Same column structure — merge by offsetting row numbers
                        last_max_row = max(
                            (c.get("row_start", 0) + c.get("num_rows", 1))
                            for c in last_cells
                        )
                        # Skip first row of next table if it looks like a
                        # repeated header (same text as first row of current table)
                        first_row_cells = [
                            c for c in first_cells
                            if c.get("row_start", 0) == 0
                        ]
                        last_first_row = [
                            c for c in last_cells
                            if c.get("row_start", 0) == 0
                        ]
                        # Check if first row of next table matches first row of current
                        skip_first_row = False
                        if first_row_cells and last_first_row:
                            first_texts = sorted(
                                c.get("text", "").strip().lower()
                                for c in first_row_cells
                            )
                            last_first_texts = sorted(
                                c.get("text", "").strip().lower()
                                for c in last_first_row
                            )
                            if first_texts == last_first_texts:
                                skip_first_row = True

                        row_offset = 1 if skip_first_row else 0
                        for cell in first_cells:
                            if skip_first_row and cell.get("row_start", 0) == 0:
                                continue
                            new_cell = dict(cell)
                            new_cell["row_start"] = (
                                cell.get("row_start", 0) - row_offset + last_max_row
                            )
                            last_cells.append(new_cell)

                        last_item["cells"] = last_cells
                        next_content.pop(first_idx)

            # Update page content references
            curr_page["content"] = curr_content
            next_page["content"] = next_content

        return pages

    def _calculate_pdf_metrics(self, pages: List[Dict]) -> Dict:
        """Calculate aggregate quality metrics for a PDF.

        Confidence scoring now uses:
        - Coherence score (1-10 from LLM) - primary metric
        - Re-extraction consistency - secondary metric
        - PyMuPDF similarity is only kept for diagnostic purposes
        """
        total_pages = len(pages)
        pages_with_errors = sum(1 for p in pages if p.get("error"))

        # Collect scores for averaging
        coherence_scores = [
            p["validation"].get("coherence_score")
            for p in pages
            if p["validation"].get("coherence_score") is not None
        ]

        # Classify pages by confidence based on coherence score
        high_confidence = 0
        medium_confidence = 0
        low_confidence = 0

        for page in pages:
            if page.get("error"):
                continue

            # Get coherence score (1-10 scale)
            coherence = page["validation"].get("coherence_score")
            if coherence is None:
                coherence = 8  # Default assumption

            if coherence >= 9:
                high_confidence += 1
            elif coherence >= 7:
                medium_confidence += 1
            else:
                low_confidence += 1

        return {
            "pages_successful": total_pages - pages_with_errors,
            "pages_failed": pages_with_errors,
            "pages_high_confidence": high_confidence,
            "pages_medium_confidence": medium_confidence,
            "pages_low_confidence": low_confidence,
            "avg_coherence_score": round(sum(coherence_scores) / len(coherence_scores), 2) if coherence_scores else None,
        }

    def get_completed_pdfs(self) -> set:
        """Get set of PDF IDs that have already been processed."""
        completed = set()
        if OUTPUT_FOLDER.exists():
            for json_file in OUTPUT_FOLDER.glob("*.json"):
                if not json_file.name.startswith("_"):
                    completed.add(json_file.stem)
        return completed

    def _get_pdf_files(self) -> List[Tuple[str, Path]]:
        """Get list of (doc_id, pdf_path) tuples to process.

        Reads source.pdf from data/{doc_id}/ folders. If a test file list
        is configured, only returns PDFs matching that list.
        """
        # Parse test file list for filtering
        target_ids = []
        if TEST_FILE_LIST:
            target_ids = parse_test_file_list(TEST_FILE_LIST)
            if target_ids:
                print(f"Filtering to {len(target_ids)} documents from test file list")

        pdf_files = []
        if target_ids:
            # Only look for specific document IDs
            for doc_id in target_ids:
                pdf_path = DATA_FOLDER / doc_id / "source.pdf"
                if pdf_path.exists():
                    pdf_files.append((doc_id, pdf_path))
                else:
                    # Try partial match (directory name may differ slightly)
                    matches = list(DATA_FOLDER.glob(f"{doc_id}*/source.pdf"))
                    if matches:
                        matched_id = matches[0].parent.name
                        pdf_files.append((matched_id, matches[0]))
                    else:
                        print(f"  Warning: PDF not found for '{doc_id}'")
        else:
            # Process all PDFs in data folder
            for source_pdf in sorted(DATA_FOLDER.glob("*/source.pdf")):
                doc_id = source_pdf.parent.name
                pdf_files.append((doc_id, source_pdf))

        return pdf_files

    def process_all_pdfs(self):
        """Process all PDFs with flat page-level concurrency using multiprocessing.Pool."""
        # Create output directories
        OUTPUT_FOLDER.mkdir(exist_ok=True)
        REPORTS_FOLDER.mkdir(exist_ok=True)

        # Get list of PDFs (filtered by test file list if configured)
        pdf_files = self._get_pdf_files()
        self.stats["total_pdfs"] = len(pdf_files)

        # Check for already completed PDFs (resume capability)
        completed = self.get_completed_pdfs()
        pending_pdfs = [(doc_id, p) for doc_id, p in pdf_files if doc_id not in completed]

        print(f"Found {len(pdf_files)} PDFs total")
        print(f"Already completed: {len(completed)}")
        print(f"Pending: {len(pending_pdfs)}")

        # Collect all pages to process across all PDFs
        all_page_tasks = []
        pdf_info = {}  # Store PDF metadata for building results later
        for doc_id, pdf_path in pending_pdfs:
            try:
                doc = pdfium.PdfDocument(str(pdf_path))
                page_count = len(doc)
                doc.close()
            except Exception as e:
                print(f"  Warning: Skipping '{doc_id}' - failed to open PDF: {sanitize_error(str(e))}")
                continue
            pdf_info[doc_id] = {
                "pdf_path": pdf_path,
                "pdf_filename": pdf_path.name,
                "total_pages": page_count,
            }
            for page_num in range(page_count):
                all_page_tasks.append((str(pdf_path), page_num, doc_id))

        print(f"Processing {len(all_page_tasks)} pages with {MAX_WORKERS} workers")
        print()

        extraction_start_time = datetime.now(tz=timezone.utc)

        # Process all pages using multiprocessing.Pool
        results_by_pdf = {}
        all_failed_pages = []
        parallel_processing = False
        pages_received = {}  # Track how many pages received per PDF
        saved_pdfs = set()  # Track PDFs already saved (to avoid double-counting stats)

        with Pool(processes=MAX_WORKERS) as pool:
            parallel_processing = True
            for doc_id, page_num, page_result in tqdm(
                pool.imap_unordered(_process_page_worker, all_page_tasks),
                total=len(all_page_tasks),
                desc="Processing pages",
            ):
                if doc_id not in results_by_pdf:
                    results_by_pdf[doc_id] = {}
                results_by_pdf[doc_id][page_num] = page_result

                # Track failed pages for retry
                if page_result.get("error"):
                    pdf_path_str = str(pdf_info[doc_id]["pdf_path"])
                    all_failed_pages.append((pdf_path_str, page_num, doc_id))

                # Update token stats
                token_usage = page_result.get("token_usage", {})
                self.stats["total_input_tokens"] += token_usage.get("input_tokens", 0)
                self.stats["total_output_tokens"] += token_usage.get("output_tokens", 0)

                # Write JSON immediately when all pages of a PDF are received
                pages_received[doc_id] = pages_received.get(doc_id, 0) + 1
                if pages_received[doc_id] == pdf_info[doc_id]["total_pages"]:
                    # Skip analytics if this PDF has failed pages — they'll be
                    # retried and analytics recorded on the final re-save
                    has_failures = any(d == doc_id for _, _, d in all_failed_pages)
                    result = self._build_and_save_pdf_result(
                        doc_id, pdf_info[doc_id], results_by_pdf[doc_id],
                        extraction_start_time, parallel_processing,
                        record_analytics=not has_failures,
                    )
                    saved_pdfs.add(doc_id)

        # Retry failed pages at end of run
        if all_failed_pages:
            print(f"\n{'='*60}")
            print(f"RETRYING {len(all_failed_pages)} FAILED PAGES")
            print(f"{'='*60}")
            retry_doc_ids = {doc_id for _, _, doc_id in all_failed_pages}
            self._retry_failed_pages(all_failed_pages, results_by_pdf)

            # Re-save PDFs that had retried pages (results may have improved)
            for doc_id in retry_doc_ids:
                if doc_id in pdf_info:
                    # Reset stats for this PDF before re-saving so they aren't double-counted
                    self._reset_pdf_stats(doc_id, saved_pdfs)
                    self._build_and_save_pdf_result(
                        doc_id, pdf_info[doc_id], results_by_pdf[doc_id],
                        extraction_start_time, parallel_processing,
                    )

        # Build all_results from saved JSON files for summary report
        all_results = []
        for pdf_id in pdf_info:
            output_path = OUTPUT_FOLDER / f"{pdf_id}.json"
            if output_path.exists():
                with open(output_path, "r", encoding="utf-8") as f:
                    all_results.append(json.load(f))

        # Generate summary report
        self._generate_summary_report(all_results)

        # Flush analytics
        if self._analytics:
            try:
                self._analytics.flush()
            except Exception as exc:
                logging.getLogger(__name__).debug("Analytics flush failed (non-fatal): %s", exc)

        return all_results

    def _build_and_save_pdf_result(self, pdf_id, info, page_results_dict,
                                    extraction_start_time, parallel_processing,
                                    record_analytics=True):
        """Post-process pages, save JSON/HTML, record analytics, and update stats.

        Returns the result dict for the PDF.
        """
        # Convert dict to sorted list by page number
        # Deep-copy so post-processing mutations don't affect results_by_pdf
        # (needed for correct re-save after retries)
        pages = [copy.deepcopy(page_results_dict[i]) for i in sorted(page_results_dict.keys())]

        # Cross-page deduplication: remove repeated headers/footers/tables
        pages = self._deduplicate_cross_page_content(pages)

        # Cross-page content merging: fix paragraphs/lists/tables split at page boundaries
        pages = self._merge_cross_page_content(pages)

        # Heading hierarchy normalization: fix flat/inconsistent heading levels
        pages = self._normalize_heading_hierarchy(pages)

        result = {
            "pdf_id": pdf_id,
            "source_path": str(info["pdf_path"]),
            "total_pages": info["total_pages"],
            "extraction_timestamp": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "pages": pages,
        }

        # Calculate quality metrics
        result["quality_metrics"] = self._calculate_pdf_metrics(pages)

        # Calculate token usage for this PDF
        pdf_input_tokens = sum(p.get("token_usage", {}).get("input_tokens", 0) for p in pages)
        pdf_output_tokens = sum(p.get("token_usage", {}).get("output_tokens", 0) for p in pages)
        result["token_usage"] = {
            "input_tokens": pdf_input_tokens,
            "output_tokens": pdf_output_tokens,
        }

        # Save individual PDF result
        output_path = OUTPUT_FOLDER / f"{pdf_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Generate per-document HTML immediately after JSON
        html_path = OUTPUT_FOLDER / f"{pdf_id}.html"
        generate_document_html(result, html_path)

        print(f"  Saved {pdf_id}.json ({info['total_pages']} pages)")

        # Record analytics for this document
        if record_analytics and self._analytics:
            try:
                doc_end_time = datetime.now(tz=timezone.utc)
                self._analytics.record_extraction(
                    document_id=pdf_id,
                    result=result,
                    start_time=extraction_start_time,
                    end_time=doc_end_time,
                )
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Analytics recording failed for %s (non-fatal): %s", pdf_id, exc
                )

        if record_analytics and self._analytics:
            try:
                token_usage = result.get("token_usage", {})
                input_tokens = token_usage.get("input_tokens")
                output_tokens = token_usage.get("output_tokens")

                coherence_scores = [
                    p.get("validation", {}).get("coherence_score")
                    for p in result.get("pages", [])
                    if p.get("validation", {}).get("coherence_score") is not None
                ]
                avg_confidence = (
                    round(sum(coherence_scores) / len(coherence_scores), 2)
                    if coherence_scores else None
                )

                self._analytics.record_stage(
                    document_id=pdf_id,
                    stage_name="extraction",
                    start_time=extraction_start_time,
                    end_time=doc_end_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    confidence_score=avg_confidence,
                )
                self._analytics.record_ai_conversion(
                    document_id=pdf_id,
                    model_name=GEMINI_MODEL,
                    prompt_tokens=input_tokens or 0,
                    completion_tokens=output_tokens or 0,
                    extraction_confidence=avg_confidence,
                    chunks_processed=result.get("total_pages"),
                    parallel_processing=parallel_processing,
                )
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Analytics stage/AI recording failed for %s (non-fatal): %s", pdf_id, exc
                )

        # Update aggregate stats
        metrics = result["quality_metrics"]
        self.stats["total_pages"] += info["total_pages"]
        self.stats["successful_extractions"] += metrics.get("pages_successful", 0)
        self.stats["failed_extractions"] += metrics.get("pages_failed", 0)
        self.stats["pages_by_confidence"]["high"] += metrics.get("pages_high_confidence", 0)
        self.stats["pages_by_confidence"]["medium"] += metrics.get("pages_medium_confidence", 0)
        self.stats["pages_by_confidence"]["low"] += metrics.get("pages_low_confidence", 0)

        return result

    def _reset_pdf_stats(self, pdf_id, saved_pdfs):
        """Subtract a previously-saved PDF's stats so it can be re-saved without double-counting."""
        if pdf_id not in saved_pdfs:
            return
        output_path = OUTPUT_FOLDER / f"{pdf_id}.json"
        if not output_path.exists():
            return
        with open(output_path, "r", encoding="utf-8") as f:
            old_result = json.load(f)
        old_metrics = old_result.get("quality_metrics", {})
        self.stats["total_pages"] -= old_result.get("total_pages", 0)
        self.stats["successful_extractions"] -= old_metrics.get("pages_successful", 0)
        self.stats["failed_extractions"] -= old_metrics.get("pages_failed", 0)
        self.stats["pages_by_confidence"]["high"] -= old_metrics.get("pages_high_confidence", 0)
        self.stats["pages_by_confidence"]["medium"] -= old_metrics.get("pages_medium_confidence", 0)
        self.stats["pages_by_confidence"]["low"] -= old_metrics.get("pages_low_confidence", 0)

    def _retry_failed_pages(self, failed_pages: List[Tuple[str, int, str]], results_by_pdf: Dict):
        """Retry failed pages using multiprocessing.Pool and update results dict.

        Args:
            failed_pages: List of (pdf_path_str, page_num, doc_id) tuples to retry
            results_by_pdf: Dict mapping pdf_id -> {page_num: page_result} (modified in place)
        """
        if not failed_pages:
            return

        print(f"Retrying {len(failed_pages)} pages...")
        successful_retries = 0
        still_failed = 0

        with Pool(processes=MAX_WORKERS) as pool:
            for doc_id, page_num, retry_result in tqdm(
                pool.imap_unordered(_process_page_worker, failed_pages),
                total=len(failed_pages),
                desc="Retrying pages",
            ):
                # Update token stats from retry
                token_usage = retry_result.get("token_usage", {})
                self.stats["total_input_tokens"] += token_usage.get("input_tokens", 0)
                self.stats["total_output_tokens"] += token_usage.get("output_tokens", 0)

                if retry_result.get("error"):
                    still_failed += 1
                else:
                    successful_retries += 1

                # Update the result in place
                if doc_id in results_by_pdf:
                    results_by_pdf[doc_id][page_num] = retry_result

        print(f"Retry complete: {successful_retries} succeeded, {still_failed} still failed")

    def _generate_summary_report(self, results: List[Dict]):
        """Generate aggregate summary report."""
        # Collect all metrics
        all_coherence_scores = []
        pdfs_by_status = {
            "fully_successful": [],
            "partial_issues": [],
            "failed": [],
        }

        for result in results:
            pdf_id = result["pdf_id"]
            metrics = result.get("quality_metrics", {})

            if metrics.get("pages_failed", 0) == 0 and metrics.get("pages_low_confidence", 0) == 0:
                pdfs_by_status["fully_successful"].append(pdf_id)
            elif metrics.get("pages_failed", 0) == result.get("total_pages", 1):
                pdfs_by_status["failed"].append(pdf_id)
            else:
                pdfs_by_status["partial_issues"].append(pdf_id)

            # Collect page-level scores
            for page in result.get("pages", []):
                val = page.get("validation", {})
                if val.get("coherence_score") is not None:
                    all_coherence_scores.append(val["coherence_score"])

        # Calculate success rate
        total_pages = self.stats["total_pages"]
        success_rate = self.stats["successful_extractions"] / total_pages if total_pages > 0 else 0

        summary = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_pdfs": self.stats["total_pdfs"],
            "pdfs_processed": len(results),
            "total_pages_processed": total_pages,
            "successful_extractions": self.stats["successful_extractions"],
            "failed_extractions": self.stats["failed_extractions"],
            "success_rate": round(success_rate, 3),
            "quality_breakdown": {
                "high_confidence": self.stats["pages_by_confidence"]["high"],
                "medium_confidence": self.stats["pages_by_confidence"]["medium"],
                "low_confidence": self.stats["pages_by_confidence"]["low"],
            },
            "avg_coherence_score": round(sum(all_coherence_scores) / len(all_coherence_scores), 2) if all_coherence_scores else None,
            "pdfs_by_status": pdfs_by_status,
            "token_usage": {
                "total_input_tokens": self.stats["total_input_tokens"],
                "total_output_tokens": self.stats["total_output_tokens"],
            },
        }

        # Save summary JSON
        summary_path = REPORTS_FOLDER / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Generate HTML report
        self._generate_html_report(summary, results)

        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Total PDFs processed: {len(results)}")
        print(f"Total pages: {total_pages}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"High confidence: {summary['quality_breakdown']['high_confidence']}")
        print(f"Medium confidence: {summary['quality_breakdown']['medium_confidence']}")
        print(f"Low confidence: {summary['quality_breakdown']['low_confidence']}")
        print()
        print(f"Token usage:")
        print(f"  Input tokens:  {self.stats['total_input_tokens']:,}")
        print(f"  Output tokens: {self.stats['total_output_tokens']:,}")
        print(f"  Total tokens:  {self.stats['total_input_tokens'] + self.stats['total_output_tokens']:,}")
        print(f"\nReports saved to: {REPORTS_FOLDER}")

    def _generate_html_report(self, summary: Dict, results: List[Dict]):
        """Generate human-readable HTML quality report."""
        # Calculate percentage safely
        total_pages = summary['total_pages_processed'] or 1
        high_pct = summary['quality_breakdown']['high_confidence'] / total_pages * 100
        med_pct = summary['quality_breakdown']['medium_confidence'] / total_pages * 100
        low_pct = summary['quality_breakdown']['low_confidence'] / total_pages * 100

        # Collect problem pages for detailed reporting
        failed_pages = []
        low_confidence_pages = []
        for result in results:
            pdf_id = result['pdf_id']
            pdf_filename = result.get('source_path', '')
            for page in result.get('pages', []):
                page_num = page.get('page_number', '?')
                # Check for extraction errors
                if page.get('error'):
                    failed_pages.append({
                        'pdf_id': pdf_id,
                        'pdf_filename': pdf_filename,
                        'page': page_num,
                        'error': page['error']
                    })
                # Check for low confidence (coherence < 7)
                coherence = page.get('validation', {}).get('coherence_score')
                if coherence is not None and coherence < 7:
                    issues = page.get('validation', {}).get('coherence_issues', [])
                    low_confidence_pages.append({
                        'pdf_id': pdf_id,
                        'pdf_filename': pdf_filename,
                        'page': page_num,
                        'score': coherence,
                        'issues': issues
                    })

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PDF Extraction Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metric {{ display: inline-block; margin-right: 30px; margin-bottom: 10px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .metric-label {{ color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .status-success {{ color: #4CAF50; }}
        .status-partial {{ color: #FF9800; }}
        .status-failed {{ color: #F44336; }}
        .score-high {{ color: #4CAF50; font-weight: bold; }}
        .score-medium {{ color: #FF9800; }}
        .score-low {{ color: #F44336; }}
        .note {{ background: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .error-section {{ background: #ffebee; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #F44336; }}
        .warning-section {{ background: #fff8e1; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #FF9800; }}
        .error-text {{ color: #c62828; font-family: monospace; font-size: 0.9em; word-break: break-all; }}
        a {{ color: #1976D2; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>PDF Extraction Quality Report</h1>
    <p>Generated: {summary['generated_at']}</p>

    <div class="summary">
        <div class="metric">
            <div class="metric-value">{summary['pdfs_processed']}</div>
            <div class="metric-label">PDFs Processed</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['total_pages_processed']}</div>
            <div class="metric-label">Total Pages</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['success_rate']:.1%}</div>
            <div class="metric-label">Success Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('avg_coherence_score', 'N/A')}/10</div>
            <div class="metric-label">Avg Coherence Score</div>
        </div>
    </div>

    <div class="note">
        <strong>Metrics Explained:</strong>
        <ul style="margin: 5px 0;">
            <li><strong>Success Rate:</strong> Percentage of pages where extraction completed without errors (API failures, parsing issues)</li>
            <li><strong>Confidence:</strong> Based on coherence score (1-10) from LLM quality assessment. High = 9-10, Medium = 7-8, Low = &lt;7</li>
        </ul>
    </div>

    <h2>Quality Breakdown</h2>
    <table>
        <tr>
            <th>Confidence Level</th>
            <th>Page Count</th>
            <th>Percentage</th>
        </tr>
        <tr>
            <td class="score-high">High (9-10)</td>
            <td>{summary['quality_breakdown']['high_confidence']}</td>
            <td>{high_pct:.1f}%</td>
        </tr>
        <tr>
            <td class="score-medium">Medium (7-8)</td>
            <td>{summary['quality_breakdown']['medium_confidence']}</td>
            <td>{med_pct:.1f}%</td>
        </tr>
        <tr>
            <td class="score-low">Low (&lt;7)</td>
            <td>{summary['quality_breakdown']['low_confidence']}</td>
            <td>{low_pct:.1f}%</td>
        </tr>
    </table>
"""

        # Add Failed Pages section if there are any
        if failed_pages:
            html += f"""
    <div class="error-section">
        <h2 style="margin-top: 0; color: #c62828;">Failed Pages ({len(failed_pages)})</h2>
        <p>These pages had extraction errors and could not be processed:</p>
        <table>
            <tr>
                <th>PDF</th>
                <th>Page</th>
                <th>Error</th>
            </tr>
"""
            for fp in failed_pages:
                error_escaped = fp['error'].replace('<', '&lt;').replace('>', '&gt;')
                html += f"""            <tr>
                <td>{fp['pdf_id']}</td>
                <td>{fp['page']}</td>
                <td class="error-text">{error_escaped}</td>
            </tr>
"""
            html += """        </table>
    </div>
"""

        # Add Low Confidence Pages section if there are any
        if low_confidence_pages:
            html += f"""
    <div class="warning-section">
        <h2 style="margin-top: 0; color: #e65100;">Low Confidence Pages ({len(low_confidence_pages)})</h2>
        <p>These pages have coherence scores below 7 and may need manual review:</p>
        <table>
            <tr>
                <th>PDF</th>
                <th>Page</th>
                <th>Score</th>
                <th>Issues</th>
            </tr>
"""
            for lcp in low_confidence_pages:
                # Format issues as a bullet list or "None reported"
                issues_html = ""
                if lcp['issues']:
                    issues_escaped = [issue.replace('<', '&lt;').replace('>', '&gt;') for issue in lcp['issues']]
                    issues_html = "<ul style='margin: 0; padding-left: 20px;'>" + "".join(f"<li>{issue}</li>" for issue in issues_escaped) + "</ul>"
                else:
                    issues_html = "<em>None reported</em>"
                html += f"""            <tr>
                <td>{lcp['pdf_id']}</td>
                <td>{lcp['page']}</td>
                <td class="score-low">{lcp['score']}</td>
                <td>{issues_html}</td>
            </tr>
"""
            html += """        </table>
    </div>
"""

        html += f"""
    <h2>PDF Status Summary</h2>
    <table>
        <tr>
            <th>Status</th>
            <th>Count</th>
            <th>PDF IDs</th>
        </tr>
        <tr>
            <td class="status-success">Fully Successful</td>
            <td>{len(summary['pdfs_by_status']['fully_successful'])}</td>
            <td>{', '.join(summary['pdfs_by_status']['fully_successful'][:5])}{'...' if len(summary['pdfs_by_status']['fully_successful']) > 5 else ''}</td>
        </tr>
        <tr>
            <td class="status-partial">Partial Issues</td>
            <td>{len(summary['pdfs_by_status']['partial_issues'])}</td>
            <td>{', '.join(summary['pdfs_by_status']['partial_issues'][:5])}{'...' if len(summary['pdfs_by_status']['partial_issues']) > 5 else ''}</td>
        </tr>
        <tr>
            <td class="status-failed">Failed</td>
            <td>{len(summary['pdfs_by_status']['failed'])}</td>
            <td>{', '.join(summary['pdfs_by_status']['failed'][:5])}{'...' if len(summary['pdfs_by_status']['failed']) > 5 else ''}</td>
        </tr>
    </table>

    <h2>Individual PDF Results</h2>
    <table>
        <tr>
            <th>PDF ID</th>
            <th>Pages</th>
            <th>Successful</th>
            <th>Avg Coherence</th>
            <th>Confidence</th>
        </tr>
"""

        for result in results:
            metrics = result.get("quality_metrics", {})
            coherence = metrics.get('avg_coherence_score')

            # Calculate confidence class based on coherence score
            if coherence is not None:
                if coherence >= 9:
                    conf_class = "score-high"
                    conf_label = "High"
                elif coherence >= 7:
                    conf_class = "score-medium"
                    conf_label = "Medium"
                else:
                    conf_class = "score-low"
                    conf_label = "Low"
            else:
                conf_class = ""
                conf_label = "N/A"

            html += f"""        <tr>
            <td>{result['pdf_id']}</td>
            <td>{result['total_pages']}</td>
            <td>{metrics.get('pages_successful', 0)}</td>
            <td>{coherence if coherence is not None else 'N/A'}</td>
            <td class="{conf_class}">{conf_label}</td>
        </tr>
"""

        html += """    </table>
</body>
</html>"""

        html_path = REPORTS_FOLDER / "quality_report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)


def main():
    """Main entry point."""
    global DATA_FOLDER, OUTPUT_FOLDER, REPORTS_FOLDER, GEMINI_MODEL

    parser = argparse.ArgumentParser(
        description="PDF Structured JSON Extraction Tool with Quality Verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory containing {doc_id}/source.pdf subfolders "
             "(default: hardcoded DATA_FOLDER constant)",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output directory for JSON/HTML files (default: output/)",
    )
    parser.add_argument(
        "--api-mode", action="store_true",
        help="Use Gemini Developer API with API keys instead of Vertex AI. "
             "Keys from GEMINI_API_KEYS env var or .env.local.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Override the Gemini model name (default: {GEMINI_MODEL})",
    )
    args = parser.parse_args()

    # Auto-load .env.local
    _load_env_local()

    # Apply overrides
    if args.data_dir:
        DATA_FOLDER = args.data_dir
    if args.output_dir:
        OUTPUT_FOLDER = args.output_dir
        REPORTS_FOLDER = OUTPUT_FOLDER / "_reports"

    # Model override (CLI flag > env var > default)
    model_override = args.model or os.environ.get("GEMINI_MODEL", "")
    if model_override:
        GEMINI_MODEL = model_override

    # Initialize Gemini client
    _init_client(api_mode=args.api_mode)

    print("=" * 60)
    print("PDF Structured JSON Extraction Tool")
    print("=" * 60)
    print()
    print(f"Data Folder: {DATA_FOLDER}")
    print(f"Output Folder: {OUTPUT_FOLDER}")
    print(f"Model: {GEMINI_MODEL}")
    print(f"Max Workers: {MAX_WORKERS}")
    print()

    extractor = PDFExtractor()
    extractor.process_all_pdfs()


if __name__ == "__main__":
    main()

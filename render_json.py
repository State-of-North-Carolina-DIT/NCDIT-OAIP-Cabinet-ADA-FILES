#!/usr/bin/env python3

"""Render extraction-test JSON files to ADA-compliant HTML.

Self-contained script — no external src/ dependencies.

Reads the extraction-test JSON schema produced by extract_structured_json.py:
  { pdf_id, total_pages, pages: [{ page_number, content: [{type, ...}] }] }

Content types handled: heading, paragraph, table, image, list, form,
                        link, video, header_footer

ADA remediation applied before rendering:
  - H1 demotion (only one H1 = document title)
  - Heading hierarchy normalization (no level skips)
  - Running header/footer and repeated heading deduplication
  - Duplicate content removal (paragraphs, images)
  - Consecutive list merging (same page and across page boundaries)
  - Interrupted ordered list continuation (start attribute)
  - Cross-page table merging (same column count, no caption on continuation)
  - Decorative image detection (including role=presentation/none)
  - Table header inference (only explicit _is_header, NOT auto row-0)
  - Table of Contents removal (page-number-based ToC is meaningless post-extraction)
  - Inline [Image: ...] placeholder stripping
  - Underlined text misidentified as links → <u> restoration
  - Empty page removal
  - Ordered list duplicate number stripping
  - Markdown-to-HTML conversion in all text fields
  - Broken hyperlink fixing (invalid URLs, missing protocols, consecutive link merging)
  - Duplicate link deduplication

No ARIA attributes, no stylesheets, no role attributes — raw simple HTML.

Usage:
    python render_json.py json_to_html_to_auditor/
    python render_json.py path/to/file.json
    python render_json.py path/to/dir/ -o /tmp/output/
    python render_json.py path/to/file.json --raw   # skip ADA remediation
"""

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path


from dotenv import load_dotenv
try:
    from ada_analytics import PipelineAnalyticsCollector
    _HAS_ANALYTICS = True
except ImportError:
    _HAS_ANALYTICS = False

# Load .env from pipeline root (one level up from src/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Maximum pixel width for embedded images. Images wider than this are scaled
# down proportionally so they always fit within a standard page viewport.
_IMAGE_MAX_WIDTH = 900


# ---------------------------------------------------------------------------
# Shared text processing — markdown to HTML conversion
# ---------------------------------------------------------------------------

def _md_to_html(text: str) -> str:
    """Convert markdown formatting in text to HTML.

    Handles: **bold**, *italic*, _italic_, [text](url), leader dots,
    literal <u> tags, and preserves newlines.
    Applied AFTER html-escaping the base text.
    """
    html_text = escape(text)

    # Unescape literal <u> and </u> tags that were in the source text
    html_text = html_text.replace("&lt;u&gt;", "<u>").replace("&lt;/u&gt;", "</u>")

    # Bold+Italic: ***text*** -> <strong><em>text</em></strong> (must come FIRST)
    html_text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", html_text, flags=re.DOTALL)
    # Bold: **text** -> <strong>text</strong> (DOTALL to span newlines)
    html_text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_text, flags=re.DOTALL)
    # Italic: *text* -> <em>text</em> (but not inside <strong> tags)
    html_text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", html_text, flags=re.DOTALL)

    # Markdown underscore italic: _text_ -> <em>text</em>
    # Only match _word_ patterns (not filenames like my_file)
    html_text = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<em>\1</em>", html_text)

    # Markdown links: [text](url) -> <a href="url">text</a>
    html_text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2">\1</a>',
        html_text,
    )

    # Leader dots: replace 4+ consecutive dots with a single ellipsis
    # Screen readers read each dot individually which is terrible UX
    html_text = re.sub(r"\.{4,}", "…", html_text)

    # Strip orphan/stray asterisks that don't form valid bold/italic pairs
    # These are artifacts from Gemini extraction (e.g., "** text" or "text **")
    # Only strip leading/trailing orphan ** or * that don't have a matching pair
    html_text = re.sub(r"^\s*\*{1,2}\s+", "", html_text)   # leading: "** text" -> "text"
    html_text = re.sub(r"\s+\*{1,2}\s*$", "", html_text)   # trailing: "text **" -> "text"

    # Collapse spaced-out letters (e.g., "A P P . A Z . g o v" -> "APP.AZ.gov")
    # Matches patterns where single chars are separated by spaces
    def _collapse_spaced(m: re.Match) -> str:
        return m.group(0).replace(" ", "")
    html_text = re.sub(r"(?<![a-zA-Z])([a-zA-Z] ){3,}[a-zA-Z](?![a-zA-Z])", _collapse_spaced, html_text)

    # Preserve newlines
    html_text = html_text.replace("\n", "<br>")
    return html_text


def _strip_list_prefix(text: str, list_type: str) -> str:
    """Strip leading number/letter prefixes from ordered list items.

    Removes patterns like: "1. ", "(1) ", "a. ", "b) ", "i. ", "iv. ", etc.
    Only strips from ordered lists to avoid removing bullet markers.
    """
    if list_type != "ordered":
        return text
    stripped = re.sub(
        r"^\s*(?:"
        r"\(?\d+[.)]\)?\s*"       # numeric: 1. 1) (1)
        r"|"
        r"\(?[a-zA-Z][.)]\)?\s*"  # letter: a. a) (a)
        r"|"
        r"\(?(?:i{1,3}|iv|vi{0,3}|ix|xi{0,3})[.)]\)?\s*"  # roman: i. ii. iii. iv.
        r")",
        "",
        text,
        count=1,
    )
    return stripped


def _detect_list_style(items: list) -> str:
    """Detect the list style type from the first item's prefix.

    Returns HTML ol type attribute value: '1' (numeric), 'a' (lowercase letter),
    'A' (uppercase letter), 'i' (lowercase roman), 'I' (uppercase roman).
    """
    for li in items:
        text = li.get("text", "") if isinstance(li, dict) else str(li)
        text = text.strip()
        if re.match(r"^\s*\(?\d+[.)]\)?", text):
            return "1"
        if re.match(r"^\s*\(?[a-z][.)]\)?", text):
            return "a"
        if re.match(r"^\s*\(?[A-Z][.)]\)?", text):
            return "A"
        if re.match(r"^\s*\(?(?:i{1,3}|iv|vi{0,3}|ix|xi{0,3})[.)]\)?", text):
            return "i"
        if re.match(r"^\s*\(?(?:I{1,3}|IV|VI{0,3}|IX|XI{0,3})[.)]\)?", text):
            return "I"
    return "1"


def _letter_to_int(letter: str) -> int:
    """Convert a single letter to its 1-based alphabet position (a/A=1, b/B=2, …)."""
    c = letter.strip().lower()
    if len(c) == 1 and c.isalpha():
        return ord(c) - ord("a") + 1
    return 1


def _roman_to_int(s: str) -> int:
    """Convert a roman numeral string to an integer (case-insensitive)."""
    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    result, prev = 0, 0
    for ch in reversed(s.lower().strip()):
        v = vals.get(ch, 0)
        result = result - v if v < prev else result + v
        prev = v
    return result


def _list_item_ordinal(text: str) -> tuple[str, int] | None:
    """Return (style, ordinal_int) for the leading prefix of an ordered list item.

    style is '1', 'a', 'A', 'i', or 'I'.  Returns None if no recognized prefix.
    """
    t = text.strip()
    m = re.match(r"^\s*\(?(\d+)[.)]\)?", t)
    if m:
        return ("1", int(m.group(1)))
    m = re.match(r"^\s*\(?([a-z])[.)]\)?", t)
    if m:
        return ("a", _letter_to_int(m.group(1)))
    m = re.match(r"^\s*\(?([A-Z])[.)]\)?", t)
    if m:
        return ("A", _letter_to_int(m.group(1)))
    m = re.match(r"^\s*\(?([ivxlcdm]+)[.)]\)?", t, re.IGNORECASE)
    if m:
        raw = m.group(1)
        style = "I" if raw[0].isupper() else "i"
        return (style, _roman_to_int(raw))
    return None


# ---------------------------------------------------------------------------
# ADA remediation — operates on the raw JSON data (list of pages)
# ---------------------------------------------------------------------------

def _apply_ada_remediation(data: dict) -> dict[str, int]:
    """Apply ADA remediation steps to extraction-test JSON. Mutates data in place."""
    stats: dict[str, int] = {}
    pages = data.get("pages", [])

    stats["h1_demoted"] = _demote_extra_h1s(pages)
    stats["headings_normalized"] = _normalize_heading_hierarchy(pages)
    stats["consecutive_headings_merged"] = _merge_consecutive_headings(pages)
    stats["running_headers_deduped"] = _deduplicate_running_headers(pages)
    stats["page_numbers_removed"] = _remove_page_numbers(pages)
    stats["toc_removed"] = _remove_table_of_contents(pages)
    stats["duplicate_content_removed"] = _deduplicate_content(pages)
    stats["images_deduped"] = _deduplicate_images(pages)
    stats["lists_merged"] = _merge_consecutive_lists(pages)
    stats["lists_merged"] += _merge_consecutive_lists_across_pages(pages)
    stats["lists_continued"] = _continue_interrupted_lists(pages)
    stats["decorative_images_marked"] = _mark_decorative_images(pages)
    stats["tables_merged_across_pages"] = _merge_consecutive_tables_across_pages(pages)
    stats["table_headers_inferred"] = _infer_table_headers(pages)
    stats["broken_links_fixed"] = _fix_broken_hyperlinks(pages)
    stats["duplicate_links_removed"] = _deduplicate_links(pages)
    stats["inline_links_merged"] = _merge_inline_links(pages)
    stats["inline_image_placeholders_stripped"] = _strip_inline_image_placeholders(pages)
    stats["boilerplate_removed"] = _remove_boilerplate(pages)
    stats["empty_pages_removed"] = _remove_empty_pages(pages)

    return stats


def _demote_extra_h1s(pages: list) -> int:
    """Ensure only the first H1 stays as H1; demote others to H2."""
    found_h1 = False
    count = 0
    for page in pages:
        for item in page.get("content", []):
            if item.get("type") == "heading" and item.get("level") == 1:
                if not found_h1:
                    found_h1 = True
                else:
                    item["level"] = 2
                    count += 1
    return count


def _normalize_heading_hierarchy(pages: list) -> int:
    """Fix heading level skips (e.g. H1 -> H4 becomes H1 -> H2)."""
    count = 0
    last_level = 0
    for page in pages:
        for item in page.get("content", []):
            if item.get("type") == "heading":
                level = item.get("level", 2)
                if last_level > 0 and level > last_level + 1:
                    new_level = last_level + 1
                    item["level"] = new_level
                    count += 1
                    last_level = new_level
                else:
                    last_level = level
    return count


_DATE_HEADING_RE = re.compile(
    r"^\s*(?:"
    # Day of week (optional) + full date with month name
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)[,\s]+"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b"
    # Pure year or "October 7, 2025" style
    r"|\b\d{4}\b"
    r")\s*",
    re.IGNORECASE,
)


def _merge_consecutive_headings(pages: list) -> int:
    """Merge consecutive headings of the same level ONLY when the second is a date.

    Fixes cases like:
      <h2>NORTH CAROLINA 911 BOARD MEETING</h2>
      <h2>Wednesday, August 14, 2019</h2>
    Becomes:
      <h2>NORTH CAROLINA 911 BOARD MEETING — Wednesday, August 14, 2019</h2>

    NOT merged: independent section headings that happen to be adjacent on the
    same page (e.g., PowerPoint slide title + sub-section header).  Only the
    date-subtitle pattern is merged to avoid incorrect concatenation of
    logically separate headings.
    """
    count = 0
    for page in pages:
        content = page.get("content", [])
        if len(content) < 2:
            continue
        merged = [content[0]]
        for item in content[1:]:
            prev = merged[-1]
            if (item.get("type") == "heading" and prev.get("type") == "heading"
                    and item.get("level") == prev.get("level")):
                item_text = item.get("text", "").strip()
                # Only merge when the second heading looks like a date/time subtitle
                if _DATE_HEADING_RE.match(item_text):
                    prev_text = prev.get("text", "").strip()
                    prev["text"] = prev_text + " — " + item_text
                    count += 1
                    continue
            merged.append(item)
        page["content"] = merged
    return count

def _deduplicate_running_headers(pages: list) -> int:
    """Remove header_footer items AND headings that repeat identically across pages.

    Gemini sometimes re-extracts the same document/section heading on every
    page (e.g., a running header rendered as a heading element). These should
    appear once, not on every page.
    """
    if len(pages) < 2:
        return 0

    # Collect header/footer texts per page
    hf_texts: dict[str, int] = {}
    for page in pages:
        for item in page.get("content", []):
            if item.get("type") == "header_footer":
                text = item.get("text", "").strip()
                # Normalize page numbers (e.g. "Page 1 of 12" -> "Page X of 12")
                normalized = re.sub(r"Page\s+\d+", "Page X", text)
                normalized = re.sub(r"^\d+$", "X", normalized)
                hf_texts[normalized] = hf_texts.get(normalized, 0) + 1

    # Texts appearing on more than half the pages are running headers
    threshold = max(2, len(pages) // 2)
    running = {t for t, c in hf_texts.items() if c >= threshold}

    # Also detect headings that repeat across many pages (running titles).
    # Count how many pages each heading text appears on.
    heading_page_count: dict[str, int] = {}
    for page in pages:
        seen_on_page: set[str] = set()
        for item in page.get("content", []):
            if item.get("type") == "heading":
                key = item.get("text", "").strip().lower()
                if key and key not in seen_on_page:
                    seen_on_page.add(key)
                    heading_page_count[key] = heading_page_count.get(key, 0) + 1
    running_headings = {t for t, c in heading_page_count.items() if c >= threshold}

    count = 0
    first_occurrence: set[str] = set()
    for page in pages:
        original = page.get("content", [])
        filtered = []
        for item in original:
            if item.get("type") == "header_footer":
                text = item.get("text", "").strip()
                normalized = re.sub(r"Page\s+\d+", "Page X", text)
                normalized = re.sub(r"^\d+$", "X", normalized)
                if normalized in running:
                    count += 1
                    continue
            elif item.get("type") == "heading":
                key = item.get("text", "").strip().lower()
                if key in running_headings:
                    if key in first_occurrence:
                        # Already rendered once — skip this duplicate
                        count += 1
                        continue
                    first_occurrence.add(key)
            filtered.append(item)
        page["content"] = filtered
    return count


def _remove_page_numbers(pages: list) -> int:
    """Remove header_footer items that are page numbers.

    Matches patterns like: "1", "- 1 -", "Page 1", "Page 1 of 12",
    "1 | Page", "**2** | Page", "p. 1", "| 1 |", etc.
    """
    page_num_patterns = [
        r"^\s*\d+\s*$",                              # bare number: "1", " 12 "
        r"^\s*-\s*\d+\s*-\s*$",                       # dash-wrapped: "- 1 -"
        r"^\s*\|\s*\d+\s*\|\s*$",                     # pipe-wrapped: "| 1 |"
        r"^\s*page\s+\d+\s*$",                        # "Page 1"
        r"^\s*page\s+\d+\s+of\s+\d+\s*$",             # "Page 1 of 12"
        r"^\s*\d+\s+of\s+\d+\s*$",                    # "1 of 4", "2 of 4"
        r"^\s*p\.?\s*\d+\s*$",                         # "p. 1", "p1"
        r"^\s*\d+\s*\|\s*page\b",                      # "1 | Page"
        r"^\s*\*{0,2}\d+\*{0,2}\s*\|\s*page\b.*$",    # "**2** | Page (Rev ...)"
        r"^\s*\d+\s*\|\s*p\s*a\s*g\s*e\s*$",          # "3 | P a g e"
        r"^\s*\S+\s+page\s+\d+\s+of\s+\d+\b.*$",     # "00234464.25 Page 3 of 39 ..."
        r"^.*\|\s*page\s+\d+\s+of\s+\d+\s*\|.*$",    # "... | Page 37 of 39 | ..."
        r"^.*\|\s*page\s+\d+\s+of\s+\d+\s*$",        # "... | Page 33 of 123" (no trailing pipe)
        r"^.*\d+\s*\|\s*p\s*a\s*g\s*e\s*$",           # "Department of X 10 | P a g e"
        r"^.*\d+\s*\|\s*p\s*a\s*g\s*$",               # Truncated: "Department of X 10 | P a g"
        r"^.*\|\s*page\s+\d+\s*/\s*\d+\s*$",             # "... | Page 1/6"
        r"^.*\|\s*page\s+\d+\s*$",                        # "... | Page 33"
    ]
    combined = re.compile("|".join(page_num_patterns), re.IGNORECASE)

    count = 0
    for page in pages:
        original = page.get("content", [])
        filtered = []
        for item in original:
            if item.get("type") == "header_footer":
                text = item.get("text", "").strip()
                if combined.match(text):
                    count += 1
                    continue
            filtered.append(item)
        page["content"] = filtered
    return count

def _deduplicate_content(pages: list) -> int:
    """Remove duplicate paragraphs and headings across pages."""
    seen_texts: set[str] = set()
    count = 0
    for page in pages:
        original = page.get("content", [])
        filtered = []
        for item in original:
            if item.get("type") in ("paragraph", "heading"):
                text = item.get("text", "").strip()
                if not text:
                    continue
                # Use first 200 chars as dedup key
                key = text[:200].lower()
                if key in seen_texts:
                    count += 1
                    continue
                seen_texts.add(key)
            filtered.append(item)
        page["content"] = filtered
    return count


def _deduplicate_images(pages: list) -> int:
    """Remove duplicate images based on description AND base64 content.

    Deduplicates by:
      1. base64 data hash (catches identical images with different descriptions)
      2. description text (catches same-description images without base64 data)
    """
    seen_hashes: set[str] = set()
    seen_descs: set[str] = set()
    count = 0
    for page in pages:
        original = page.get("content", [])
        filtered = []
        for item in original:
            if item.get("type") == "image":
                b64 = item.get("base64_data") or ""
                desc = (item.get("description") or "").strip().lower()

                # Check by content hash first (most reliable)
                if b64:
                    content_hash = hashlib.md5(b64[:1000].encode()).hexdigest()
                    if content_hash in seen_hashes:
                        count += 1
                        continue
                    seen_hashes.add(content_hash)
                # Fall back to description dedup for images without base64
                elif desc and desc not in (
                    "unidentified image", "image", "decorative image"
                ):
                    if desc in seen_descs:
                        count += 1
                        continue
                    seen_descs.add(desc)
            filtered.append(item)
        page["content"] = filtered
    return count


def _merge_consecutive_lists(pages: list) -> int:
    """Merge consecutive lists of the same type."""
    count = 0
    for page in pages:
        content = page.get("content", [])
        if len(content) < 2:
            continue
        merged = [content[0]]
        for item in content[1:]:
            prev = merged[-1]
            if (item.get("type") == "list" and prev.get("type") == "list"
                    and item.get("list_type") == prev.get("list_type")):
                prev["items"].extend(item.get("items", []))
                count += 1
            else:
                merged.append(item)
        page["content"] = merged
    return count

def _merge_consecutive_lists_across_pages(pages: list) -> int:
    """Merge ordered lists that span PDF page breaks.

    When a numbered/lettered list is split across PDF pages, the JSON has two
    separate list objects on consecutive pages.  This merges them when the
    second list's first item is the direct continuation of the first list's
    last item (e.g., ends at "f." and continues at "g.").
    """
    count = 0
    for i in range(len(pages) - 1):
        content1 = pages[i].get("content", [])
        content2 = pages[i + 1].get("content", [])
        if not content1 or not content2:
            continue
        last = content1[-1]
        first = content2[0]
        if (last.get("type") != "list" or first.get("type") != "list"):
            continue
        if last.get("list_type") != first.get("list_type"):
            continue
        last_items = last.get("items", [])
        first_items = first.get("items", [])
        if not last_items or not first_items:
            continue
        last_text = (last_items[-1].get("text", "") if isinstance(last_items[-1], dict)
                     else str(last_items[-1])).strip()
        first_text = (first_items[0].get("text", "") if isinstance(first_items[0], dict)
                      else str(first_items[0])).strip()
        last_ord = _list_item_ordinal(last_text)
        first_ord = _list_item_ordinal(first_text)
        if last_ord and first_ord and last_ord[0] == first_ord[0]:
            if first_ord[1] == last_ord[1] + 1:
                last["items"].extend(first_items)
                content2.pop(0)
                count += 1
    return count


def _merge_consecutive_tables_across_pages(pages: list) -> int:
    """Merge tables that span PDF page boundaries.

    When a table is split across two pages, the JSON has two separate table
    objects. This merges them when:
      - The last item on page N is a table and the first item on page N+1 is a table
      - Both tables have the same number of columns
      - The second table has no caption/title (it's a continuation, not a new table)

    Row indices on the second table's cells are shifted so they append after
    the first table's last row.
    """
    count = 0
    for i in range(len(pages) - 1):
        content1 = pages[i].get("content", [])
        content2 = pages[i + 1].get("content", [])
        if not content1 or not content2:
            continue
        last = content1[-1]
        first = content2[0]
        if last.get("type") != "table" or first.get("type") != "table":
            continue
        # Don't merge if the second table has its own caption/title — it's a new table
        if first.get("caption") or first.get("title"):
            continue
        cells1 = last.get("cells", [])
        cells2 = first.get("cells", [])
        if not cells1 or not cells2:
            continue
        # Compare column counts
        cols1 = max(c.get("column_start", 0) for c in cells1) + 1
        cols2 = max(c.get("column_start", 0) for c in cells2) + 1
        if cols1 != cols2:
            continue
        # Shift row indices on the second table's cells
        max_row1 = max(c.get("row_start", 0) for c in cells1) + 1
        for cell in cells2:
            cell["row_start"] = cell.get("row_start", 0) + max_row1
        last["cells"].extend(cells2)
        content2.pop(0)
        count += 1
    return count


def _continue_interrupted_lists(pages: list) -> int:
    """Fix ordered lists that restart at 1 after being interrupted by non-list content.

    When an ordered list is split by an intervening paragraph or heading on
    the same page (e.g., a note between items 6 and 7), the second list
    loses its continuation numbering. This detects that pattern and injects
    the correct ``start`` attribute so the second list continues where the
    first left off.
    """
    count = 0
    for page in pages:
        content = page.get("content", [])
        if len(content) < 3:
            continue
        # Walk through looking for list -> non-list -> list patterns
        for i in range(len(content) - 2):
            first = content[i]
            second = content[i + 2]
            middle = content[i + 1]
            if (first.get("type") != "list" or second.get("type") != "list"):
                continue
            if first.get("list_type") != "ordered" or second.get("list_type") != "ordered":
                continue
            if middle.get("type") == "list":
                continue
            first_items = first.get("items", [])
            second_items = second.get("items", [])
            if not first_items or not second_items:
                continue
            last_text = (first_items[-1].get("text", "") if isinstance(first_items[-1], dict)
                         else str(first_items[-1])).strip()
            next_text = (second_items[0].get("text", "") if isinstance(second_items[0], dict)
                         else str(second_items[0])).strip()
            last_ord = _list_item_ordinal(last_text)
            next_ord = _list_item_ordinal(next_text)
            if last_ord and next_ord and last_ord[0] == next_ord[0]:
                if next_ord[1] == last_ord[1] + 1:
                    # The second list continues from the first — mark it
                    second["_start"] = next_ord[1]
                    count += 1
    return count


def _mark_decorative_images(pages: list) -> int:
    """Mark images as decorative based on description, dimensions, AND content.

    Also treats images with role="presentation" or role="none" as decorative —
    these ARIA roles signal that the image conveys no information, so an empty
    alt attribute is the correct ADA-compliant replacement.
    """
    decorative_descriptions = {"", "decorative image"}
    generic_descriptions = {"unidentified image", "image", "logo"}
    count = 0
    for page in pages:
        for item in page.get("content", []):
            if item.get("type") == "image":
                desc = (item.get("description") or "").strip().lower()
                b64 = item.get("base64_data") or ""
                has_substantial_data = len(b64) > 500
                bbox = item.get("bbox")

                # role="presentation" / role="none" means explicitly decorative
                role = (item.get("role") or "").strip().lower()
                if role in ("presentation", "none"):
                    item["_decorative"] = True
                    count += 1
                    continue

                # Full-page screenshots: bbox covers ≥95% of a standard PDF page
                # (612×792 pt = 484,704 sq pt).  These are fallback captures by
                # the extraction pipeline and duplicate all the text already
                # extracted from the same page — suppress them.
                if bbox:
                    w = bbox.get("x1", 0) - bbox.get("x0", 0)
                    h = bbox.get("y1", 0) - bbox.get("y0", 0)
                    area = w * h
                    _STANDARD_PAGE_AREA = 612 * 792  # letter-size PDF
                    if area >= _STANDARD_PAGE_AREA * 0.90:
                        item["_decorative"] = True
                        count += 1
                        continue

                if desc in decorative_descriptions and not has_substantial_data:
                    item["_decorative"] = True
                    count += 1
                    continue

                if desc in generic_descriptions:
                    if not has_substantial_data:
                        item["_decorative"] = True
                        count += 1
                        continue
                    if bbox:
                        w = bbox.get("x1", 0) - bbox.get("x0", 0)
                        h = bbox.get("y1", 0) - bbox.get("y0", 0)
                        if w > 0 and h > 0 and (w < 10 or h < 10):
                            item["_decorative"] = True
                            count += 1
                            continue
                    if desc == "unidentified image":
                        item["description"] = "Document image"
                    continue


                # Check bbox dimensions for design elements (thin strips, lines)
                if bbox:
                    w = bbox.get("x1", 0) - bbox.get("x0", 0)
                    h = bbox.get("y1", 0) - bbox.get("y0", 0)
                    if w > 0 and h > 0:

                        # Thin strips: either dimension < 10px
                        if w < 10 or h < 10:
                            item["_decorative"] = True
                            count += 1
                            continue
                        # Extreme aspect ratios (gradient strips, decorative lines)
                        ratio = max(w, h) / min(w, h)
                        if ratio > 15:
                            item["_decorative"] = True
                            count += 1
                            continue
    return count


def _infer_table_headers(pages: list) -> int:
    """Mark row-0 cells as headers ONLY if they look like real headers.

    Criteria: all row-0 cells must be short text (< 60 chars), there must be
    at least 2 columns, AND none of the row-0 cells should contain mostly
    numeric data (which suggests data, not headers).
    """
    count = 0
    for page in pages:
        for item in page.get("content", []):
            if item.get("type") != "table":
                continue
            cells = item.get("cells", [])
            if not cells:
                continue
            row0 = [c for c in cells if c.get("row_start", 0) == 0]
            if not row0:
                continue
            all_short = all(len(c.get("text", "")) < 60 for c in row0)
            # Check that row-0 cells don't look like data (mostly numbers, dates, etc.)
            any_numeric = any(
                re.match(r"^\s*[\d$,.%]+\s*$", c.get("text", "").strip())
                for c in row0
                if c.get("text", "").strip()
            )
            if all_short and len(row0) >= 2 and not any_numeric:
                for c in row0:
                    c["_is_header"] = True
                count += 1
    return count


def _merge_inline_links(pages: list) -> int:
    """Merge standalone link items into adjacent paragraphs when they appear mid-sentence.

    Pattern: paragraph (ends incomplete/mid-sentence) → link → paragraph (starts as continuation)
    Result: single merged paragraph with the link rendered as inline markdown [text](url)

    Detects mid-sentence splits by checking:
    - Following paragraph starts with whitespace (e.g., " for reference.")
    - Following paragraph starts with punctuation (e.g., ". The next sentence", "), or")
    - Preceding paragraph does not end with sentence-final punctuation (.!?;:"')
    """
    count = 0

    # Patterns indicating the NEXT paragraph continues from the link position
    _continuation_start = re.compile(
        r'^[ \t]'              # starts with space/tab (e.g., " for reference")
        r'|^[.,;:)\]\'"]'     # starts with closing/separating punctuation
    )
    # Sentence-final punctuation at the END of the preceding paragraph
    _sentence_ending = re.compile(r'[.!?;:\"\']\s*$')

    for page in pages:
        content = page.get("content", [])
        i = 0
        new_content = []
        while i < len(content):
            item = content[i]

            # Look for para → link → para triplet
            if (
                i + 2 < len(content)
                and item.get("type") == "paragraph"
                and content[i + 1].get("type") == "link"
                and content[i + 2].get("type") == "paragraph"
            ):
                para_text = item.get("text", "")
                link_item = content[i + 1]
                next_text = content[i + 2].get("text", "")

                next_is_continuation = bool(_continuation_start.match(next_text))
                para_is_incomplete = not _sentence_ending.search(para_text)

                if next_is_continuation and para_is_incomplete:
                    link_text = link_item.get("text", "")
                    link_url = link_item.get("url", "")
                    merged_text = f"{para_text}[{link_text}]({link_url}){next_text}"
                    new_content.append({"type": "paragraph", "text": merged_text})
                    i += 3
                    count += 1
                    continue

            new_content.append(item)
            i += 1

        page["content"] = new_content

    return count


def _is_valid_url(url: str) -> bool:
    """Check if a string looks like a valid URL or URL-like reference."""
    if not url:
        return False
    url = url.strip()
    if re.match(r'^(?:https?://|mailto:|tel:|ftp://)', url, re.IGNORECASE):
        return True
    if url.startswith('/') and len(url) > 1:
        return True
    if re.match(r'^[\w.-]+\.\w{2,}(?:/\S*)?$', url):
        return True
    return False


def _fix_url_protocol(url: str) -> str:
    """Add missing protocol to bare domain URLs.

    www.example.com -> https://www.example.com
    example.gov/path -> https://example.gov/path
    """
    url = url.strip()
    if not url:
        return url
    if re.match(r'^(?:https?://|mailto:|tel:|ftp://|/)', url, re.IGNORECASE):
        return url
    if re.match(r'^[\w.-]+\.\w{2,}(?:/\S*)?$', url):
        return 'https://' + url
    return url


def _fix_broken_hyperlinks(pages: list) -> int:
    """Fix or remove broken hyperlinks in extracted content.

    Handles:
    1. Add missing protocol to bare domain URLs (www.x.com -> https://www.x.com)
    2. Convert link items with invalid URLs to paragraphs
    3. Merge consecutive link items with the same URL
    4. Fix bare domain URLs in markdown links within paragraph text
    """
    fixed_count = 0

    for page in pages:
        content = page.get("content", [])
        if not content:
            continue

        result = []
        for item in content:
            if item.get("type") != "link":
                result.append(item)
                continue

            link_text = item.get("text", "").strip()
            link_url = item.get("url", "").strip()

            # Fix 1: Add missing protocol to bare domain URLs
            if link_url and not re.match(r'^(?:https?://|mailto:|tel:|ftp://|/)', link_url, re.IGNORECASE):
                fixed_url = _fix_url_protocol(link_url)
                if fixed_url != link_url:
                    item["url"] = fixed_url
                    link_url = fixed_url
                    fixed_count += 1

            # Fix 2: Check if URL is valid after potential protocol fix
            url_is_valid = _is_valid_url(link_url)

            # Fix 3: If URL == text and URL is not valid, convert to paragraph
            url_equals_text = (
                link_url == link_text
                or link_url.rstrip('.') == link_text.rstrip('.')
                or link_url.lower().strip() == link_text.lower().strip()
            )

            if url_equals_text and not url_is_valid:
                # URL == display text and not a real URL means Gemini
                # misinterpreted underlined text as a hyperlink. Preserve
                # the underline styling so fidelity to the original PDF
                # is maintained.
                result.append({"type": "paragraph", "text": f"<u>{link_text}</u>"})
                fixed_count += 1
                continue

            # Fix 4: If URL is clearly not valid, convert to paragraph
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
                link_url = item.get("url", "")
                texts = [item.get("text", "")]
                j = i + 1
                while j < len(result) and result[j].get("type") == "link" and result[j].get("url", "") == link_url:
                    texts.append(result[j].get("text", ""))
                    j += 1
                if j > i + 1:
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

        page["content"] = merged

    return fixed_count


def _deduplicate_links(pages: list) -> int:
    """Remove standalone link elements whose URL already appears in paragraph text or as a prior link.

    Also fixes broken links: when a link has display text as its URL (e.g., href="CLICK HERE")
    and another link with the same display text has a valid URL, the broken one gets corrected.

    Also strips URLs that Gemini appended to the display text:
      "CLICK HERE https://example.com" → "CLICK HERE" (with real href)
    """
    count = 0

    # Pre-pass 0: strip URLs Gemini appended to link display text.
    # Pattern: "<display text> <url>" where the appended URL matches the href.
    _url_suffix_re = re.compile(r"\s+https?://\S+$")
    for page in pages:
        for item in page.get("content", []):
            if item.get("type") == "link":
                text = (item.get("text") or "").strip()
                url = (item.get("url") or "").strip()
                if url and _url_suffix_re.search(text):
                    # If text ends with the URL, strip it from display text
                    stripped = _url_suffix_re.sub("", text).strip()
                    if stripped:
                        item["text"] = stripped

    # Pre-pass: fix broken link URLs by finding correct URLs for the same display text.
    # Build a map: normalized display text -> correct URL (from any link with a real URL)
    text_to_real_url: dict[str, str] = {}
    for page in pages:
        for item in page.get("content", []):
            if item.get("type") == "link":
                url = (item.get("url") or "").strip()
                text = (item.get("text") or "").strip()
                # A "real" URL starts with http/https/mailto/ftp or contains a dot
                if url and url != text and (
                    url.startswith(("http://", "https://", "mailto:", "ftp://"))
                    or "." in url
                ):
                    norm_text = " ".join(text.split()).lower()
                    if norm_text:
                        text_to_real_url[norm_text] = url

    # Fix broken links using the map
    for page in pages:
        for item in page.get("content", []):
            if item.get("type") == "link":
                url = (item.get("url") or "").strip()
                text = (item.get("text") or "").strip()
                # Check if URL looks broken (url == text and not a real URL)
                url_is_broken = (
                    url == text
                    or (not url.startswith(("http://", "https://", "mailto:", "ftp://", "/"))
                        and "." not in url)
                )
                if url_is_broken:
                    norm_text = " ".join(text.split()).lower()
                    if norm_text in text_to_real_url:
                        item["url"] = text_to_real_url[norm_text]

    # First pass: collect ALL URLs mentioned in paragraphs/headings/table cells/lists across ALL pages
    global_text_urls: set[str] = set()
    for page in pages:
        for item in page.get("content", []):
            item_type = item.get("type")
            if item_type in ("paragraph", "heading"):
                text = item.get("text", "")
                for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
                    global_text_urls.add(match.group(2).strip().lower())
                for match in re.finditer(r"https?://\S+", text):
                    global_text_urls.add(match.group(0).strip().lower())
            elif item_type == "table":
                # Also scan table cells — links inside tables should suppress duplicate standalone links
                for cell in item.get("cells", []):
                    cell_text = cell.get("text", "")
                    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", cell_text):
                        global_text_urls.add(match.group(2).strip().lower())
                    for match in re.finditer(r"https?://\S+", cell_text):
                        global_text_urls.add(match.group(0).strip().lower())
            elif item_type == "list":
                # Also scan list items
                for li in item.get("items", []):
                    for li_item in [li] + li.get("children", []):
                        li_text = li_item.get("text", "")
                        for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", li_text):
                            global_text_urls.add(match.group(2).strip().lower())
                        for match in re.finditer(r"https?://\S+", li_text):
                            global_text_urls.add(match.group(0).strip().lower())

    # Second pass: remove duplicate link items (globally tracked)
    global_seen_urls: set[str] = set()
    for page in pages:
        content = page.get("content", [])
        filtered = []
        for item in content:
            if item.get("type") == "link":
                url = (item.get("url") or "").strip().lower()
                # Remove if URL already in paragraph text or already seen as a standalone link
                if url in global_text_urls or url in global_seen_urls:
                    count += 1
                    continue
                global_seen_urls.add(url)
            filtered.append(item)
        page["content"] = filtered

    # Third pass: remove short shadow paragraphs whose text exactly matches a link's display text.
    # These arise when Gemini extracts hyperlink text from table cells as separate paragraphs,
    # then PyMuPDF provides the same text as link display text. The paragraph is redundant.
    for page in pages:
        # Collect all link display texts on this page
        link_texts: set[str] = set()
        for item in page.get("content", []):
            if item.get("type") == "link":
                txt = (item.get("text") or "").strip().lower()
                if txt and len(txt) < 100:
                    link_texts.add(txt)

        if not link_texts:
            continue

        # Also add normalized versions (strip surrounding parens) for flexible matching
        link_texts_normalized: set[str] = link_texts | {
            t.strip("()[] \t") for t in link_texts
        }

        content = page.get("content", [])
        filtered = []
        for item in content:
            if item.get("type") == "paragraph":
                para_text = (item.get("text") or "").strip()
                # Only remove short paragraphs that exactly match a link display text
                if para_text and len(para_text) < 100 and para_text.lower() in link_texts_normalized:
                    count += 1
                    continue
            filtered.append(item)
        page["content"] = filtered

    return count


def _remove_table_of_contents(pages: list) -> int:
    """Remove Table of Contents sections from the document.

    After PDF extraction page numbers are meaningless, so ToC entries serve no
    purpose and introduce noise/hallucinations.

    Detection:
    - A heading whose text matches "table of contents", "contents", "toc", etc.
    - All subsequent content items that look like ToC entries (text ending with
      leader dots + page number, e.g. "Introduction......... 3") or short
      continuations, up until a real section heading is encountered.
    - Also detects "List of Figures", "List of Tables" appendices by the same rule.
    """
    toc_heading_re = re.compile(
        r"^\s*(?:"
        r"table\s+of\s+contents?"
        r"|list\s+of\s+(?:figures?|tables?|illustrations?|exhibits?|appendix|appendices)"
        r"|contents?"
        r"|toc"
        r")\s*$",
        re.IGNORECASE,
    )
    # ToC entry: text ending with leader dots + number, or multiple spaces + number
    toc_entry_re = re.compile(r"\.{2,}\s*\d+\s*$|\s{3,}\d+\s*$|\t\d+\s*$")

    # Flatten all items with page/item indices for multi-page spanning
    flat: list[tuple[int, int, dict]] = []
    for pi, page in enumerate(pages):
        for ii, item in enumerate(page.get("content", [])):
            flat.append((pi, ii, item))

    if not flat:
        return 0

    to_remove: set[tuple[int, int]] = set()
    i = 0
    while i < len(flat):
        pi, ii, item = flat[i]
        if item.get("type") == "heading" and toc_heading_re.match(item.get("text", "").strip()):
            # Mark this heading for removal and scan forward for ToC entries
            to_remove.add((pi, ii))
            i += 1
            while i < len(flat):
                cpi, cii, cur = flat[i]
                ctype = cur.get("type")
                ctext = cur.get("text", "").strip()

                if ctype == "heading":
                    # Another ToC-style heading (e.g., "List of Figures") — consume it too
                    if toc_heading_re.match(ctext):
                        to_remove.add((cpi, cii))
                        i += 1
                        continue
                    # Real section heading — end of ToC
                    break

                if ctype in ("paragraph", "header_footer"):
                    if not ctext:
                        to_remove.add((cpi, cii))
                        i += 1
                        continue
                    if toc_entry_re.search(ctext):
                        to_remove.add((cpi, cii))
                        i += 1
                        continue
                    # Short unlabeled text (< 80 chars) inside ToC block — likely a
                    # section label or continuation (e.g., "Appendix A")
                    if len(ctext) < 80:
                        to_remove.add((cpi, cii))
                        i += 1
                        continue
                    # Substantial paragraph — end of ToC
                    break

                if ctype == "list":
                    list_items = cur.get("items", [])
                    if list_items:
                        entry_count = sum(
                            1 for li in list_items
                            if toc_entry_re.search(
                                (li.get("text", "") if isinstance(li, dict) else str(li)).strip()
                            )
                        )
                        # If at least half the list items look like ToC entries, remove the list
                        if entry_count >= max(1, len(list_items) // 2):
                            to_remove.add((cpi, cii))
                            i += 1
                            continue
                    # List doesn't look like ToC — end of ToC
                    break

                # Any other type (table, image, form, …) — end of ToC
                break
        else:
            i += 1

    if not to_remove:
        return 0

    # Apply removals
    for pi, page in enumerate(pages):
        content = page.get("content", [])
        page["content"] = [item for ii, item in enumerate(content) if (pi, ii) not in to_remove]

    return len(to_remove)


_INLINE_IMAGE_RE = re.compile(r"\[Image:\s*[^\]]*\]")


def _strip_inline_image_placeholders(pages: list) -> int:
    """Remove [Image: ...] placeholder text from paragraphs and headings.

    Gemini sometimes inserts alt-text descriptions inline (e.g.,
    "[Image: NC DIT logo]", "[Image: Green paintbrush icon]") as plain
    text within paragraphs instead of creating proper image elements.
    These read badly on screen readers and clutter the output.
    """
    count = 0
    for page in pages:
        content = page.get("content", [])
        filtered = []
        for item in content:
            if item.get("type") in ("paragraph", "heading"):
                text = item.get("text", "")
                new_text = _INLINE_IMAGE_RE.sub("", text).strip()
                if new_text != text.strip():
                    count += 1
                    if not new_text:
                        # Entire paragraph was just an image placeholder — drop it
                        continue
                    item["text"] = new_text
            filtered.append(item)
        page["content"] = filtered
    return count


def _remove_boilerplate(pages: list) -> int:
    """Remove boilerplate content like 'page intentionally left blank'."""
    boilerplate_patterns = [
        r"^\s*(?:this\s+)?page\s+(?:is\s+)?intentionally\s+left\s+blank\s*\.?\s*$",
        r"^\s*(?:this\s+)?page\s+left\s+(?:intentionally\s+)?blank\s*\.?\s*$",
    ]
    combined = re.compile("|".join(boilerplate_patterns), re.IGNORECASE)
    count = 0
    for page in pages:
        original = page.get("content", [])
        filtered = []
        for item in original:
            if item.get("type") in ("paragraph", "heading"):
                text = item.get("text", "").strip()
                if combined.match(text):
                    count += 1
                    continue
            filtered.append(item)
        page["content"] = filtered
    return count


def _remove_empty_pages(pages: list) -> int:
    """Remove pages with no content after deduplication."""
    count = 0
    i = 0
    while i < len(pages):
        if not pages[i].get("content"):
            pages.pop(i)
            count += 1
        else:
            i += 1
    return count


def _s(val, default: str = "") -> str:
    """Safely convert a value to str, treating None as default."""
    return default if val is None else str(val)


def _render_heading(item: dict) -> str:
    level = max(1, min(6, item.get("level", 2)))
    text = _md_to_html(_s(item.get("text")))
    return f"<h{level}>{text}</h{level}>"


def _render_paragraph(item: dict) -> str:
    text = _s(item.get("text"))
    html_text = _md_to_html(text)
    return f"<p>{html_text}</p>"


def _render_table(item: dict) -> str:
    cells = item.get("cells", [])
    if not cells:
        caption = item.get("caption") or item.get("title") or "Empty table"
        return f"<table><caption>{_md_to_html(caption)}</caption><tr><td>(empty table)</td></tr></table>"

    max_row = max(c.get("row_start", 0) for c in cells)
    max_col = max(c.get("column_start", 0) for c in cells)

    # Build a lookup: (row, col) -> cell
    cell_map: dict[tuple[int, int], dict] = {}
    for c in cells:
        key = (c.get("row_start", 0), c.get("column_start", 0))
        cell_map[key] = c

    # Track which cells are covered by rowspan/colspan
    covered: set[tuple[int, int]] = set()
    for c in cells:
        r = c.get("row_start", 0)
        col = c.get("column_start", 0)
        rs = c.get("num_rows", c.get("row_span", 1)) or 1
        cs = c.get("num_columns", c.get("column_span", 1)) or 1
        for dr in range(rs):
            for dc in range(cs):
                if dr == 0 and dc == 0:
                    continue
                covered.add((r + dr, col + dc))

    # aria_label on a table is the accessible name — map it to <caption> when
    # no explicit caption/title is present (aria-label is stripped from output
    # per our no-ARIA policy, so <caption> is the semantic replacement).
    caption = item.get("caption") or item.get("title") or item.get("aria_label") or ""
    # Use _md_to_html (not bare escape) so **bold** in captions renders correctly
    caption_html = f"<caption>{_md_to_html(caption)}</caption>" if caption else ""

    # Trim trailing empty rows: find the last row where at least one cell has content
    last_content_row = max_row
    while last_content_row > 0:
        row_cells = [c for c in cells if c.get("row_start", 0) == last_content_row]
        if any(c.get("text", "").strip() for c in row_cells):
            break
        last_content_row -= 1

    html = f"<table>{caption_html}"

    for r in range(last_content_row + 1):
        html += "<tr>"
        for c_idx in range(max_col + 1):
            if (r, c_idx) in covered:
                continue
            cell = cell_map.get((r, c_idx))
            if cell is None:
                html += "<td></td>"
                continue

            cell_text = _md_to_html(_s(cell.get("text")))
            # ONLY use _is_header flag — do NOT auto-mark row 0 as header
            is_header = cell.get("_is_header", False)
            tag = "th" if is_header else "td"
            attrs = ""

            if is_header:
                # scope="col" for row-0 column headers; scope="row" for column-0
                # row headers. scope is essential for screen reader navigation.
                cell_row = cell.get("row_start", 0)
                cell_col = cell.get("column_start", 0)
                if cell_row == 0:
                    attrs += ' scope="col"'
                elif cell_col == 0:
                    attrs += ' scope="row"'

            rs = cell.get("num_rows", cell.get("row_span", 1)) or 1
            cs = cell.get("num_columns", cell.get("column_span", 1)) or 1
            if rs > 1:
                attrs += f' rowspan="{rs}"'
            if cs > 1:
                attrs += f' colspan="{cs}"'

            html += f"<{tag}{attrs}>{cell_text}</{tag}>"
        html += "</tr>"

    html += "</table>"
    return html


def _scale_image_b64(b64: str, fmt: str) -> tuple[str, str]:
    """Scale down a base64-encoded image if its width exceeds _IMAGE_MAX_WIDTH.

    Returns (new_b64, new_fmt). If PIL is unavailable or scaling fails,
    returns the original (b64, fmt) unchanged.
    """
    if not _PIL_AVAILABLE or not b64:
        return b64, fmt
    try:
        raw = base64.b64decode(b64)
        img = _PILImage.open(io.BytesIO(raw))
        if img.width <= _IMAGE_MAX_WIDTH:
            return b64, fmt
        new_h = int(img.height * _IMAGE_MAX_WIDTH / img.width)
        img = img.resize((_IMAGE_MAX_WIDTH, new_h), _PILImage.LANCZOS)
        buf = io.BytesIO()
        save_fmt = "PNG" if fmt.lower() not in ("jpeg", "jpg") else "JPEG"
        img.save(buf, format=save_fmt)
        new_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        new_fmt = "png" if save_fmt == "PNG" else "jpeg"
        return new_b64, new_fmt
    except Exception:
        return b64, fmt


def _render_image(item: dict) -> str:
    desc = _s(item.get("description"))
    caption = _s(item.get("caption"))
    # Treat "Document image" / "document image" as generic — mark decorative if no real data
    is_decorative = item.get("_decorative", False) or not desc or desc.lower() in (
        "unidentified image", "image", "decorative image", "document image",
    )
    b64 = item.get("base64_data", "")
    fmt = item.get("format", "png")
    b64, fmt = _scale_image_b64(b64, fmt)

    if is_decorative:
        if b64:
            return f'<img src="data:image/{fmt};base64,{b64}" alt="">'
        return "<!-- decorative image -->"

    alt_text = escape(desc)
    if b64:
        src = f"data:image/{fmt};base64,{b64}"
        img_tag = f'<img src="{src}" alt="{alt_text}">'
    else:
        # Render as visible text placeholder instead of hidden comment
        # so screen readers and users know an image was intended here
        return f"<p>[Image: {alt_text}]</p>"

    if caption:
        cap_stripped = caption.strip()
        # Suppress meaningless figcaptions: very short (< 5 chars), pure numbers/percentages,
        # or generic phrases that don't describe the image
        is_meaningless = (
            len(cap_stripped) < 5
            or re.match(r"^\s*[\d,.%$]+\s*$", cap_stripped)
            or cap_stripped.lower() in ("image", "figure", "photo", "logo", "icon")
        )
        if not is_meaningless:
            return f"<figure>{img_tag}<figcaption>{escape(caption)}</figcaption></figure>"
    return img_tag


def _render_list(item: dict) -> str:
    list_type = item.get("list_type", "unordered")
    tag = "ol" if list_type == "ordered" else "ul"

    # Detect list style (a, A, i, I, 1) and start number from first item prefix
    ol_attrs = ""
    if list_type == "ordered":
        items = item.get("items", [])
        style = _detect_list_style(items)
        if style != "1":
            ol_attrs += f' type="{style}"'
        # Detect start value from first item for ALL styles (numeric, letter, roman).
        # This handles lists that continue from a previous page after being split
        # at a PDF page boundary (e.g., a list that starts at "g." on page 2).
        if items:
            first_text = items[0].get("text", "") if isinstance(items[0], dict) else str(items[0])
            first_text = first_text.strip()
            # Use _start from remediation if set, otherwise detect from prefix
            explicit_start = item.get("_start")
            if explicit_start and explicit_start > 1:
                ol_attrs += f' start="{explicit_start}"'
            else:
                ord_info = _list_item_ordinal(first_text)
                if ord_info and ord_info[1] > 1:
                    ol_attrs += f' start="{ord_info[1]}"'

    items_html = ""
    for li in item.get("items", []):
        text = li.get("text", "") if isinstance(li, dict) else str(li)
        # Strip duplicate numbering from ordered list items
        text = _strip_list_prefix(text, list_type)
        li_html = _md_to_html(text)
        # Render children as nested list
        children = li.get("children", []) if isinstance(li, dict) else []
        if children:
            child_tag = "ol" if list_type == "ordered" else "ul"
            li_html += f"<{child_tag}>"
            for child in children:
                child_text = child.get("text", "") if isinstance(child, dict) else str(child)
                child_text = _strip_list_prefix(child_text, list_type)
                li_html += f"<li>{_md_to_html(child_text)}</li>"
            li_html += f"</{child_tag}>"
        items_html += f"<li>{li_html}</li>"
    return f"<{tag}{ol_attrs}>{items_html}</{tag}>"


def _render_form(item: dict) -> str:
    title = item.get("title", "Form")
    fields = item.get("fields", [])
    if not fields:
        return f"<table><caption>{escape(title)}</caption><tr><td>(empty form)</td></tr></table>"

    html = "<table>"
    html += f"<caption>{escape(title)}</caption>"
    html += "<tr><th>Field</th><th>Type</th><th>Value</th></tr>"
    for field in fields:
        label = escape(_s(field.get("label")))
        ftype = escape(_s(field.get("field_type")))
        value = field.get("value")
        value_str = escape(str(value)) if value is not None else ""

        options = field.get("options", [])
        if options:
            value_str += " [" + ", ".join(escape(str(o)) for o in options) + "]"

        html += f"<tr><td>{label}</td><td>{ftype}</td><td>{value_str}</td></tr>"
    html += "</table>"
    return html


def _render_link(item: dict) -> str:
    text = escape(_s(item.get("text")))
    url = _s(item.get("url"))
    url_esc = escape(url)
    return f'<p><a href="{url_esc}">{text}</a></p>'


def _render_video(item: dict) -> str:
    url = escape(_s(item.get("url")))
    desc = escape(_s(item.get("description"), "Video"))
    return f'<p><a href="{url}">{desc}</a></p>'


def _render_header_footer(item: dict) -> str:
    # Simple rendering — no role attributes, no <small>, markdown converted
    text = _md_to_html(_s(item.get("text")))
    return f"<p>{text}</p>"



# Dispatch table
_RENDERERS = {
    "heading": _render_heading,
    "paragraph": _render_paragraph,
    "table": _render_table,
    "image": _render_image,
    "list": _render_list,
    "form": _render_form,
    "link": _render_link,
    "video": _render_video,
    "header_footer": _render_header_footer,
}


def render_content_item(item: dict) -> str:
    """Render a single content item. NEVER returns empty string."""
    renderer = _RENDERERS.get(item.get("type", ""), None)
    if renderer:
        result = renderer(item)
        if result and result.strip():
            return result
        return _render_fallback(item)
    return _render_fallback(item)


def _render_fallback(item: dict) -> str:
    """Fallback renderer — guarantees non-empty output for any item."""
    item_type = escape(_s(item.get("type"), "unknown"))
    text = _s(item.get("text") or item.get("description") or item.get("title") or "")
    if text:
        return f"<p>{_md_to_html(text)}</p>"
    return f"<!-- {item_type} element (no text content) -->"


# ---------------------------------------------------------------------------
# Full document rendering
# ---------------------------------------------------------------------------

def _element_id(page_idx: int, item_idx: int, item: dict) -> str:
    """Generate a unique ID for an element based on position and content."""
    item_type = item.get("type", "unknown")
    text = _s(item.get("text") or item.get("description") or item.get("title") or "")
    return f"p{page_idx}:i{item_idx}:{item_type}:{text[:50]}"


def _reconcile_and_render(pages: list, max_passes: int = 3) -> list[tuple[int, int, dict, str]]:
    """Render all elements and reconcile until every element is accounted for."""
    expected: list[tuple[int, int, dict]] = []
    for page_idx, page in enumerate(pages):
        for item_idx, item in enumerate(page.get("content", [])):
            expected.append((page_idx, item_idx, item))

    results: list[tuple[int, int, dict, str]] = []
    missing: list[tuple[int, int, dict]] = []

    for page_idx, item_idx, item in expected:
        rendered = render_content_item(item)
        if rendered and rendered.strip():
            results.append((page_idx, item_idx, item, rendered))
        else:
            missing.append((page_idx, item_idx, item))

    if not missing:
        return results

    for page_idx, item_idx, item in missing:
        fallback = _render_fallback(item)
        insert_pos = 0
        for k, (pi, ii, _, _) in enumerate(results):
            if (pi, ii) < (page_idx, item_idx):
                insert_pos = k + 1
        results.insert(insert_pos, (page_idx, item_idx, item, fallback))

    if max_passes > 1:
        rendered_ids = {_element_id(pi, ii, it) for pi, ii, it, _ in results}
        expected_ids = {_element_id(pi, ii, it) for pi, ii, it in expected}
        still_missing = expected_ids - rendered_ids
        if still_missing:

            return _reconcile_and_render(pages, max_passes - 1)

    return results


def render_document(data: dict) -> str:
    pdf_id = data.get("pdf_id", "Document")
    title = pdf_id.replace("-", " ").title()

    pages = data.get("pages", [])

    # Final verification: every element from every page is accounted for
    rendered_elements = _reconcile_and_render(pages)

    expected_count = sum(len(p.get("content", [])) for p in pages)
    actual_count = len(rendered_elements)
    if actual_count != expected_count:
        print(
            f"  INTEGRITY ERROR: expected {expected_count} elements, "
            f"got {actual_count} after reconciliation",
            file=sys.stderr,
        )

    # Assemble HTML — no page breaks at all
    body_lines = []
    for page_idx, item_idx, item, html in rendered_elements:

        body_lines.append(html)

    body_html = "\n".join(body_lines)


    # Raw simple HTML — no ARIA, no roles.
    # viewport meta and the img style rule are required for responsive layout
    # and to prevent large images from overflowing their containers.

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
img {{ display: block; max-width: 100%; height: auto; }}
figure {{ max-width: 100%; }}
</style>
</head>
<body>
{body_html}
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def render_one(json_path: Path, output_path: Path, raw: bool = False) -> bool | None:
    """Render a single JSON file to HTML. Returns True on success, False on error, None if skipped."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))

        if "pages" not in data:
            return None

        pre_count = sum(len(p.get("content", [])) for p in data.get("pages", []))

        if not raw:
            stats = _apply_ada_remediation(data)
            changes = {k: v for k, v in stats.items() if v > 0}
            if changes:
                parts = ", ".join(f"{k}={v}" for k, v in changes.items())
                print(f"  ADA  {json_path.name}: {parts}")


        post_count = sum(len(p.get("content", [])) for p in data.get("pages", []))

        html = render_document(data)

        rendered_count = 0
        missing_elements = []
        for page_idx, page in enumerate(data.get("pages", [])):
            for item_idx, item in enumerate(page.get("content", [])):
                r = render_content_item(item)
                if r and r.strip():
                    rendered_count += 1
                else:
                    missing_elements.append(
                        f"pg{page.get('page_number', page_idx+1)}[{item_idx}] "
                        f"type={item.get('type')}"
                    )

        if missing_elements:
            print(
                f"  WARN {json_path.name}: {len(missing_elements)} elements "
                f"produced empty renders: {', '.join(missing_elements)}",
                file=sys.stderr,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        ada_note = f", ADA removed {pre_count - post_count}" if not raw and pre_count != post_count else ""
        print(
            f"  OK   {json_path.name} -> {output_path.name} "
            f"({len(html):,} bytes, {rendered_count}/{pre_count} elements{ada_note})"
        )
        return True
    except Exception as e:
        print(f"  FAIL {json_path.name}: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render extraction-test JSON to simple raw HTML."
    )
    parser.add_argument(
        "input", type=Path,
        help="Path to a .json file, a directory of .json files, or a parent "
             "directory with subdirectories containing .json files",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output HTML path (single file) or directory (batch mode)",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Skip ADA post-processing (raw render only)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    if args.input.is_file():
        out = args.output or args.input.with_suffix(".html")
        if not render_one(args.input, out, raw=args.raw):
            sys.exit(1)
        return


    json_files = sorted(args.input.glob("*.json"))
    nested = sorted(args.input.glob("*/*.json"))
    if nested:
        json_files.extend(nested)
    if not json_files:
        print(f"No .json files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output or args.input
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(json_files)} JSON files in {args.input}/\n")

    # Optional analytics
    collector = None
    if _HAS_ANALYTICS:
        try:
            collector = PipelineAnalyticsCollector(
                tenant_id=os.environ.get("PROJECT_ID", "camp-ai-nc"),
            )
        except Exception:
            pass

    ok, fail, skipped = 0, 0, 0
    for jf in json_files:

        if args.output:
            rel = jf.relative_to(args.input)
            html_path = out_dir / rel.with_suffix(".html")
        else:
            html_path = jf.with_suffix(".html")

        render_start = datetime.now(timezone.utc)
        result = render_one(jf, html_path, raw=args.raw)
        render_end = datetime.now(timezone.utc)

        if result is True:
            ok += 1
            if collector:
                try:
                    doc_id = jf.stem if jf.parent == args.input else jf.parent.name
                    collector.record_rendering(
                        document_id=doc_id,
                        remediation_stats=None,
                        start_time=render_start,
                        end_time=render_end,
                    )
                    collector.record_stage(
                        document_id=doc_id,
                        stage_name="rendering",
                        start_time=render_start,
                        end_time=render_end,
                    )
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "Analytics recording failed for %s (non-fatal): %s", doc_id, exc
                    )
        elif result is False:
            fail += 1
        else:
            skipped += 1

    if collector:
        try:
            collector.flush()
        except Exception as exc:
            logging.getLogger(__name__).debug("Analytics flush failed (non-fatal): %s", exc)

    print(f"\nDone: {ok} rendered, {fail} failed, {skipped} skipped")
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

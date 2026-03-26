"""Evaluation prompts for the standalone auditor.

Contains all prompt templates for the 3-call fidelity scoring system
and the Final Decider (4th call). Separated from auditor.py to keep
prompts easy to review, iterate, and version independently.

Prompt engineering decisions grounded in:
- CritiqueLLM: forensic framing > adversarial framing
- CheckEval/DeCE: binary checklists > holistic scoring
- Castillo benchmark: JSON-in-Prompt > response_schema (11pt accuracy gap)
- Evidently AI / Cameron Wolfe: reasoning-before-verdict ordering
- MQM standard: 1:5:25 severity weighting (Minor:Major:Critical)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auditor import AuditConcern, SignalBreakdown

# Sentinel imported at runtime to avoid circular dependency
FABRICATED_TABLE_SENTINEL = 999.0

# ============================================================================
# Shared Fidelity Components
# ============================================================================

ROLE_PREAMBLE = """You are a forensic document auditor specializing in government document \
conversion quality. Your role is to verify that a converted document \
faithfully represents its source material. You are thorough, skeptical, \
and evidence-driven. You verify every claim against the source document. \
You never assume correctness — you confirm it."""

ACCEPTABLE_CHANGES = """The following changes are EXPECTED and are NOT defects. Do not flag them:

- Document titles may be AI-generated (added for accessibility if the source PDF lacks a proper title)
- Image alt text is always AI-generated (images in the source PDF have no text alternatives)
- Table captions and summaries may be AI-generated (added for screen reader users)
- Heading levels may be adjusted (normalized for screen reader navigation)
- Reading order may differ slightly (optimized for accessibility)
- Text reformatting (whitespace changes, punctuation normalization)
- Addition of ARIA labels, skip links, table-of-contents navigation
- "(opens in new window)" text after external links
- Section labels like "(part 1)" or "(part 2)" added to long sections
- Footer with document metadata added by the pipeline
- The source PDF may be a scanned document. Minor OCR differences between your reading of the source and the output text are acceptable."""

EVIDENCE_REQUIREMENTS = """IMPORTANT: You must identify and document ALL defects BEFORE computing any scores. \
List every issue you find with evidence. Only after you have completed your analysis should \
you produce the summary scores.

For each defect you find, you MUST provide:
1. The exact text or description from the SOURCE PDF that is affected
2. The exact text or description from the OUTPUT that demonstrates the defect
3. Where in the document this occurs (section name, page reference, or element index)

If you cannot cite specific evidence, do not flag the defect."""

ONE_AT_A_TIME = """Evaluate each check below ONE AT A TIME. Complete your analysis for each \
check before moving to the next. Do not let the result of one check \
influence your analysis of another."""

JSON_FORMAT = """Respond with a JSON object in the following format. Process the fields \
in the order shown — complete your analysis before determining the verdict.

{{
  "checks": [
    {{
      "check_id": "XX",
      "check_name": "Human-readable name of the check",
      "a_analysis": "Your detailed analysis of whether this defect exists, examining the source and output carefully...",
      "b_evidence_source": "Exact quote or description from the source PDF (empty string if PASS)",
      "b_evidence_output": "Exact quote or description from the output (empty string if PASS)",
      "b_evidence_location": "Where in the document — section name, page number, or element index (empty string if PASS)",
      "c_verdict": "PASS or FAIL",
      "d_severity": "MINOR or MAJOR or CRITICAL (only include this field if c_verdict is FAIL)"
    }}
  ],
  "summary": {{
    "total_defects": 0,
    "critical_count": 0,
    "major_count": 0,
    "minor_count": 0,
    "penalty_sum": 0,
    "score": 1.0
  }}
}}

Scoring rules for the summary:
- penalty_sum = (critical_count * 25) + (major_count * 5) + (minor_count * 1)
- score = max(0.0, 1.0 - penalty_sum / {normalization_factor})
- The score must be a float between 0.0 and 1.0

You MUST include an entry for EVERY check listed below, even if the verdict is PASS."""


# ============================================================================
# Call 1: Content Fidelity (C1-C6) — normalization factor 86
# ============================================================================

def build_content_fidelity_prompt(extraction_json: str) -> str:
    """Build the Call 1 prompt. Input to Gemini: this text + source PDF."""
    json_format = JSON_FORMAT.replace("{normalization_factor}", "86")
    return f"""{ROLE_PREAMBLE}

## TASK

You are evaluating the CONTENT FIDELITY of a converted government document. \
The source PDF (attached) is the authoritative original. The output below is the \
pipeline's conversion result in structured JSON format. Your job is to determine \
whether the output faithfully preserves the source document's content.

The question is: "Does the output contain the same INFORMATION as the source?" — \
not "Does the output contain the same CHARACTERS as the source?"

## ACCEPTABLE CHANGES

{ACCEPTABLE_CHANGES}

## DOCUMENT TYPE CLASSIFICATION (evaluate FIRST)

Before performing content checks, classify this document. If it matches one of \
these exclusion categories, set document_classification to the matching category. \
You MUST still complete ALL content checks even if the document is excluded.

- "FORM": The document is primarily a fillable or printable form. This includes \
documents that begin with a form and then have additional text (e.g., a W9 has a \
form on page 1 then instructions on pages 2-6 — still classify as FORM). Also \
includes procurement/bid forms (IFBs, RFPs, RFQs) with fillable fields, documents \
where more than 50% of content is form fields, signature lines, checkboxes, or \
blank lines for handwritten input. Forms are useless when converted to HTML because \
they lose fillable/printable functionality.

- "SINGLE_GRAPHIC": The document is a single page consisting ONLY of one image, map, \
graphic, chart, or visual element — with NO substantive text content. CRITICAL RULES: \
(1) A document with 2 or more pages is NEVER a SINGLE_GRAPHIC. \
(2) A single-page document that has tables, paragraphs of text content, or lists \
alongside images is NOT a SINGLE_GRAPHIC — it is STANDARD. A title or brief caption \
near a graphic is acceptable, but if there are data tables, body paragraphs, or \
structured text, it is STANDARD regardless of image count. \
(3) A scanned document that contains text is NOT a SINGLE_GRAPHIC — it is STANDARD. \
Only classify as SINGLE_GRAPHIC when the page is PURELY a standalone visual: a map, \
infographic, poster, chart, or diagram with no accompanying text content.

- "SLIDE_DECK": The document is an image-based slide presentation. Characteristics \
include: landscape/horizontal page orientation, image-only pages or multiple images \
per page, background images or colors on each page, slide-like layout. These do not \
have a clear HTML conversion.

- "STANDARD": The document does not match any exclusion category. Process normally.

## BINARY CHECKS

{ONE_AT_A_TIME}

CHECK C1: Fabricated paragraphs or sentences
Severity if FAIL: CRITICAL
Look for text in the output that does not appear in any form in the source PDF. \
This includes: fabricated table-of-contents entries, invented section headings \
like "Page 2 Content" or "Slide 1: Title", hallucinated body text, made-up \
introductions or conclusions. Does NOT include expected AI additions listed above \
(titles, alt text, captions, ARIA labels, section labels, footer). \
The key test: can you find the basis for this text anywhere in the source? \
If not, it is fabricated. \
IMPORTANT: Before flagging fabrication, verify the text has NO basis in the \
source PDF. Headings, section labels, and titles that appear in the source \
(even with minor rewording) are NOT fabricated. Only flag text that is entirely \
invented with no corresponding source content.

CHECK C2: Fabricated contact information
Severity if FAIL: CRITICAL
Look for phone numbers, email addresses, mailing addresses, or URLs in the output \
that do not exist in the source PDF. This is the highest-risk defect — a citizen \
calling a fabricated phone number on a government form is a catastrophic failure. \
Check EVERY phone number, email, and URL in the output against the source. \
Even one mismatch = CRITICAL.

CHECK C3: Complete section or page missing
Severity if FAIL: CRITICAL
Look for an entire section or page present in the source PDF that is absent from \
the output. This includes pages silently dropped during conversion and entire \
sections omitted. A section that is present but truncated is C5, not C3. \
The test: can you find this section's CONTENT (not just its heading) somewhere \
in the output?

CHECK C4: Table with significantly fewer rows
Severity if FAIL: MAJOR
Look for tables in the output that have materially fewer rows than the \
corresponding table in the source PDF. "Materially fewer" means more than \
approximately 20% of rows are missing. \
Minor differences from reformatting (plus or minus 1-2 rows from header/footer \
handling) are acceptable. Count the actual data rows in both source and output.

CHECK C5: Significant text omission within a section
Severity if FAIL: MAJOR
Look for sections that exist in the output but are missing substantial content \
compared to the source. The section starts correctly but is cut short, or key \
paragraphs within the section are absent. \
IMPORTANT: Only flag C5 if MORE THAN 25% of a section's text is missing. \
A single missing sentence from a long section is NOT C5. \
A few missing URLs or link text is NOT C5 — those are link issues (handled \
separately by S7). C5 is for substantial paragraph-level content loss only.

CHECK C6: Minor text differences beyond acceptable reformatting
Severity if FAIL: MINOR
Look for wording changes that alter meaning (not just formatting). Examples: \
changing "shall" to "may" in a legal document, changing a number, altering a \
date. Minor rephrasing that preserves meaning (e.g., "in order to" becoming \
"to") is NOT a defect. OCR differences (single-letter substitutions, missing \
hyphens or punctuation, minor spacing changes) are NOT defects — they reflect \
the source PDF's own text quality. Only flag if the meaning is materially \
changed (e.g., "shall" vs "may", a number or date change, or an entire word \
replaced with a different word). The test: would a government employee get \
different information from the output than from the source?

## EVIDENCE REQUIREMENTS

{EVIDENCE_REQUIREMENTS}

## OUTPUT FORMAT

{json_format}

ADDITIONALLY, include these top-level fields in your JSON response (BEFORE the "checks" array):
- "document_classification": One of "STANDARD", "FORM", "SINGLE_GRAPHIC", or "SLIDE_DECK"
- "document_classification_rationale": Brief explanation of why you chose this classification

## OUTPUT DOCUMENT (extraction JSON — pipeline's conversion result)

```json
{extraction_json}
```

## SOURCE DOCUMENT

The source PDF is attached as a file. Compare the output above against this source."""


# ============================================================================
# Call 2: Structural Fidelity (S1-S5, S7) — normalization factor 14
# ============================================================================

def build_structural_fidelity_prompt(extraction_json: str) -> str:
    """Build the Call 2 prompt. Input to Gemini: this text + source PDF."""
    json_format = JSON_FORMAT.replace("{normalization_factor}", "15")
    return f"""{ROLE_PREAMBLE}

## TASK

You are evaluating the STRUCTURAL FIDELITY of a converted government document. \
The source PDF (attached) is the authoritative original. The output below is the \
pipeline's conversion result in structured JSON format. Your job is to determine \
whether the output faithfully preserves the source document's organization and \
structure.

Focus on: headings, tables, lists, reading order, and content duplication. \
This is about STRUCTURE, not text content (Call 1 handles content accuracy).

## ACCEPTABLE CHANGES

{ACCEPTABLE_CHANGES}

## BINARY CHECKS

{ONE_AT_A_TIME}

CHECK S1: Content duplicated
Severity if FAIL: MINOR
Look for the same text block appearing 2 or more times in the output when it \
appears only once in the source. This includes: duplicated paragraphs, repeated \
sections, content appearing in both a list and a subsequent paragraph. Does NOT \
include intentional repetition in the source (a refrain, a header on every page, \
a repeated disclaimer). The test: does this text appear more times in the output \
than in the source?

CHECK S2: Heading hierarchy broken
Severity if FAIL: MAJOR
Look for heading levels that skip (H1 to H3 with no H2), nest incorrectly, or \
are assigned wrong levels compared to the source. Screen readers use heading \
hierarchy for navigation — H1 should be the document title, H2 for major \
sections, H3 for subsections. A skip from H2 to H4 is a defect. Heading LEVEL \
adjustments for accessibility (e.g., H4 promoted to H2 to fix a flat hierarchy) \
are acceptable per the acceptable changes list. \
Also check for headings at the SAME visual level in the source being incorrectly \
nested in the output (e.g., two sibling H2 sections in the source where one is \
rendered as H3 inside the other). Paragraphs that are clearly section headings \
in the source but rendered as plain <p> tags are also defects.

CHECK S3: Table structure corrupted
Severity if FAIL: MAJOR when header row is misidentified (data as headers or \
headers as data), column labels use <td> instead of <th>, or a caption row is \
coded as a header row. MINOR for cosmetic structural differences that do not \
affect data interpretation (e.g., merged cell alignment, minor border differences).
Look for tables where headers are misidentified (data in header row, or header \
data in body), columns are merged or split incorrectly, or the table's structural \
organization does not match the source. This is about STRUCTURE (headers, rows, \
columns), not DATA (C4/C5 cover data completeness). \
Specific patterns to check: \
(1) The correct row must be marked as the table header (<th>). The first DATA row \
should NOT be marked as a header if it contains data values, not column labels. \
(2) Column headers must use <th> tags, not <td>. If the source table has clear \
column labels, they must be <th> in the output. \
(3) Single-column tables: data that should be a list or paragraphs should NOT be \
rendered as a one-column table. \
(4) Non-tabular content (form checkboxes, paragraphs, unstructured text) should \
NOT be rendered as a data table. \
(5) When a large table spans multiple PDF pages with repeated headers, it should be \
rendered as one continuous table, not multiple separate tables per page. \
(6) A row with a colspan spanning all columns that contains a title or caption \
should be the table <caption>, not a header row.

CHECK S4: Reading order significantly different
Severity if FAIL: MINOR
Look for content blocks whose reading order diverges materially from the source. \
Minor reordering for accessibility is acceptable. Moving a conclusion before an \
introduction, or placing a sidebar's content in the middle of the main text, is \
a defect. The test: if you read the output top-to-bottom, does the information \
flow logically and match the source's intended order?

CHECK S5: Lists converted incorrectly
Severity if FAIL: MINOR
Look for list nesting that is wrong (flat list rendered as nested or vice versa), \
list items merged or split incorrectly, numbered lists that lose numbering, or \
bullet lists that become numbered or vice versa. Minor formatting differences are \
acceptable — focus on whether the list's MEANING is preserved. \
Also check these specific patterns: \
(1) Numbering TYPE must match the source — if the PDF uses letters (a, b, c), \
the output must use letters, not numbers. If the PDF uses Roman numerals \
(I, II, III, IV), they must not be confused with English letters (I, J, K, L). \
(2) A single continuous numbered list in the source that gets RESET (restarts \
from 1) in the output is a defect — the numbering should continue. \
(3) Items visually indented as sub-points in the source (nested under a parent \
bullet or definition) but rendered at the same level as the parent in the \
output — this flattens the hierarchy and loses meaning. Specific sub-point \
patterns to look for: Roman numerals (i, ii, iii) that should be nested under \
numbered items (1, 2, 3) but appear at the same level; letters (a, b, c) that \
should be sub-items of parent bullets rendered as siblings. Compare nesting \
depth in the source vs the output — if the source has 2+ nesting levels but \
the output is flat, that is a FAIL.

CHECK S6: Footnotes preserved and properly placed
Severity if FAIL: MINOR
Look for footnotes in the source (superscript numbers, asterisks, "Note" sections, \
endnote lists). Verify: (1) all footnotes from the source are present in the output, \
(2) footnote numbers/symbols match their reference points in the body text, \
(3) footnotes are not duplicated, (4) footnote content is not mixed into the main \
body text as if it were a regular paragraph. If the source has no footnotes, PASS.

CHECK S7: Link injection issues
Severity if FAIL: MAJOR
Look for links in the output that are broken (wrong URL), missing (a URL clearly \
visible in the PDF is not clickable), or fabricated (a link appears with no basis \
in the source). Focus on government URLs (.gov, .edu, .org) and contact links \
(email, phone). Minor link text formatting differences are acceptable.

## EVIDENCE REQUIREMENTS

{EVIDENCE_REQUIREMENTS}

## OUTPUT FORMAT

{json_format}

## OUTPUT DOCUMENT (extraction JSON — pipeline's conversion result)

```json
{extraction_json}
```

## SOURCE DOCUMENT

The source PDF is attached as a file. Compare the structural organization of the \
output above against this source."""


# ============================================================================
# Call 3: Visual Fidelity (V1-V5) — normalization factor 21
# V1 MAJOR=5, V2 MINOR=1, V3 MAJOR=5, V4 MAJOR=5, V5 MAJOR=5
# ============================================================================

VISUAL_ACCEPTABLE_CHANGES = """The output is HTML, not a PDF clone. The following visual differences are expected \
and are NOT defects:
- Different fonts, font sizes, and text styling
- Different margins, padding, and page dimensions
- Different colors and background styling
- Responsive HTML layout vs fixed PDF layout
- Navigation elements added (table of contents, skip links)
- "(opens in new window)" indicators after links
- Decorative images (background patterns, watermarks, page borders) being hidden — this is an intentional accessibility improvement
- Footer with pipeline metadata
- Section labels like "(part 1)" in headings
- Document titles and section headings appearing in BOTH a table of contents / \
navigation area AND the body content — this is intentional accessible design \
(WCAG navigation landmarks). A heading listed in the TOC and repeated as a body \
heading is NOT content duplication.

The question is: does the visual ORGANIZATION of information match? Can a user \
find the same information in roughly the same place?"""

VISUAL_CHECKS = """CHECK V1: Tables visually broken or misrendered
Severity if FAIL: MAJOR
Look for tables that are visually corrupted in the rendered HTML — missing \
borders making it impossible to distinguish rows and columns, overlapping cell \
content, columns misaligned, headers not visually distinct from body rows, or \
tables rendered as plain text paragraphs. A table that looks like a table and \
has readable content is fine even if the styling differs from the PDF. \
The test: can a sighted user read and understand the table data?

CHECK V2: Images missing or broken
Severity if FAIL: MINOR
Look for images that are clearly present and meaningful in the source PDF but \
are missing, broken, or absent from the HTML output. Check for: <img> elements \
with empty or invalid src attributes, meaningful images from the source that \
have no corresponding element in the output, images marked as decorative \
(role="presentation" or empty alt) that are actually meaningful content. \
Decorative images (background patterns, watermarks, page borders) being hidden \
is NOT a defect — it is an intentional accessibility improvement. \
The test: are MEANINGFUL images (photos, charts, diagrams, logos) present?

CHECK V3: Content visually duplicated on page
Severity if FAIL: MAJOR
Look for the same block-level content (paragraph, table, list, or section) \
appearing twice in the BODY of the rendered output. This is the visual \
confirmation of content duplication — catching it here provides defense in \
depth. \
Also check for images that are duplicated — the same image appearing twice, \
or a PARTIAL crop of an image appearing alongside the full image. Scanned \
document pages rendered as full-page images alongside the extracted text \
content are also duplication (the user sees the content twice: once as an \
image and once as text). \
\
Do NOT flag any of these as duplication: \
- CRITICAL: Alt text (alt="...") is invisible metadata for screen readers — \
  it is NEVER visible content. Do NOT count alt text matching nearby text as \
  duplication. An image appearing once with descriptive alt text is correct, \
  not duplicated. Only flag when the SAME VISIBLE body text appears twice. \
- Href attribute values in anchor tags are NOT visible text occurrences. \
- A heading appearing once in a TOC/navigation region AND once in the body \
  is expected accessible design (WCAG navigation landmarks). Only flag if \
  the same heading appears MORE THAN TWICE in total. \
- "(opens in new window)" indicators after links are intentional additions. \
\
Only flag when the SAME body paragraph, table, list, or section content \
appears twice in the body. Ignore navigation/TOC regions when counting.

CHECK V4: Images placed next to wrong text
Severity if FAIL: MAJOR
When images ARE present, check that each image is positioned near the text \
that references it. An image placed next to the wrong text block (e.g., a \
screenshot of step 3 next to the instructions for step 5) is a major defect \
because it changes the meaning of the content. The test: does each image \
appear in the correct context relative to its surrounding text?

CHECK V5: Text formatting not preserved (bold, italic, underline)
Severity if FAIL: MAJOR when formatting carries meaning (e.g., bold terms in a \
definition list, italicized legal citations, underlined warnings), MINOR when \
formatting is purely decorative and its absence does not change comprehension.
Look for text that is bold, italic, underlined, or otherwise visually emphasized \
in the source PDF but rendered as plain unstyled text in the output HTML. Focus \
on: section titles that lose bold styling, defined terms or keywords that should \
be bold, emphasized text (italic/underline) that conveys importance or \
distinction. Do NOT flag differences in font family, font size, or color — \
those are expected HTML/PDF differences. The test: is meaningful text emphasis \
from the source preserved in the output?"""


def build_visual_fidelity_prompt(rendered_html: str | None = None) -> str:
    """Build the Call 3 prompt. Input to Gemini: this text + source PDF."""
    json_format = JSON_FORMAT.replace("{normalization_factor}", "21")
    if rendered_html is not None:
        return f"""{ROLE_PREAMBLE}

## TASK

You are evaluating the VISUAL FIDELITY of a converted government document. \
You have two inputs:
1. The SOURCE PDF (attached) — this is the authoritative original
2. The rendered HTML output (below) — this is the converted result

Your job is to analyze the HTML and determine whether it would faithfully \
represent the source document's visual presentation when rendered in a browser. \
Examine the HTML elements, structure, and inline CSS to evaluate layout, table \
rendering, image presence, content duplication, and overall quality. \
Content accuracy is handled by a separate evaluation; your focus is on visual \
presentation, layout, and rendering quality.

## ACCEPTABLE VISUAL DIFFERENCES

{VISUAL_ACCEPTABLE_CHANGES}

## BINARY CHECKS

{ONE_AT_A_TIME}

{VISUAL_CHECKS}

## EVIDENCE REQUIREMENTS

{EVIDENCE_REQUIREMENTS}

When describing evidence, reference what you observe in the HTML and PDF:
- "In the source PDF, page 3 shows a 3-column table with bordered cells"
- "In the HTML, the table uses <div> elements instead of <table>, losing visual structure"
- "The source PDF shows a seal/logo; the HTML has no corresponding <img> element"

## OUTPUT FORMAT

{json_format}

## RENDERED HTML OUTPUT

The complete rendered HTML output is below. Analyze its structure, elements, \
and inline CSS to evaluate how it would appear when rendered in a browser.

```html
{rendered_html}
```

## SOURCE DOCUMENT

The source PDF is attached as a file. Compare the HTML above against this \
source to evaluate visual fidelity."""
    else:
        return f"""{ROLE_PREAMBLE}

## TASK

You are evaluating the VISUAL FIDELITY of a converted government document. \
You have the SOURCE PDF (attached) but no rendered HTML is available. \
Evaluate based on the source PDF quality and note that visual fidelity \
cannot be fully assessed without the rendered output.

## OUTPUT FORMAT

{json_format}

Respond with all checks set to PASS and note in analysis that HTML was not available."""


# ============================================================================
# Call 4: Final Decider — optional synthesis + routing call
# ============================================================================

DECIDER_ROLE = """\
You are the final quality arbiter for North Carolina government document \
conversions (PDF to accessible HTML). You receive comprehensive quality \
signals from multiple independent evaluation systems and make the routing \
decision. You are thorough, evidence-driven, and balanced.

IMPORTANT: A good document incorrectly sent to human review (false positive) \
is 3x more costly than a borderline document auto-approved (false negative). \
When in doubt and signals are mostly positive, lean toward AUTO_APPROVE."""

DECIDER_ROUTING = """\
You must route this document to one of three outcomes:

AUTO_APPROVE — No major or critical unresolved defects. \
Minor cosmetic issues (heading style, spacing) do not block approval. \
IMPORTANT: Do NOT auto-approve if ANY of these structural concerns exist, \
even if fidelity scores are high: \
(1) heading hierarchy broken or headings misclassified as paragraphs, \
(2) list numbering resets or wrong numbering type, \
(3) images placed next to wrong text or out of order, \
(4a) hyperlinks completely missing from output (content loss), \
(4b) hyperlinks displaced from inline to separate location — blocks AUTO_APPROVE \
only when severity is MAJOR (3+ displaced links indicate a systemic problem; \
1-2 displaced links with minor severity are a cosmetic issue, not a blocker), \
(5) full-page scanned images duplicated alongside extracted text, \
(6) page numbers or footers retained in the HTML output, \
(7) scanned document with zero extractable source text (baseline_words=0) — \
output likely duplicates content as both page images and OCR'd text, \
(8) output has more images than source PDF (image_count_ratio > 1.0) — \
indicates image duplication. \
These issues break accessibility even when content is otherwise preserved.

EXCEPTION — Formatting-only concerns: If the ONLY concerns are about text \
formatting (V5 formatting preservation, literal asterisks/markdown syntax \
like **bold** or *italic* appearing in text), and ALL content checks (C1-C6) \
and structural checks (S1-S7) are PASS, the document MAY be auto-approved. \
Minor formatting issues (asterisks for bold, markdown syntax not rendered) \
alone do not warrant human review when content and structure are intact.

EXCEPTION — Minor-only concerns with high fidelity: When fidelity_composite \
>= 0.85 AND fidelity_routing is AUTO_APPROVE AND every concern has severity \
"minor" (no major or critical concerns at all), the document SHOULD be \
auto-approved. Examples of minor-only patterns that should NOT block approval: \
a single displaced link with minor severity, unrendered markdown asterisks, \
minor text differences. Remember: sending a \
good document to unnecessary human review is 3x more costly than approving \
a document with only minor cosmetic issues.

HUMAN_REVIEW — There are concerns that need human judgment. This is the \
default when signals disagree, when there are moderate defects, or when \
you're uncertain. When in doubt, HUMAN_REVIEW.

REJECT — The document has severe defects that make it unfit for publication. \
Near-total content loss, pervasive hallucination, or catastrophic structural \
failure. Use sparingly — only for clearly broken documents."""

DECIDER_CORROBORATION = """\
For each concern, note whether other sources agree or disagree:
- AGREEMENT between LLM fidelity and programmatic signals = HIGH confidence in the finding
- DISAGREEMENT on major/critical = HUMAN_REVIEW. Minor-only disagreement is not a barrier to AUTO_APPROVE
- When all sources agree the document is clean = HIGH confidence in AUTO_APPROVE
- When all sources agree the document is broken = HIGH confidence in REJECT

IMPORTANT — Unreliable programmatic signals:
Programmatic text comparison metrics (shingling_recall, shingling_precision, \
word_count_ratio, fabrication_detected_count) depend on PyMuPDF text extraction \
from the source PDF. PyMuPDF is UNRELIABLE for:
- Non-Latin scripts (Arabic, Thai, CJK, Cyrillic) — extraction returns garbled text
- Scanned documents (baseline_words=0 or very low) — no text to compare against
- Complex layouts with rotated/overlapping text
When baseline_words < 20 but gemini_words is large, or when shingling metrics \
are near-zero despite high LLM fidelity scores, TRUST THE LLM FIDELITY SCORES \
over programmatic text comparison signals. The LLM sees the actual PDF and HTML \
and can evaluate content correctly regardless of script or scan quality.
Structural programmatic signals (header/footer elements, displaced links, form \
elements) remain reliable because they read the extraction JSON structure, not \
PyMuPDF text."""

DECIDER_SCORING_BANDS = """\
Scoring Bands (Reference):
- fidelity_composite >= 0.85: Strong quality (primary AUTO_APPROVE pathway)
- fidelity_composite 0.60-0.85: Moderate quality (requires agreement from other sources)
- fidelity_composite < 0.60: Weak quality (HUMAN_REVIEW/MANDATORY_REVIEW pathway)
- fidelity_composite < 0.35: Critical defects (AUTO_REJECT pathway)
- shingling_recall >= 0.85: Content preservation strong
- shingling_recall < 0.75: Potential content loss
- word_count_ratio < 0.60: Possible truncation (< 0.30 = critical)

Signal agreement interpretation:
- converge_with_minors + fidelity_composite >= 0.85: AUTO_APPROVE
- partial_disagree: evaluate the specific concerns
- major_disagree: default to HUMAN_REVIEW"""

DECIDER_OUTPUT_FORMAT = """\
Respond with a JSON object. Fields are alphabetically prefixed to enforce \
processing order — you MUST write reasoning before the routing verdict.

{
  "a_reasoning": "2-4 sentences explaining your decision, referencing specific signal values and concerns",
  "b_signal_agreement": "converge or converge_with_minors or partial_disagree or major_disagree",
  "c_routing": "AUTO_APPROVE or HUMAN_REVIEW or REJECT",
  "d_confidence": 0.0-1.0,
  "e_critical_findings": ["one-liner per critical finding, empty array if none"],
  "f_major_findings": ["one-liner per major finding, empty array if none"],
  "g_minor_findings": ["one-liner per minor finding, empty array if none"],
  "h_action_items": ["what a human reviewer should focus on, empty array if AUTO_APPROVE"]
}"""


def build_final_decider_prompt(
    signals: "SignalBreakdown",
    concerns: "list[AuditConcern]",
    document_id: str = "unknown",
) -> str:
    """Build the complete Final Decider prompt."""
    signal_summary = _build_signal_summary(signals)
    concern_narratives = _build_concern_narratives(concerns)

    return f"""{DECIDER_ROLE}

## Document
ID: {document_id}

## Signal Summary
{json.dumps(signal_summary, indent=2)}

## Concerns Identified ({len(concerns)} total)
{concern_narratives}

## Routing Instructions
{DECIDER_ROUTING}

## Corroboration Analysis
{DECIDER_CORROBORATION}

## Scoring Bands
{DECIDER_SCORING_BANDS}

## Output Format
{DECIDER_OUTPUT_FORMAT}"""


# ============================================================================
# Helpers for Final Decider prompt building
# ============================================================================

def _build_signal_summary(signals: "SignalBreakdown") -> dict:
    summary: dict = {
        "llm_fidelity": {},
        "programmatic": {},
        "accessibility": {},
        "pipeline": {},
    }
    if signals.fidelity_available:
        summary["llm_fidelity"] = {
            "available": True,
            "composite_score": signals.fidelity_composite,
            "content_score": signals.fidelity_content,
            "structural_score": signals.fidelity_structural,
            "visual_score": signals.fidelity_visual,
            "routing": signals.fidelity_routing,
        }
    else:
        summary["llm_fidelity"] = {"available": False}

    prog: dict = {}
    if signals.shingling_recall is not None:
        prog["shingling_recall"] = round(signals.shingling_recall, 3)
    if signals.shingling_precision is not None:
        prog["shingling_precision"] = round(signals.shingling_precision, 3)
    if signals.duplication_ratio is not None:
        prog["duplication_ratio"] = round(signals.duplication_ratio, 3)
    if signals.word_count_ratio is not None:
        prog["word_count_ratio"] = round(signals.word_count_ratio, 3)
    if signals.baseline_words is not None:
        prog["baseline_words"] = signals.baseline_words
    if signals.table_count_ratio is not None:
        if signals.table_count_ratio < FABRICATED_TABLE_SENTINEL:
            prog["table_count_ratio"] = round(signals.table_count_ratio, 3)
        else:
            prog["table_count_note"] = "fabricated_tables_detected"
    if signals.image_count_ratio is not None:
        prog["image_count_ratio"] = round(signals.image_count_ratio, 3)
    summary["programmatic"] = prog

    if signals.axe_available:
        summary["accessibility"] = {
            "available": True,
            "total_violations": signals.axe_violations,
            "critical": signals.axe_critical,
            "serious": signals.axe_serious,
            "moderate": signals.axe_moderate,
            "minor": signals.axe_minor,
        }
    else:
        summary["accessibility"] = {"available": False}

    summary["pipeline"] = {
        "success": signals.pipeline_success,
        "confidence": signals.pipeline_confidence,
        "routing": signals.pipeline_routing,
    }
    summary["signal_agreement"] = signals.signal_agreement
    return summary


def _build_concern_narratives(concerns: "list[AuditConcern]") -> str:
    if not concerns:
        return "No concerns identified by any source. All signals indicate a clean document."
    severity_order = {"critical": 0, "major": 1, "minor": 2}
    sorted_concerns = sorted(concerns, key=lambda c: severity_order.get(c.severity, 3))
    lines: list[str] = []
    for i, c in enumerate(sorted_concerns, 1):
        corr = " [CORROBORATED]" if c.corroborated else ""
        check_ref = f" ({c.check_id})" if c.check_id else ""
        evidence_ref = f"\n   Evidence: {c.analysis}" if c.analysis else ""
        lines.append(
            f"{i}. [{c.severity.upper()}] [{c.source}]{check_ref}{corr}: {c.description}"
            f"{evidence_ref}"
        )
    return "\n".join(lines)

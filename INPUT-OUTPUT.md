# Pipeline Input-Output Quick Reference

Quick reference for pipeline script execution and file dependencies. Each script section includes: minimal CLI command, read/write source paths, and complete lists of input/output files with descriptions.

LAST UPDATED: 2026-03-18

## extract_structured_json.py

**CLI:** `python extract_structured_json.py`

**Read Source:** `../workspace/input` (configurable via DATA_FOLDER)  
**Write Source:** `../workspace/output` (configurable via OUTPUT_FOLDER)

**Reads:**
- `data/{doc_id}/source.pdf` — source PDF documents
- `.env` — configuration file
- `PROMPT_FOR_EXTRACT.md` — Gemini extraction prompt
- `PROMPT_FOR_VALIDATE.md` — Gemini validation prompt

**Writes:**
- `output/{doc_id}/result.json` — structured extraction data
- `output/{doc_id}/output.html` — HTML preview 
- `output/_reports/summary.json` — aggregate statistics
- `output/_reports/quality_report.html` — quality dashboard

## generate_sample_review.py (optional)

**CLI:** `python generate_sample_review.py`

**Read Source:** `../workspace/output`  
**Write Source:** `../workspace/output/_reports`

**Reads:**
- `output/*.json` — all extraction result files
- `data/{doc_id}/source.pdf` — source PDFs for page images

**Writes:**
- `output/_reports/sample_pages_review.html` — sample pages review report

## render_json.py

**CLI:** `python render_json.py <input> [-o <output>] [--raw]`

**Read Source:** CLI argument (`<input>`)  
**Write Source:** Alongside input or CLI argument (`-o <output>`)

**Reads:**
- `{input}.json` — extraction JSON file(s)

**Writes:**
- `{filename}.html` — ADA-compliant HTML rendering

## auditor.py

**CLI:** `python auditor.py output/doc.json --pdf data/doc/source.pdf --html output/doc.html`

**Read Source:** CLI arguments  
**Write Source:** Same directory as input JSON or CLI argument (`-o <output>`)

**Reads:**
- `{input}.json` — extraction JSON file
- `{pdf_path}` — source PDF file  
- `{html_path}` — rendered HTML file
- `.env` — configuration file
- `auditor_prompts.py` — LLM prompt templates
- `sanitize.py` — error sanitization utilities

**Writes:**
- `{doc_id}-audit-report.json` — comprehensive audit report with quality scores, routing decisions, and axe-core WCAG 2.1 AA accessibility compliance results (informational only, does not affect scoring or routing)

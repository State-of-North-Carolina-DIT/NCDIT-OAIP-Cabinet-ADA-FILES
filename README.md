# Pipeline — PDF to ADA-Compliant HTML

Conversion pipeline that extracts structured content from PDFs and renders ADA-compliant HTML. Designed for both local development and deployment as a Docker container.

## Scripts

| Script | Purpose |
|--------|---------|
| `src/extract_structured_json.py` | Extracts structured JSON (paragraphs, tables, images, etc.) from PDFs using Gemini |
| `src/render_json.py` | Renders extraction JSON to ADA-compliant HTML with remediation |
| `src/generate_sample_review.py` | Generates a sample review HTML for QA of extraction results |

## Quick Start

```bash
# Install dependencies
pip install -r src/requirements.txt

# Copy and configure environment
cp .env.dist .env
# Edit .env — at minimum set PROJECT_ID and DATA_FOLDER

# Place input PDFs in the expected structure
# workspace/input/{doc_id}/source.pdf

# Run extraction
python src/extract_structured_json.py

# Render JSON output to HTML (standalone, no .env needed)
python src/render_json.py ../../workspace/output/
```

## Configuration

Environment variables are loaded from `pipeline/.env` (see `.env.dist` for all options with defaults). Key settings:

- `PROJECT_ID` — GCP project for Vertex AI / Gemini
- `GEMINI_LOCATION` — Vertex AI location (default `global`)
- `DATA_FOLDER` — input directory (default `../workspace/input`)
- `OUTPUT_FOLDER` — output directory (default `../workspace/output`)

## Repo Structure

```
NCDIT-Cabinet-ADA-FILES/
├── {agency}/htmls/           ← 14 agencies (commerce, deq, dmva, doa, dpi, it, labor, ncagr, ncdcr, ncdhhs, ncdoi, ncdor, ncdps, nctreasurer)
│   ├── {doc_folder}/         ← one folder per document
│   │   ├── source.pdf        ← original PDF
│   │   ├── {name}.json       ← pipeline extraction output
│   │   ├── {name}.html       ← pipeline rendered HTML
│   │   ├── {name}-audit-report.json          ← audit report (newest auditor)
│   │   └── {name}-audit-report-baseline.json ← audit report (stable auditor)
│   ├── AUDITOR-REPORT.md              ← batch summary for this agency
│   ├── AUDITOR-REPORT-baseline.md     ← stable version batch summary
│   ├── audit-batch-results.json       ← raw batch results (newest)
│   └── audit-batch-results-baseline.json ← raw batch results (stable)
├── failed/                   ← 17 docs where the pipeline failed (source.pdf only)
│   └── failed.csv            ← list of failed docs with original URLs
└── cabinet-agencies/         ← CSV lists of all docs per agency
```

## Documentation

- [INPUT-OUTPUT.md](INPUT-OUTPUT.md) — detailed reference for all inputs, outputs, and file formats
- [CLAUDE.md](CLAUDE.md) — development guidelines

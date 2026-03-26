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

## Documentation

- [INPUT-OUTPUT.md](INPUT-OUTPUT.md) — detailed reference for all inputs, outputs, and file formats
- [CLAUDE.md](CLAUDE.md) — development guidelines

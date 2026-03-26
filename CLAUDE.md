# Pipeline

The pipeline converts government PDFs into WCAG 2.1 AA compliant HTML using Gemini AI. It operates in two modes: standalone CLI scripts for local/batch processing, and a unified FastAPI service deployed to Cloud Run.

## Team Ownership

Two teams contribute to this codebase with distinct responsibilities:

**Google RIT** (`@google.com`) — Pipeline scripts and AI prompts. Primary owners of:
- `src/extract_structured_json.py` — Gemini-based PDF extraction
- `src/render_json.py` — JSON-to-ADA-compliant HTML rendering
- `src/auditor.py` — Quality audit with fidelity scoring
- `src/auditor_prompts.py` — LLM prompt templates for auditor
- `src/generate_sample_review.py` — QA sample page review tool
- `src/PROMPT_FOR_EXTRACT.md` — Gemini extraction prompt
- `src/PROMPT_FOR_VALIDATE.md` — Gemini validation prompt
- `src/sanitize.py` — API key redaction utility

**Nerdery** (`@nerdery.com`) — Service layer, deployment, and analytics. Primary owners of:
- `src/main_service.py` — FastAPI service (production entrypoint)
- `src/lib/` — Service infrastructure (workspace, storage, metadata, runners)
- `src/ada_analytics/` — BigQuery analytics subsystem
- `Dockerfile`, `docker-compose.yaml` — Container configuration

**Shared files** (both teams may modify):
- `src/requirements.txt` — Python dependencies. Google adds pipeline deps, Nerdery adds service/analytics deps.
- `.env.dist` — Environment variable reference. Script-level vars (lines 7-55) are Google's domain; service-level vars (lines 58+) are Nerdery's domain.
- `INPUT-OUTPUT.md` — Script I/O contract documentation (see Documentation Upkeep below).

When making changes to shared files, be mindful that the other team depends on them. Avoid removing dependencies or environment variables without confirming they are unused by both teams.

## Architecture

### Two Execution Modes

**CLI mode (local/batch):** Run scripts directly from `src/`. Place PDFs in `workspace/input/{doc_id}/source.pdf`, then run each step sequentially:
```bash
python src/extract_structured_json.py
python src/render_json.py workspace/output/
python src/auditor.py output/doc.json --pdf data/doc/source.pdf --html output/doc.html
```

**Service mode (Cloud Run):** A single Docker image is deployed as three separate Cloud Run services. Each service hits a different endpoint on the same FastAPI app (`main_service.py`):

| Endpoint | Step | What it does |
|----------|------|--------------|
| `POST /extract` | 1 | PDF → structured JSON via Gemini |
| `POST /render` | 2 | JSON → ADA-compliant HTML |
| `POST /audit` | 3 | Quality audit with fidelity scoring |
| `GET /health` | — | Health check |

All endpoints accept `{"document_id": "...", "process_id": "..."}`.

### Service Request Flow

Each endpoint follows the same pattern:
1. Publish `"processing"` event to Pub/Sub (`lib/metadata.py`)
2. Create isolated temp workspace (`lib/workspace.py`)
3. Download required artifacts from GCS (`lib/storage.py`)
4. Run the pipeline script as a subprocess (`lib/runners/`)
5. Upload output artifacts to GCS
6. Record analytics to BigQuery (optional, `ada_analytics/`)
7. Publish `"completed"` or `"failed"` event to Pub/Sub
8. Clean up temp workspace

### Subprocess Isolation

The service wraps CLI scripts via `subprocess.run()` rather than importing them directly. This ensures full environment isolation per request. The runners in `lib/runners/` override env vars like `DATA_FOLDER`, `OUTPUT_FOLDER`, and `MAX_WORKERS` to match the temp workspace layout.

### GCS Artifact Registry

`lib/storage.py` maps logical artifact names to GCS paths and local workspace paths:
- `source_pdf` → `{tenant}/{doc_id}/source.pdf`
- `extracted_json` → `{tenant}/{doc_id}/artifacts/extracted.json`
- `extracted_html` → `{tenant}/{doc_id}/artifacts/extracted.html`
- `rendered_html` → `{tenant}/{doc_id}/artifacts/rendered.html`
- `audit_report` → `{tenant}/{doc_id}/artifacts/reports/audit.json`
- `summary_json` → `{tenant}/{doc_id}/artifacts/reports/summary.json`
- `quality_report_html` → `{tenant}/{doc_id}/artifacts/reports/quality.html`

If the scripts produce new output files, the artifact registry in `storage.py` and the `INPUT-OUTPUT.md` documentation must both be updated.

## Key Files

```
apps/pipeline/
├── Dockerfile                        # Python 3.13-slim, runs main_service.py on :8080
├── docker-compose.yaml               # Local dev: mounts gcloud creds, maps to :8081
├── .env.dist                         # All env vars with defaults and descriptions
├── INPUT-OUTPUT.md                   # Script I/O contract (keep in sync!)
├── src/
│   ├── main_service.py               # FastAPI service — production entrypoint
│   ├── extract_structured_json.py    # Step 1: PDF extraction via Gemini (~2000 lines)
│   ├── render_json.py                # Step 2: JSON → HTML with ADA remediation (~735 lines)
│   ├── auditor.py                    # Step 3: Quality audit with LLM scoring (~2300 lines)
│   ├── auditor_prompts.py            # Prompt templates for auditor fidelity checks
│   ├── generate_sample_review.py     # Optional QA report generator
│   ├── sanitize.py                   # API key redaction from error strings
│   ├── PROMPT_FOR_EXTRACT.md         # Gemini extraction prompt template
│   ├── PROMPT_FOR_VALIDATE.md        # Gemini validation prompt template
│   ├── requirements.txt              # Python dependencies (shared by both teams)
│   ├── lib/
│   │   ├── version.py                # Pipeline version: 2.{PIPELINE_BUILD}
│   │   ├── workspace.py              # Temp workspace context manager
│   │   ├── metadata.py               # Pub/Sub step event publisher
│   │   ├── storage.py                # GCS artifact download/upload + registry
│   │   └── runners/
│   │       ├── extract.py            # Subprocess wrapper for extraction
│   │       ├── render.py             # Subprocess wrapper for rendering
│   │       └── audit.py              # Subprocess wrapper for auditing
│   └── ada_analytics/
│       ├── config.py                 # BigQuery config via pydantic-settings
│       ├── models.py                 # BQ data models for metrics
│       ├── bigquery_sink.py          # BQ write sink (supports simulation mode)
│       ├── pipeline_collector.py     # Sync analytics collector for scripts
│       ├── processing_context.py     # ContextVar-based session tracking
│       └── stage_decorator.py        # @log_pipeline_stage async decorator
└── tests/
    └── __init__.py                   # (no tests yet)
```

## Environment Variables

See `.env.dist` for the full reference with defaults and descriptions. Key groupings:

**GCP/Gemini (script-level):** `PROJECT_ID`, `GEMINI_LOCATION`, `GEMINI_MODEL`, `MAX_OUTPUT_TOKENS`, `MAX_WORKERS`

**Feature flags:** `ENABLE_COHERENCE_CHECK`, `ENABLE_IMAGE_EXTRACTION`, `ENABLE_VIDEO_DETECTION`

**Directories:** `DATA_FOLDER`, `OUTPUT_FOLDER` (default to `../workspace/input` and `../workspace/output`)

**Service-level:** `GOOGLE_CLOUD_PROJECT`, `DOCUMENT_STORAGE_BUCKET`, `TENANT_ID`, `PIPELINE_EVENTS_TOPIC`, `ENVIRONMENT`

**Analytics:** `BIGQUERY_ANALYTICS_ENABLED`, `BIGQUERY_ANALYTICS_PROJECT_ID`

**Auditor-specific:** `FIDELITY_MODEL`, `FIDELITY_MAX_TOKENS`, `API_MODE`, `GEMINI_API_KEY`/`GEMINI_API_KEYS`

## Build & Run

### Local CLI (pipeline scripts)
```bash
cd apps/pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt
cp .env.dist .env   # edit .env — at minimum set PROJECT_ID
python src/extract_structured_json.py
python src/render_json.py workspace/output/
python src/auditor.py output/doc.json --pdf data/doc/source.pdf --html output/doc.html
```

### Local Docker (service mode)
```bash
cd apps/pipeline
cp .env.dist .env   # configure all required vars
docker compose up --build
# Service available at http://localhost:8081
```

### Production
The Dockerfile builds from `python:3.13-slim`, installs deps from `src/requirements.txt`, and runs `main_service.py` on port 8080 as a non-root user. The `PIPELINE_BUILD` build arg is injected at build time for versioning.

## Documentation Upkeep

Any changes that affect the inputs or outputs of the pipeline scripts must be reflected in `INPUT-OUTPUT.md`. This includes new/renamed files, changed directory structures, added/removed environment variables, and modified CLI arguments.

Scripts covered by this requirement:
- `extract_structured_json.py`
- `generate_sample_review.py`
- `render_json.py`
- `auditor.py`

Additionally, if new output artifacts are produced in service mode, update the `ARTIFACT_REGISTRY` in `src/lib/storage.py` so the service knows how to upload/download them.

## Design Patterns

- **Single image, multiple services:** One Docker image deployed three times to Cloud Run, each hitting a different endpoint.
- **Subprocess isolation:** The service wraps CLI scripts via `subprocess.run()` to ensure per-request isolation and avoid shared state.
- **Artifact registry:** `storage.py` decouples GCS blob layout from local file naming.
- **Context manager workspaces:** `workspace.py` creates and automatically cleans up temp directories per request.
- **Optional analytics:** All `ada_analytics` imports are guarded by `try/except ImportError`. BigQuery sink supports simulation mode (log-only) for local dev.
- **Error sanitization:** `sanitize.py` strips API keys from error messages before logging or returning them.
- **Multi-key rotation:** The auditor supports multiple Gemini API keys with rate-limit tracking and automatic rotation.

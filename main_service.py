#!/usr/bin/env python3
"""
Ada Compliance Engine - Unified Pipeline Service

Single FastAPI service deployable as extractor, renderer, or auditor.
Each request gets an isolated temp workspace that is cleaned up on completion.
"""

import asyncio
import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lib.workspace import create_workspace
from lib.metadata import publish_step_event
from lib.storage import download_artifacts, upload_artifacts
from lib.runners.extract import run_extraction
from lib.runners.render import run_render
from lib.runners.audit import run_audit

try:
    from ada_analytics import PipelineAnalyticsCollector
    _HAS_ANALYTICS = True
except ImportError:
    _HAS_ANALYTICS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TENANT_ID = os.getenv("TENANT_ID", "")


def _read_report(path: Path) -> str | None:
    """Read a report file and return its contents as a string, or None if absent."""
    try:
        return path.read_text(encoding="utf-8") if path.exists() else None
    except Exception:
        return None

app = FastAPI(
    title="Ada Compliance Engine - Pipeline Service",
    description="Unified service for extract, render, and audit pipelines",
    version="2.0.0",
)


class PipelineRequest(BaseModel):
    document_id: str = Field(..., description="Document identifier")
    process_id: str | None = Field(None, description="Optional process identifier")


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy"}


@app.post("/extract")
async def do_extract(request: PipelineRequest):
    """Run the extraction pipeline on a single document."""
    document_id = request.document_id
    try:
        await publish_step_event(document_id, "extract", "processing")

        with create_workspace(document_id) as ws:
            await download_artifacts(ws, ["source_pdf"])
            await asyncio.to_thread(run_extraction, ws)
            uploaded = await upload_artifacts(ws, [
                "extracted_json",
                "extracted_html",
                "summary_json",
                "quality_report_html",
            ])
            if _HAS_ANALYTICS:
                try:
                    collector = PipelineAnalyticsCollector(tenant_id=TENANT_ID)
                    collector.record_reports(document_id, {
                        "extracted": _read_report(ws.output_dir / f"{document_id}.json"),
                    })
                    collector.flush()
                except Exception:
                    logger.debug("Analytics error recording reports in /extract", exc_info=True)

        await publish_step_event(document_id, "extract", "completed", artifacts=list(uploaded.keys()))
        return {"status": "completed", "document_id": document_id, "outputs": uploaded}

    except Exception as e:
        logger.exception("Extraction failed for %s", document_id)
        await publish_step_event(document_id, "extract", "failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/render")
async def do_render(request: PipelineRequest):
    """Run the render pipeline on a single document."""
    document_id = request.document_id
    try:
        await publish_step_event(document_id, "render", "processing")

        with create_workspace(document_id) as ws:
            artifacts = await download_artifacts(ws, [
                "source_pdf",
                "extracted_json",
            ])
            await asyncio.to_thread(run_render, ws, artifacts["extracted_json"])
            uploaded = await upload_artifacts(ws, ["rendered_html"])

        await publish_step_event(document_id, "render", "completed", artifacts=list(uploaded.keys()))
        return {"status": "completed", "document_id": document_id, "outputs": uploaded}

    except Exception as e:
        logger.exception("Render failed for %s", document_id)
        await publish_step_event(document_id, "render", "failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audit")
async def do_audit(request: PipelineRequest):
    """Run the audit pipeline on a single document."""
    document_id = request.document_id
    try:
        await publish_step_event(document_id, "audit", "processing")

        with create_workspace(document_id) as ws:
            artifacts = await download_artifacts(ws, [
                "source_pdf",
                "extracted_json",
                "rendered_html",
            ])
            await asyncio.to_thread(run_audit, ws, artifacts["extracted_json"], artifacts["rendered_html"])
            uploaded = await upload_artifacts(ws, ["audit_report"])
            if _HAS_ANALYTICS:
                try:
                    collector = PipelineAnalyticsCollector(tenant_id=TENANT_ID)
                    collector.record_reports(document_id, {
                        "audit": _read_report(ws.output_dir / f"{document_id}-audit-report.json"),
                        "fidelity": _read_report(ws.output_dir / "fidelity-report.json"),
                        "axe": _read_report(ws.output_dir / "axe-report.json"),
                    })
                    collector.flush()
                except Exception:
                    logger.debug("Analytics error recording reports in /audit", exc_info=True)

        await publish_step_event(document_id, "audit", "completed", artifacts=list(uploaded.keys()))
        return {"status": "completed", "document_id": document_id, "outputs": uploaded}

    except Exception as e:
        logger.exception("Audit failed for %s", document_id)
        await publish_step_event(document_id, "audit", "failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

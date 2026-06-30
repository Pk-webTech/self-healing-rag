"""
self-healing-rag/api/routers/ingest.py
POST /ingest/text   — ingest raw text
POST /ingest/url    — fetch + ingest a URL
POST /ingest/files  — upload file(s) for ingestion
GET  /ingest/stats  — vector store stats
DELETE /ingest/reset — clear the vector store
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.deps import get_ingestion_pipeline
from api.schemas import IngestResponse, IngestTextRequest, IngestURLRequest, StoreStatsResponse
from core.config import get_settings
from ingestion import IngestionPipeline

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
settings = get_settings()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".html"}


@router.post("/text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(
    body: IngestTextRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Ingest a raw text string directly."""
    try:
        stats = pipeline.ingest_text(body.text, source=body.source)
        return IngestResponse(status="ok", **stats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/url", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_url(
    body: IngestURLRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Fetch a URL and ingest its content."""
    try:
        stats = pipeline.ingest_url(str(body.url))
        return IngestResponse(status="ok", **stats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/files", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_files(
    files: list[UploadFile] = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Upload one or more files (PDF/TXT/MD/HTML) for ingestion."""
    tmp_dir = Path(tempfile.mkdtemp())
    saved_paths: list[Path] = []

    try:
        for upload in files:
            suffix = Path(upload.filename or "file").suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXTENSIONS}",
                )
            dest = tmp_dir / (upload.filename or "upload")
            dest.write_bytes(await upload.read())
            saved_paths.append(dest)

        stats = pipeline.ingest_files(saved_paths)
        return IngestResponse(status="ok", **stats)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("/stats", response_model=StoreStatsResponse)
async def store_stats(
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> StoreStatsResponse:
    """Return current vector store statistics."""
    return StoreStatsResponse(
        total_chunks=pipeline.store.count(),
        provider=settings.vector_store_cfg.get("provider", "chroma"),
    )


@router.delete("/reset", status_code=status.HTTP_204_NO_CONTENT)
async def reset_store(
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> None:
    """⚠️ Delete all chunks from the vector store. Irreversible."""
    pipeline.store.reset()
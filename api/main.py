"""
self-healing-rag/api/main.py
FastAPI application factory.
Phase 1 routes: /query, /ingest, /health
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import ingest, query
from api.schemas import HealthResponse
from api.deps import _shared_store
from core.config import get_settings
from core.logger import logger

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Self-Healing RAG API starting up…")
    # Warm up shared singletons on startup
    store = _shared_store()
    logger.info(f"Vector store ready — {store.count()} chunks indexed")
    yield
    logger.info("🛑 Self-Healing RAG API shutting down")


def create_app() -> FastAPI:
    cfg = settings.yaml["app"]
    app = FastAPI(
        title="Self-Healing RAG API",
        description=(
            "Adaptive Retrieval-Augmented Generation with autonomous quality control "
            "and self-healing loops."
        ),
        version=cfg["version"],
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.yaml["api"]["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── routers ──────────────────────────────────────────────
    app.include_router(query.router)
    app.include_router(ingest.router)

    # ── health ───────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        store = _shared_store()
        return HealthResponse(
            status="ok",
            version=cfg["version"],
            vector_store_count=store.count(),
            timestamp=datetime.utcnow(),
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    api_cfg = settings.yaml["api"]
    uvicorn.run(
        "api.main:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        reload=settings.yaml["app"]["debug"],
        log_level=settings.log_level.lower(),
    )
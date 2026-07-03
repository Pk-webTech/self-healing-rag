"""
self-healing-rag/api/main.py
FastAPI application factory.
Phase 1: /query, /ingest, /health
Phase 3: /heal/events, /heal/stats, /heal/query-logs
Phase 4: /adapt/run, /adapt/stats
Phase 5: /metrics, /traces, /alerts/history, /alerts/window-stats
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routers import adapt, heal, ingest, observability, query
from api.schemas import HealthResponse
from api.deps import _shared_store, _shared_observability_manager
from core.config import get_settings
from core.logger import logger
from db.session import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Self-Healing RAG API starting up…")
    await init_db()
    store = _shared_store()
    logger.info(f"Vector store ready — {store.count()} chunks indexed")
    _shared_observability_manager()   # warm up singleton
    yield
    logger.info("🛑 Self-Healing RAG API shutting down")


def create_app() -> FastAPI:
    cfg = settings.yaml["app"]
    app = FastAPI(
        title="Self-Healing RAG API",
        description=(
            "Adaptive RAG with autonomous quality control, self-healing loops, "
            "adaptive learning, and full observability."
        ),
        version=cfg["version"],
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.yaml["api"]["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware (Phase 5) ───────────────────────────
    # Skips telemetry-only paths to avoid self-referential noise.
    _SKIP_PATHS = frozenset({"/metrics", "/health", "/docs", "/redoc", "/openapi.json"})

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        try:
            obs = _shared_observability_manager()
            if obs.metrics is not None:
                obs.metrics.query_latency_ms.observe(elapsed_ms)
        except Exception:
            pass
        return response

    # ── routers ──────────────────────────────────────────────────────
    app.include_router(query.router)
    app.include_router(ingest.router)
    app.include_router(heal.router)
    app.include_router(adapt.router)
    app.include_router(observability.router)

    # ── health ───────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        store = _shared_store()
        try:
            count = store.count()
            _shared_observability_manager().update_store_metrics(count)
        except Exception:
            count = 0
        return HealthResponse(
            status="ok",
            version=cfg["version"],
            vector_store_count=count,
            timestamp=datetime.now(timezone.utc),
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
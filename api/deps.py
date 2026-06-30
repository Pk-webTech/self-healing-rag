"""
self-healing-rag/api/deps.py
FastAPI dependency providers — singletons shared across requests.
"""
from __future__ import annotations

from functools import lru_cache

from ingestion import IngestionPipeline
from ingestion.embedder import Embedder
from ingestion.vector_store import build_vector_store
from generation.generator import Generator
from retrieval import RetrievalPipeline


@lru_cache(maxsize=1)
def _shared_store():
    return build_vector_store()


@lru_cache(maxsize=1)
def _shared_embedder():
    return Embedder()


@lru_cache(maxsize=1)
def _shared_ingestion_pipeline():
    store = _shared_store()
    embedder = _shared_embedder()
    return IngestionPipeline(vector_store=store, embedder=embedder)


@lru_cache(maxsize=1)
def _shared_retrieval_pipeline():
    store = _shared_store()
    embedder = _shared_embedder()
    return RetrievalPipeline(store=store, embedder=embedder)


@lru_cache(maxsize=1)
def _shared_generator():
    return Generator()


def get_ingestion_pipeline() -> IngestionPipeline:
    return _shared_ingestion_pipeline()


def get_retrieval_pipeline() -> RetrievalPipeline:
    return _shared_retrieval_pipeline()


def get_generation_pipeline() -> Generator:
    return _shared_generator()
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
from evaluation import EvaluationPipeline
from healing import HealingLoop
from adaptation import AdaptiveLearner
from observability import ObservabilityManager


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


@lru_cache(maxsize=1)
def _shared_evaluation_pipeline():
    return EvaluationPipeline()


def get_ingestion_pipeline() -> IngestionPipeline:
    return _shared_ingestion_pipeline()


def get_retrieval_pipeline() -> RetrievalPipeline:
    return _shared_retrieval_pipeline()


def get_generation_pipeline() -> Generator:
    return _shared_generator()


def get_evaluation_pipeline() -> EvaluationPipeline:
    return _shared_evaluation_pipeline()


@lru_cache(maxsize=1)
def _shared_healing_loop():
    store = _shared_store()
    embedder = _shared_embedder()
    retrieval = _shared_retrieval_pipeline()
    generator = _shared_generator()
    evaluator = _shared_evaluation_pipeline()
    return HealingLoop(
        retrieval_pipeline=retrieval,
        generator=generator,
        evaluator=evaluator,
        store=store,
        embedder=embedder,
    )


def get_healing_loop() -> HealingLoop:
    return _shared_healing_loop()


@lru_cache(maxsize=1)
def _shared_adaptive_learner():
    store = _shared_store()
    retrieval = _shared_retrieval_pipeline()
    return AdaptiveLearner(store=store, retrieval_pipeline=retrieval)


def get_adaptive_learner() -> AdaptiveLearner:
    return _shared_adaptive_learner()


@lru_cache(maxsize=1)
def _shared_observability_manager():
    return ObservabilityManager()


def get_observability_manager() -> ObservabilityManager:
    return _shared_observability_manager()
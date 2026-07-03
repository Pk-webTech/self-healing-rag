"""
self-healing-rag/core/config.py
Centralised settings loaded from .env + configs/config.yaml.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"
PROMPTS_PATH = ROOT / "configs" / "prompts.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM providers ────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    generation_provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="openai", alias="GENERATION_PROVIDER"
    )
    embedding_provider: Literal["openai", "huggingface"] = Field(
        default="openai", alias="EMBEDDING_PROVIDER"
    )

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", alias="OLLAMA_MODEL")

    # ── Security ─────────────────────────────────────────────
    api_secret_key: str = Field(default="dev-secret", alias="API_SECRET_KEY")
    api_key_header: str = Field(default="X-API-Key", alias="API_KEY_HEADER")

    # ── Database ─────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite+aiosqlite:///data/self_healing_rag.db",
        alias="DATABASE_URL",
    )

    # ── Vector store ─────────────────────────────────────────
    chroma_persist_dir: str = Field(default="data/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field(default="self_healing_rag", alias="CHROMA_COLLECTION")

    # ── Logging ──────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ── YAML config (loaded separately) ──────────────────────
    @property
    def yaml(self) -> dict:
        return _load_yaml(CONFIG_PATH)

    @property
    def prompts(self) -> dict:
        return _load_yaml(PROMPTS_PATH)

    # ── Shortcut accessors ───────────────────────────────────
    @property
    def ingestion(self) -> dict:
        return self.yaml["ingestion"]

    @property
    def embedding_cfg(self) -> dict:
        return self.yaml["embedding"]

    @property
    def vector_store_cfg(self) -> dict:
        return self.yaml["vector_store"]

    @property
    def retrieval_cfg(self) -> dict:
        return self.yaml["retrieval"]

    @property
    def generation_cfg(self) -> dict:
        return self.yaml["generation"]

    @property
    def evaluation_cfg(self) -> dict:
        return self.yaml["evaluation"]

    @property
    def adaptation_cfg(self) -> dict:
        return self.yaml["adaptation"]

    @property
    def observability_cfg(self) -> dict:
        return self.yaml["observability"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
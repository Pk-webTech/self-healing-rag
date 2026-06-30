"""
self-healing-rag/generation/generator.py
LLM wrapper supporting OpenAI, Anthropic, and Ollama.
Includes latency tracking, token counting, and source extraction.
"""
from __future__ import annotations

import re
import time
from typing import Literal

from core.config import get_settings
from core.logger import logger
from core.models import GenerationResult, RetrievalResult
from generation.prompt_templates import PromptBuilder

settings = get_settings()


def _extract_sources(context_text: str) -> list[str]:
    """Pull [Source N: <name>] labels from the context block."""
    return re.findall(r"\[Source \d+: ([^\|]+?)\s*\|", context_text)


# ── provider implementations ──────────────────────────────────

def _generate_openai(system: str, user: str) -> tuple[str, int, int]:
    from openai import OpenAI
    cfg = settings.generation_cfg
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=cfg["openai_model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )
    answer = resp.choices[0].message.content.strip()
    prompt_tok = resp.usage.prompt_tokens
    completion_tok = resp.usage.completion_tokens
    return answer, prompt_tok, completion_tok


def _generate_anthropic(system: str, user: str) -> tuple[str, int, int]:
    import anthropic
    cfg = settings.generation_cfg
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model=cfg["anthropic_model"],
        max_tokens=cfg["max_tokens"],
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=cfg["temperature"],
    )
    answer = msg.content[0].text.strip()
    prompt_tok = msg.usage.input_tokens
    completion_tok = msg.usage.output_tokens
    return answer, prompt_tok, completion_tok


def _generate_ollama(system: str, user: str) -> tuple[str, int, int]:
    import requests
    cfg = settings.generation_cfg
    payload = {
        "model": settings.ollama_model,
        "prompt": f"{system}\n\n{user}",
        "stream": False,
        "options": {
            "temperature": cfg["temperature"],
            "num_predict": cfg["max_tokens"],
        },
    }
    resp = requests.post(
        f"{settings.ollama_base_url}/api/generate", json=payload, timeout=120
    )
    resp.raise_for_status()
    data = resp.json()
    answer = data.get("response", "").strip()
    prompt_tok = data.get("prompt_eval_count", 0)
    completion_tok = data.get("eval_count", 0)
    return answer, prompt_tok, completion_tok


_PROVIDERS = {
    "openai": _generate_openai,
    "anthropic": _generate_anthropic,
    "ollama": _generate_ollama,
}


# ── public class ──────────────────────────────────────────────

class Generator:
    """
    LLM generation wrapper.

    Usage:
        gen = Generator()
        result: GenerationResult = gen.generate(retrieval_result)
    """

    def __init__(
        self,
        provider: Literal["openai", "anthropic", "ollama"] | None = None,
    ) -> None:
        self.provider = provider or settings.generation_provider
        if self.provider not in _PROVIDERS:
            raise ValueError(f"Unknown generation provider: {self.provider}")
        self._fn = _PROVIDERS[self.provider]
        self._pb = PromptBuilder()
        logger.info(f"Generator initialised: provider={self.provider}")

    def generate(self, retrieval: RetrievalResult) -> GenerationResult:
        system, user = self._pb.rag_prompt(
            context=retrieval.context_text,
            query=retrieval.query,
        )

        t0 = time.perf_counter()
        try:
            answer, prompt_tok, completion_tok = self._fn(system, user)
        except Exception as exc:
            logger.error(f"LLM generation failed: {exc}")
            raise
        latency_ms = (time.perf_counter() - t0) * 1000

        sources = _extract_sources(retrieval.context_text)

        # determine model name for logging
        cfg = settings.generation_cfg
        model_name = {
            "openai": cfg.get("openai_model", "gpt-4o-mini"),
            "anthropic": cfg.get("anthropic_model", "claude-3-5-haiku"),
            "ollama": settings.ollama_model,
        }.get(self.provider, self.provider)

        logger.info(
            f"Generated answer: provider={self.provider} model={model_name} "
            f"tokens={prompt_tok}+{completion_tok} latency={latency_ms:.0f}ms"
        )

        return GenerationResult(
            answer=answer,
            query=retrieval.query,
            context_text=retrieval.context_text,
            sources=sources,
            model=model_name,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            latency_ms=latency_ms,
        )
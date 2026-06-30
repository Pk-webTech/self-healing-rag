"""
self-healing-rag/retrieval/query_processor.py
Query expansion strategies:
  - identity (no-op): return original query
  - HyDE: generate a hypothetical answer, embed that instead
  - multi_query: generate N rephrasings, retrieve for each
"""
from __future__ import annotations

import json

from core.config import get_settings
from core.logger import logger

settings = get_settings()


def _call_llm(system: str, user: str, temperature: float = 0.5) -> str:
    """Minimal LLM call for query expansion (uses generation provider)."""
    provider = settings.generation_provider
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.generation_cfg["openai_model"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        msg = client.messages.create(
            model=settings.generation_cfg["anthropic_model"],
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()
    if provider == "ollama":
        import requests
        payload = {
            "model": settings.ollama_model,
            "prompt": f"{system}\n\n{user}",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 300},
        }
        resp = requests.post(
            f"{settings.ollama_base_url}/api/generate", json=payload, timeout=60
        )
        return resp.json().get("response", "").strip()
    raise ValueError(f"Unknown provider: {provider}")


class QueryProcessor:
    """
    Optionally transforms the raw user query before retrieval.

    Usage:
        qp = QueryProcessor()
        queries = qp.process("What is RAG?")
        # → ["What is RAG?"]  (identity, or expanded list)
    """

    def __init__(
        self,
        use_hyde: bool | None = None,
        use_multi_query: bool | None = None,
        n_multi_query: int = 3,
    ) -> None:
        cfg = settings.retrieval_cfg
        self.use_hyde = use_hyde if use_hyde is not None else cfg.get("use_hyde", False)
        self.use_multi_query = (
            use_multi_query
            if use_multi_query is not None
            else cfg.get("use_multi_query", False)
        )
        self.n_multi_query = n_multi_query

    def process(self, query: str) -> list[str]:
        """Return a list of query strings to retrieve for."""
        if self.use_hyde:
            return [self._hyde(query)]
        if self.use_multi_query:
            return self._multi_query(query)
        return [query]

    # ── strategies ────────────────────────────────────────────

    def _hyde(self, query: str) -> str:
        """HyDE: embed a hypothetical document instead of the raw query."""
        prompts = settings.prompts
        try:
            hypo = _call_llm(
                system=prompts["hyde_system"],
                user=prompts["hyde_user"].format(query=query),
            )
            logger.debug(f"HyDE passage: {hypo[:80]}…")
            return hypo
        except Exception as exc:
            logger.warning(f"HyDE failed ({exc}), falling back to original query")
            return query

    def _multi_query(self, query: str) -> list[str]:
        """Generate N rephrasings of the query."""
        prompts = settings.prompts
        try:
            raw = _call_llm(
                system=prompts["multi_query_system"].format(n=self.n_multi_query),
                user=prompts["multi_query_user"].format(query=query),
                temperature=0.6,
            )
            # strip markdown code fences if present
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            queries = json.loads(raw)
            if not isinstance(queries, list):
                raise ValueError("Expected a JSON array")
            result = [query] + [q for q in queries if isinstance(q, str)]
            logger.debug(f"Multi-query expansions: {result}")
            return result[:self.n_multi_query + 1]
        except Exception as exc:
            logger.warning(f"Multi-query failed ({exc}), falling back to original query")
            return [query]
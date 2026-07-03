"""
self-healing-rag/core/llm_client.py
Shared lightweight LLM call helper, used by any module that needs a single
non-streaming completion (query expansion, evaluation judges, healing actions)
without going through the full Generator/RetrievalResult machinery.

NOTE: This consolidates what was previously duplicated logic in
retrieval/query_processor.py — kept here as the single source of truth.
max_tokens and temperature are now caller-controlled (previously hardcoded
to 300, which silently truncated longer JSON judge responses).
"""
from __future__ import annotations

from core.config import get_settings
from core.logger import logger

settings = get_settings()


def call_llm(
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
    provider: str | None = None,
) -> str:
    """
    Minimal single-turn LLM call. Returns the raw text response.
    Raises on provider/network failure — caller is responsible for try/except
    and fallback behaviour (judges and healing actions must never crash the
    pipeline on a transient LLM error).
    """
    prov = provider or settings.generation_provider

    if prov == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.generation_cfg["openai_model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    if prov == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        msg = client.messages.create(
            model=settings.generation_cfg["anthropic_model"],
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        return msg.content[0].text.strip()

    if prov == "ollama":
        import requests
        payload = {
            "model": settings.ollama_model,
            "prompt": f"{system}\n\n{user}",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = requests.post(
            f"{settings.ollama_base_url}/api/generate", json=payload, timeout=90
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    raise ValueError(f"Unknown LLM provider: {prov}")


def call_llm_json(
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 600,
    provider: str | None = None,
) -> dict:
    """
    Call the LLM and parse a JSON object from the response.
    Strips markdown code fences defensively (models often wrap JSON in
    ```json ... ``` even when told not to).
    Raises json.JSONDecodeError on unparseable output — caller must handle.
    """
    import json

    raw = call_llm(system, user, temperature=temperature, max_tokens=max_tokens, provider=provider)
    cleaned = raw.strip()

    # strip common markdown fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        cleaned = cleaned.removeprefix("json").strip()
    cleaned = cleaned.strip("`").strip()

    # some models prepend prose before the JSON object — extract the first {...} block
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    return json.loads(cleaned)
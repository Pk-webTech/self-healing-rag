"""
self-healing-rag/generation/prompt_templates.py
Loads and formats prompt templates from configs/prompts.yaml.
"""
from __future__ import annotations

from core.config import get_settings

settings = get_settings()


class PromptBuilder:
    """
    Builds structured prompts for the LLM.

    Usage:
        pb = PromptBuilder()
        system, user = pb.rag_prompt(context="...", query="What is X?")
    """

    def __init__(self) -> None:
        self._prompts = settings.prompts

    def rag_prompt(self, context: str, query: str) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt) for RAG generation."""
        system = self._prompts["rag_system"]
        user = self._prompts["rag_user"].format(context=context, query=query)
        return system, user
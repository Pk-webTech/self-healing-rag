"""
self-healing-rag/ingestion/chunker.py
Chunking strategies:
  - recursive: split on paragraph/sentence/word boundaries (default)
  - semantic:  split on double-newlines respecting a max token budget
"""
from __future__ import annotations

import re
import uuid
from typing import Literal

import tiktoken

from core.config import get_settings
from core.logger import logger
from core.models import Chunk, Document

settings = get_settings()
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _gen_chunk_id(source: str, index: int) -> str:
    base = f"{source}::{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


# ── recursive splitter ────────────────────────────────────────

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Port of LangChain's RecursiveCharacterTextSplitter logic."""
    if separators is None:
        separators = _SEPARATORS

    def _split(text: str, seps: list[str]) -> list[str]:
        sep = seps[0] if seps else ""
        splits = text.split(sep) if sep else list(text)
        chunks, current, current_len = [], [], 0

        for s in splits:
            s_len = _count_tokens(s)
            if current_len + s_len + (1 if current else 0) > chunk_size and current:
                chunk_text = sep.join(current).strip()
                if _count_tokens(chunk_text) >= settings.ingestion["min_chunk_length"]:
                    chunks.append(chunk_text)
                # keep overlap
                overlap_buf, overlap_len = [], 0
                for piece in reversed(current):
                    piece_len = _count_tokens(piece)
                    if overlap_len + piece_len > chunk_overlap:
                        break
                    overlap_buf.insert(0, piece)
                    overlap_len += piece_len
                current = overlap_buf
                current_len = overlap_len

            if s_len > chunk_size and seps[1:]:
                # recurse with finer separators
                sub_chunks = _split(s, seps[1:])
                chunks.extend(sub_chunks)
            else:
                current.append(s)
                current_len += s_len + (1 if sep else 0)

        if current:
            last = sep.join(current).strip()
            if _count_tokens(last) >= settings.ingestion["min_chunk_length"]:
                chunks.append(last)
        return chunks

    return _split(text, separators)


# ── semantic splitter ─────────────────────────────────────────

def _semantic_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split on paragraph boundaries (\n\n). If a paragraph still exceeds
    chunk_size, fall back to recursive splitting on that paragraph.
    """
    paragraphs = re.split(r"\n\n+", text)
    chunks, current, current_len = [], [], 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = _count_tokens(para)

        if para_len > chunk_size:
            # flush current buffer first
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            # recursively split oversized paragraph
            sub = _recursive_split(para, chunk_size, chunk_overlap)
            chunks.extend(sub)
        elif current_len + para_len > chunk_size and current:
            chunks.append("\n\n".join(current))
            # carry overlap
            current = current[-1:] if current else []
            current_len = _count_tokens(current[0]) if current else 0
            current.append(para)
            current_len += para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if _count_tokens(c) >= settings.ingestion["min_chunk_length"]]


# ── public API ────────────────────────────────────────────────

class Chunker:
    """
    Convert Document objects into a flat list of Chunk objects.

    Usage:
        chunker = Chunker()
        chunks = chunker.chunk(doc)
        all_chunks = chunker.chunk_many(docs)
    """

    def __init__(
        self,
        strategy: Literal["recursive", "semantic"] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        cfg = settings.ingestion
        self.strategy = strategy or cfg["chunking_strategy"]
        self.chunk_size = chunk_size or cfg["chunk_size"]
        self.chunk_overlap = chunk_overlap or cfg["chunk_overlap"]

    def _split_text(self, text: str) -> list[str]:
        if self.strategy == "semantic":
            return _semantic_split(text, self.chunk_size, self.chunk_overlap)
        return _recursive_split(text, self.chunk_size, self.chunk_overlap)

    def chunk(self, doc: Document) -> list[Chunk]:
        raw_chunks = self._split_text(doc.content)
        logger.debug(
            f"Chunked '{doc.metadata.get('filename', doc.source)}' → "
            f"{len(raw_chunks)} chunks (strategy={self.strategy})"
        )
        chunks = []
        for i, text in enumerate(raw_chunks):
            cid = _gen_chunk_id(doc.source, i)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    content=text,
                    source=doc.source,
                    doc_type=doc.doc_type,
                    chunk_index=i,
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": self.strategy,
                        "token_count": _count_tokens(text),
                    },
                )
            )
        return chunks

    def chunk_many(self, docs: list[Document]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))
        logger.info(f"Total chunks produced: {len(all_chunks)}")
        return all_chunks
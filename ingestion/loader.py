"""
self-healing-rag/ingestion/loader.py
Multi-format document loader.
Supports: PDF, TXT, MD, HTML, and web URLs.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from bs4 import BeautifulSoup

from core.config import get_settings
from core.logger import logger
from core.models import Document

if TYPE_CHECKING:
    pass

settings = get_settings()


# ── helpers ──────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _clean(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── loaders ──────────────────────────────────────────────────

def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("Install pypdf: pip install pypdf") from e

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(pages)


def _load_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (SelfHealingRAG/0.1)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    ct = resp.headers.get("content-type", "")
    if "pdf" in ct:
        # save temp and re-use pdf loader
        tmp = Path("/tmp/_shr_web.pdf")
        tmp.write_bytes(resp.content)
        return _load_pdf(tmp)
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


# ── public API ───────────────────────────────────────────────

class DocumentLoader:
    """
    Load one or many documents from file paths or URLs.

    Usage:
        loader = DocumentLoader()
        docs = loader.load_file("path/to/file.pdf")
        docs = loader.load_directory("data/raw/")
        docs = loader.load_url("https://example.com/paper.pdf")
    """

    EXTENSION_MAP: dict[str, str] = {
        ".pdf": "pdf",
        ".txt": "text",
        ".md": "markdown",
        ".markdown": "markdown",
        ".html": "html",
        ".htm": "html",
    }

    def load_file(self, path: str | Path) -> Document:
        path = Path(path)
        ext = path.suffix.lower()
        doc_type = self.EXTENSION_MAP.get(ext)
        if doc_type is None:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.info(f"Loading {doc_type.upper()} → {path.name}")

        loaders = {
            "pdf": _load_pdf,
            "text": _load_txt,
            "markdown": _load_markdown,
            "html": _load_html,
        }
        raw = loaders[doc_type](path)
        content = _clean(raw)

        if not content:
            raise ValueError(f"Empty document after loading: {path}")

        return Document(
            content=content,
            source=str(path.resolve()),
            doc_type=doc_type,
            metadata={
                "filename": path.name,
                "file_size_bytes": path.stat().st_size,
                "content_hash": _sha256(content),
            },
        )

    def load_url(self, url: str) -> Document:
        logger.info(f"Loading URL → {url}")
        raw = _load_url(url)
        content = _clean(raw)
        if not content:
            raise ValueError(f"Empty content from URL: {url}")
        return Document(
            content=content,
            source=url,
            doc_type="html",
            metadata={"url": url, "content_hash": _sha256(content)},
        )

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> list[Document]:
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        supported = set(self.EXTENSION_MAP.keys())
        paths = [p for p in directory.glob(pattern) if p.suffix.lower() in supported]

        logger.info(f"Found {len(paths)} document(s) in {directory}")
        docs: list[Document] = []
        for p in paths:
            try:
                docs.append(self.load_file(p))
            except Exception as exc:
                logger.warning(f"Skipping {p.name}: {exc}")
        return docs

    def load_many(self, sources: list[str]) -> list[Document]:
        """
        Load from a mixed list of file paths and URLs.
        """
        docs: list[Document] = []
        for src in sources:
            try:
                if src.startswith("http://") or src.startswith("https://"):
                    docs.append(self.load_url(src))
                else:
                    docs.append(self.load_file(src))
            except Exception as exc:
                logger.warning(f"Skipping source '{src}': {exc}")
        return docs
"""Builds the real DocumentService from settings (.env / environment variables).

Kept out of main.py so the API and the single-process Streamlit demo share it.
"""
import tempfile
from pathlib import Path

from app import config
from app.embeddings import Embedder, make_embedder
from app.llm import LLM, make_llm
from app.service import DocumentService
from app.vectorstore import InMemoryStore


def build_store():
    if config.STORE.lower().strip() == "memory":
        return InMemoryStore()
    if config.STORE.lower().strip() == "chroma":
        from app.vectorstore import ChromaStore

        return ChromaStore(path=str(config.ROOT / "chroma_db"))
    raise ValueError(f"Unknown STORE '{config.STORE}'. Use 'chroma' or 'memory'.")


def build_default_service() -> DocumentService:
    return DocumentService(
        store=build_store(),
        embedder=make_embedder(config.EMBEDDER),
        llm=make_llm(config.LLM_PROVIDER, config),
        upload_dir=config.ROOT / "uploads",
        k=config.TOP_K,
        min_score=config.MIN_SCORE,
    )


def build_session_service(
    embedder: Embedder, llm: LLM, max_bytes: int = 5 * 1024 * 1024
) -> DocumentService:
    """A private service for one demo visitor: its own empty in-memory index, so
    visitors never see each other's documents. The (heavy) embedder and the LLM
    client are shared and passed in."""
    return DocumentService(
        store=InMemoryStore(),
        embedder=embedder,
        llm=llm,
        upload_dir=Path(tempfile.mkdtemp(prefix="rag-demo-")),
        k=config.TOP_K,
        min_score=config.MIN_SCORE,
        max_bytes=max_bytes,
    )

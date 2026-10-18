"""Runs the same checks against the real Chroma store (skipped if chromadb
is not installed)."""
import pytest

pytest.importorskip("chromadb")

from app.embeddings import HashingEmbedder
from app.retrieval import retrieve
from app.vectorstore import ChromaStore
from tests.test_vectorstore import make_store


def test_chroma_store_end_to_end(tmp_path):
    store = make_store(ChromaStore(path=str(tmp_path / "db")))
    assert store.count() == 4
    hits = retrieve(store, HashingEmbedder(), "Which moons does Mars have", k=1)
    assert hits[0].source == "space.pdf" and hits[0].page == 2
    assert 0.0 < hits[0].score <= 1.0

    store.delete_source("space.pdf")
    assert store.count() == 2


def test_chroma_data_survives_restart(tmp_path):
    path = str(tmp_path / "db")
    make_store(ChromaStore(path=path))
    assert ChromaStore(path=path).count() == 4


def test_chroma_lists_sources_with_chunk_counts(tmp_path):
    store = make_store(ChromaStore(path=str(tmp_path / "db")))
    assert store.list_sources() == {"cats.pdf": 2, "space.pdf": 2}
    store.delete_source("cats.pdf")
    assert store.list_sources() == {"space.pdf": 2}

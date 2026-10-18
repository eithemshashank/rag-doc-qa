import pytest

from app.embeddings import HashingEmbedder
from app.ingest import Chunk
from app.retrieval import index_chunks, retrieve
from app.vectorstore import InMemoryStore

DOCS = [
    ("cats.pdf", 1, "Cats are small carnivorous mammals that purr and hunt mice."),
    ("cats.pdf", 2, "A kitten needs milk and warmth during its first weeks."),
    ("space.pdf", 1, "The Moon orbits the Earth and causes ocean tides."),
    ("space.pdf", 2, "Mars has two small moons called Phobos and Deimos."),
]


def make_chunks():
    return [Chunk(text, src, page, f"{src}-p{page}-c1") for src, page, text in DOCS]


def make_store(store=None):
    store = store or InMemoryStore()
    index_chunks(store, HashingEmbedder(), make_chunks())
    return store


def test_index_counts_chunks():
    assert make_store().count() == 4


def test_retrieval_finds_the_right_chunk():
    hits = retrieve(make_store(), HashingEmbedder(), "What do kitten need to survive milk", k=1)
    assert hits[0].source == "cats.pdf" and hits[0].page == 2

    hits = retrieve(make_store(), HashingEmbedder(), "Which moons does Mars have", k=1)
    assert hits[0].source == "space.pdf" and hits[0].page == 2


def test_results_are_sorted_by_score():
    hits = retrieve(make_store(), HashingEmbedder(), "moon tides ocean", k=4)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert len(hits) == 4


def test_empty_store_returns_nothing():
    assert retrieve(InMemoryStore(), HashingEmbedder(), "anything") == []


def test_delete_source_removes_its_chunks():
    store = make_store()
    store.delete_source("cats.pdf")
    assert store.count() == 2
    hits = retrieve(store, HashingEmbedder(), "kitten milk", k=4)
    assert all(h.source == "space.pdf" for h in hits)


def test_adding_same_chunk_id_replaces_it():
    store = make_store()
    index_chunks(store, HashingEmbedder(), make_chunks()[:1])
    assert store.count() == 4


def test_mismatched_embeddings_raise():
    with pytest.raises(ValueError):
        InMemoryStore().add(make_chunks(), [[0.1, 0.2]])

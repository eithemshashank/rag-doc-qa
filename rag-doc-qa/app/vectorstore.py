"""Step 2b: store chunk vectors and search them.

Two stores with the same interface:
- ChromaStore: the real vector database (saved on disk).
- InMemoryStore: a simple numpy version used in tests and to show how
  similarity search works.

Vectors are normalised, so cosine similarity is just a dot product.
"""
from dataclasses import dataclass

from collections import Counter

import numpy as np

from app.ingest import Chunk


@dataclass
class Hit:
    text: str
    source: str
    page: int
    score: float  # cosine similarity, higher is more similar (max 1.0)


class InMemoryStore:
    def __init__(self):
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("need exactly one embedding per chunk")
        for chunk, vec in zip(chunks, embeddings):
            self._chunks[chunk.chunk_id] = chunk  # same id replaces the old chunk
            self._vectors[chunk.chunk_id] = np.asarray(vec, dtype=float)

    def search(self, query_embedding: list[float], k: int = 4) -> list[Hit]:
        if not self._chunks:
            return []
        ids = list(self._chunks)
        matrix = np.vstack([self._vectors[i] for i in ids])
        scores = matrix @ np.asarray(query_embedding, dtype=float)
        best = np.argsort(scores)[::-1][:k]
        return [
            Hit(self._chunks[ids[i]].text, self._chunks[ids[i]].source,
                self._chunks[ids[i]].page, float(scores[i]))
            for i in best
        ]

    def delete_source(self, source: str) -> None:
        for chunk_id in [i for i, c in self._chunks.items() if c.source == source]:
            del self._chunks[chunk_id], self._vectors[chunk_id]

    def list_sources(self) -> dict[str, int]:
        """Return {file name: number of chunks} for everything stored."""
        return dict(Counter(c.source for c in self._chunks.values()))

    def count(self) -> int:
        return len(self._chunks)


class ChromaStore:
    def __init__(self, path: str = "chroma_db", collection: str = "documents"):
        import chromadb  # imported lazily

        client = chromadb.PersistentClient(path=path)
        self._col = client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("need exactly one embedding per chunk")
        if not chunks:
            return
        self._col.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )

    def search(self, query_embedding: list[float], k: int = 4) -> list[Hit]:
        if self._col.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._col.count()),
        )
        hits = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            # Chroma returns cosine *distance*; similarity = 1 - distance
            hits.append(Hit(text, meta["source"], int(meta["page"]), 1.0 - float(dist)))
        return hits

    def delete_source(self, source: str) -> None:
        self._col.delete(where={"source": source})

    def list_sources(self) -> dict[str, int]:
        """Return {file name: number of chunks} for everything stored."""
        if self._col.count() == 0:
            return {}
        metadatas = self._col.get(include=["metadatas"])["metadatas"]
        return dict(Counter(m["source"] for m in metadatas))

    def count(self) -> int:
        return self._col.count()

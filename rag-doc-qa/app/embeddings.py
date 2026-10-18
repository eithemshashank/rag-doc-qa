"""Step 2a: turn text into vectors ("embeddings").

An embedding is a list of numbers that captures the meaning of a text. Texts
with similar meaning get vectors that point in similar directions, so we can
find relevant chunks by comparing vectors instead of matching exact words.
"""
import re
import zlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text."""


class SentenceTransformerEmbedder:
    """Real embeddings from a free local model (no API key needed).

    The model is downloaded the first time it runs (about 90 MB).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported lazily

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=float).tolist()


class HashingEmbedder:
    """A tiny stand-in embedder used only in tests.

    It hashes each word into a slot of a fixed-size vector, so texts that share
    words get similar vectors. It does NOT understand meaning (synonyms won't
    match), but it is fast, deterministic, and needs no downloads.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = np.zeros(self.dim)
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[zlib.crc32(word.encode()) % self.dim] += 1.0
            norm = np.linalg.norm(vec)
            out.append((vec / norm if norm else vec).tolist())
        return out

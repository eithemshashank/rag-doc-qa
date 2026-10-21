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


class OnnxEmbedder:
    """The same all-MiniLM-L6-v2 model, run with ONNX Runtime (via fastembed).

    No PyTorch is needed, so it installs fast and uses far less memory. That is
    what makes the free Streamlit Cloud deployment possible. The vectors are
    normalised here so cosine similarity stays a plain dot product.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from fastembed import TextEmbedding  # imported lazily

        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = np.asarray(list(self.model.embed(texts)), dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).tolist()


def make_embedder(name: str = "sentence-transformers") -> Embedder:
    """Pick the embedder named by EMBEDDER in .env: 'sentence-transformers' or 'onnx'."""
    name = name.lower().strip()
    if name in ("sentence-transformers", "torch"):
        return SentenceTransformerEmbedder()
    if name == "onnx":
        return OnnxEmbedder()
    raise ValueError(f"Unknown EMBEDDER '{name}'. Use 'sentence-transformers' or 'onnx'.")


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

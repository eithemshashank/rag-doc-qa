"""Step 2c: connect chunks, embedder and store into two simple actions."""
from app.embeddings import Embedder
from app.ingest import Chunk
from app.vectorstore import Hit


def index_chunks(store, embedder: Embedder, chunks: list[Chunk], batch_size: int = 64) -> int:
    """Embed the chunks (in batches) and save them in the store."""
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        store.add(batch, embedder.embed([c.text for c in batch]))
    return len(chunks)


def retrieve(store, embedder: Embedder, question: str, k: int = 4) -> list[Hit]:
    """Return the k chunks whose meaning is closest to the question."""
    return store.search(embedder.embed([question])[0], k=k)

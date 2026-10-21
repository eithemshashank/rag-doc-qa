"""Try the whole retrieval pipeline on one of your own PDFs.

    python scripts/try_search.py yourfile.pdf "your question here"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config
from app.embeddings import make_embedder
from app.ingest import ingest_pdf
from app.retrieval import index_chunks, retrieve
from app.vectorstore import ChromaStore

if len(sys.argv) != 3:
    sys.exit('Usage: python scripts/try_search.py file.pdf "question"')

pdf, question = sys.argv[1], sys.argv[2]
embedder = make_embedder(config.EMBEDDER)
store = ChromaStore(path="chroma_db")

chunks = ingest_pdf(pdf)
print(f"Read {len(chunks)} chunks from {pdf}")
index_chunks(store, embedder, chunks)

for rank, hit in enumerate(retrieve(store, embedder, question, k=3), start=1):
    print(f"\n#{rank}  score {hit.score:.2f}  {hit.source}, page {hit.page}")
    print(hit.text[:300])

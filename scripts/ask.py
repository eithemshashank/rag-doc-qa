"""Ask a question about a PDF, end to end (needs an API key in .env).

    python scripts/ask.py yourfile.pdf "your question here"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config
from app.embeddings import make_embedder
from app.ingest import ingest_pdf
from app.llm import make_llm
from app.qa import answer_question
from app.retrieval import index_chunks
from app.vectorstore import ChromaStore

if len(sys.argv) != 3:
    sys.exit('Usage: python scripts/ask.py file.pdf "question"')

pdf, question = sys.argv[1], sys.argv[2]
embedder = make_embedder(config.EMBEDDER)
store = ChromaStore(path="chroma_db")
index_chunks(store, embedder, ingest_pdf(pdf))

llm = make_llm(config.LLM_PROVIDER, config)
result = answer_question(
    question, store, embedder, llm, k=config.TOP_K, min_score=config.MIN_SCORE
)

print("\nANSWER\n" + result.text)
print("\nSOURCES")
for s in result.sources:
    print(f"- {s.source}, page {s.page} (similarity {s.score})")

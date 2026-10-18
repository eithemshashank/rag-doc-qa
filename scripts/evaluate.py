"""Run the evaluation (needs an API key in .env).

    python scripts/evaluate.py

Reads eval/docs/*.pdf and eval/questions.json, asks every question, prints a
summary and writes eval/results.md. Uses an in-memory index, so it does not
touch your chroma_db or uploads.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config
from app.embeddings import SentenceTransformerEmbedder
from app.evaluation import load_questions, run_eval, summarize, to_markdown
from app.ingest import ingest_pdf
from app.llm import make_llm
from app.retrieval import index_chunks
from app.vectorstore import InMemoryStore

embedder = SentenceTransformerEmbedder()
store = InMemoryStore()
for pdf in sorted((ROOT / "eval" / "docs").glob("*.pdf")):
    index_chunks(store, embedder, ingest_pdf(pdf))
    print("indexed", pdf.name)

llm = make_llm(config.LLM_PROVIDER, config)
questions = load_questions(ROOT / "eval" / "questions.json")
rows = run_eval(questions, store, embedder, llm, k=config.TOP_K,
                min_score=config.MIN_SCORE, pause=2.0)
summary = summarize(rows)

for i, r in enumerate(rows, 1):
    print(f"{i:>2}. {'PASS' if r['correct'] else 'FAIL'}  {r['question']}")
print()
print(summary)

model = getattr(config, f"{config.LLM_PROVIDER.upper()}_MODEL", config.LLM_PROVIDER)
(ROOT / "eval" / "results.md").write_text(
    to_markdown(rows, summary, f"{config.LLM_PROVIDER}:{model}"), encoding="utf-8")
print("\nWrote eval/results.md")

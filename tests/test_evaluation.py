from pathlib import Path

from app.embeddings import HashingEmbedder
from app.evaluation import contains_all, is_refusal, load_questions, run_eval, summarize
from app.ingest import ingest_pdf
from app.llm import FakeLLM
from app.qa import NOT_FOUND
from app.retrieval import index_chunks
from app.vectorstore import InMemoryStore

EVAL = Path(__file__).resolve().parents[1] / "eval"


def test_contains_all_accepts_any_spelling_per_group():
    expect = [["Tuesday", "Tue"], ["Thursday", "Thu"]]
    assert contains_all("Office days: Tue and Thu", expect)
    assert not contains_all("Office days: Tue only", expect)


def test_is_refusal():
    assert is_refusal(NOT_FOUND)
    assert not is_refusal("It is 75%.")


def test_questions_point_at_real_pages():
    """Every expected fact really is on the page the question says (checks the data)."""
    from pypdf import PdfReader

    for item in load_questions(EVAL / "questions.json"):
        if item.get("answerable", True):
            page = PdfReader(EVAL / "docs" / item["file"]).pages[item["page"] - 1]
            assert contains_all(page.extract_text(), item["expect"]), item["q"]


def test_run_eval_scores_a_refusing_llm():
    embedder, store = HashingEmbedder(), InMemoryStore()
    for pdf in (EVAL / "docs").glob("*.pdf"):
        index_chunks(store, embedder, ingest_pdf(pdf))
    rows = run_eval(load_questions(EVAL / "questions.json"), store, embedder,
                    FakeLLM(NOT_FOUND), min_score=0.0)
    s = summarize(rows)
    assert s["answerable"] == 12 and s["unanswerable"] == 3
    assert s["refusals_correct"] == 3      # refusing is right for these
    assert s["answers_correct"] == 0       # refusing is wrong for the rest

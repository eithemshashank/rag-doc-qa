import pytest

from app.embeddings import HashingEmbedder
from app.ingest import Chunk
from app.llm import FakeLLM
from app.qa import NOT_FOUND, answer_question, build_prompt
from app.retrieval import index_chunks
from app.vectorstore import Hit, InMemoryStore

DOCS = [
    ("policy.pdf", 3, "Refunds are issued within 14 days of the return request."),
    ("policy.pdf", 4, "Shipping to Hyderabad takes three business days."),
    ("guide.pdf", 1, "The warranty covers manufacturing defects for two years."),
]


def make_store():
    store = InMemoryStore()
    chunks = [Chunk(t, s, p, f"{s}-p{p}-c1") for s, p, t in DOCS]
    index_chunks(store, HashingEmbedder(), chunks)
    return store


def test_prompt_contains_question_and_labelled_context():
    hits = [Hit("Refunds take 14 days.", "policy.pdf", 3, 0.9)]
    prompt = build_prompt("How long do refunds take?", hits)
    assert "[policy.pdf, p.3]" in prompt
    assert "Refunds take 14 days." in prompt
    assert "How long do refunds take?" in prompt
    assert "ONLY the context" in prompt


def test_answer_returns_llm_text_and_sources():
    llm = FakeLLM("Refunds take 14 days [policy.pdf, p.3].")
    result = answer_question(
        "How many days for refunds to be issued", make_store(), HashingEmbedder(), llm,
        k=2, min_score=0.1,
    )
    assert result.text.startswith("Refunds take 14 days")
    assert result.sources[0].source == "policy.pdf" and result.sources[0].page == 3
    assert len(llm.prompts) == 1
    assert "Refunds are issued within 14 days" in llm.prompts[0]


def test_irrelevant_question_skips_the_llm():
    llm = FakeLLM("should not be used")
    result = answer_question(
        "quantum chromodynamics lagrangian", make_store(), HashingEmbedder(), llm,
        min_score=0.3,
    )
    assert result.text == NOT_FOUND and result.sources == []
    assert llm.prompts == []  # no LLM call was made


def test_model_saying_not_found_gives_no_sources():
    llm = FakeLLM(NOT_FOUND)
    result = answer_question(
        "refunds days", make_store(), HashingEmbedder(), llm, min_score=0.1
    )
    assert result.text == NOT_FOUND and result.sources == []


def test_empty_store_gives_not_found():
    llm = FakeLLM()
    result = answer_question("anything", InMemoryStore(), HashingEmbedder(), llm)
    assert result.text == NOT_FOUND and llm.prompts == []


def test_empty_question_raises():
    with pytest.raises(ValueError):
        answer_question("   ", make_store(), HashingEmbedder(), FakeLLM())

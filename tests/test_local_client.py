"""The single-process client used by the public demo."""
import pytest

from app.embeddings import HashingEmbedder, make_embedder
from app.factory import build_session_service
from app.llm import FakeLLM
from tests.test_service import make_pdf_bytes
from ui.api_client import ApiError
from ui.local_client import LocalClient


def make_client(max_questions=0):
    service = build_session_service(HashingEmbedder(), FakeLLM("Refunds take 14 days."))
    service.min_score = 0.1
    return LocalClient(service, max_questions=max_questions)


def test_upload_list_ask_delete_like_the_api_client(tmp_path):
    client = make_client()
    data = make_pdf_bytes(tmp_path, ["Refunds take 14 days."])
    assert client.upload("policy.pdf", data)["filename"] == "policy.pdf"
    assert client.list_documents() == [{"filename": "policy.pdf", "chunks": 1}]
    assert client.health() == {"status": "ok", "documents": 1, "chunks": 1}

    result = client.ask("How many days do refunds take")
    assert "14 days" in result["answer"]
    assert result["sources"][0]["source"] == "policy.pdf"

    client.delete("policy.pdf")
    assert client.list_documents() == []
    with pytest.raises(ApiError, match="not found"):
        client.delete("policy.pdf")


def test_bad_upload_becomes_api_error():
    with pytest.raises(ApiError, match="Only PDF"):
        make_client().upload("notes.txt", b"hello")


def test_visitors_do_not_share_documents(tmp_path):
    a, b = make_client(), make_client()
    a.upload("a.pdf", make_pdf_bytes(tmp_path, ["Secret plan."], "a.pdf"))
    assert a.list_documents() != []
    assert b.list_documents() == []


def test_question_limit_is_enforced():
    client = make_client(max_questions=2)
    client.ask("one")
    client.ask("two")
    assert client.questions_left == 0
    with pytest.raises(ApiError, match="Demo limit"):
        client.ask("three")


def test_llm_failure_is_hidden_from_the_visitor(tmp_path):
    class Broken:
        def generate(self, prompt):
            raise RuntimeError("gsk_secret-key leaked")

    service = build_session_service(HashingEmbedder(), Broken())
    service.min_score = 0.1
    client = LocalClient(service)
    client.upload("p.pdf", make_pdf_bytes(tmp_path, ["Refunds take 14 days."], "p.pdf"))
    with pytest.raises(ApiError) as err:
        client.ask("refunds days")
    assert "secret" not in str(err.value)


def test_make_embedder_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown EMBEDDER"):
        make_embedder("word2vec")

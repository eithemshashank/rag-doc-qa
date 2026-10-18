"""API tests. They need fastapi and httpx (both in requirements.txt)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from app.embeddings import HashingEmbedder
from app.llm import FakeLLM
from app.main import create_app
from app.service import DocumentService
from app.vectorstore import InMemoryStore
from tests.test_service import make_pdf_bytes


class BrokenLLM:
    def generate(self, prompt):
        raise RuntimeError("secret-key-123 leaked in an error")


def make_client(tmp_path, llm=None):
    service = DocumentService(
        InMemoryStore(), HashingEmbedder(), llm or FakeLLM("Refunds take 14 days."),
        tmp_path / "uploads", min_score=0.1,
    )
    return TestClient(create_app(service))


def upload(client, tmp_path, name="policy.pdf", pages=("Refunds take 14 days.",)):
    data = make_pdf_bytes(tmp_path, list(pages), name)
    return client.post("/documents", files={"file": (name, data, "application/pdf")})


def test_health(tmp_path):
    body = make_client(tmp_path).get("/health").json()
    assert body == {"status": "ok", "documents": 0, "chunks": 0}


def test_full_flow_upload_list_ask_delete(tmp_path):
    client = make_client(tmp_path)

    res = upload(client, tmp_path)
    assert res.status_code == 201
    assert res.json()["filename"] == "policy.pdf"

    assert client.get("/documents").json() == [{"filename": "policy.pdf", "chunks": 1}]

    res = client.post("/ask", json={"question": "How many days do refunds take"})
    assert res.status_code == 200
    body = res.json()
    assert "14 days" in body["answer"]
    assert body["sources"][0]["source"] == "policy.pdf"

    assert client.delete("/documents/policy.pdf").status_code == 204
    assert client.get("/documents").json() == []
    assert client.delete("/documents/policy.pdf").status_code == 404


def test_non_pdf_upload_is_400(tmp_path):
    client = make_client(tmp_path)
    res = client.post("/documents", files={"file": ("a.txt", b"hello", "text/plain")})
    assert res.status_code == 400
    assert "Only PDF" in res.json()["detail"]


def test_fake_pdf_is_400(tmp_path):
    client = make_client(tmp_path)
    res = client.post("/documents", files={"file": ("a.pdf", b"not a pdf", "application/pdf")})
    assert res.status_code == 400


def test_empty_question_is_422(tmp_path):
    client = make_client(tmp_path)
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "k": 99}).status_code == 422


def test_llm_failure_is_502_and_hides_details(tmp_path):
    client = make_client(tmp_path, BrokenLLM())
    upload(client, tmp_path)
    res = client.post("/ask", json={"question": "How many days do refunds take"})
    assert res.status_code == 502
    assert "secret-key-123" not in res.text

import pytest
from reportlab.pdfgen import canvas

from app.embeddings import HashingEmbedder
from app.llm import FakeLLM
from app.qa import NOT_FOUND
from app.service import DocumentError, DocumentService
from app.vectorstore import InMemoryStore


def make_pdf_bytes(tmp_path, pages, name="doc.pdf"):
    path = tmp_path / ("src_" + name)
    c = canvas.Canvas(str(path))
    for text in pages:
        if text:
            c.drawString(72, 750, text)
        c.showPage()
    c.save()
    return path.read_bytes()


def make_service(tmp_path, llm=None, **kw):
    return DocumentService(
        InMemoryStore(), HashingEmbedder(), llm or FakeLLM("An answer."),
        tmp_path / "uploads", min_score=0.1, **kw,
    )


def test_upload_indexes_and_lists_the_document(tmp_path):
    svc = make_service(tmp_path)
    data = make_pdf_bytes(tmp_path, ["Refunds take 14 days.", "Warranty lasts two years."])
    info = svc.add_pdf("policy.pdf", data)
    assert info == {"filename": "policy.pdf", "pages": 2, "chunks": 2}
    assert svc.list_documents() == [{"filename": "policy.pdf", "chunks": 2}]
    assert (tmp_path / "uploads" / "policy.pdf").exists()
    assert svc.stats() == {"documents": 1, "chunks": 2}


def test_reupload_replaces_old_chunks(tmp_path):
    svc = make_service(tmp_path)
    svc.add_pdf("a.pdf", make_pdf_bytes(tmp_path, ["one", "two", "three"]))
    svc.add_pdf("a.pdf", make_pdf_bytes(tmp_path, ["only one page"]))
    assert svc.list_documents() == [{"filename": "a.pdf", "chunks": 1}]


def test_ask_uses_the_uploaded_document(tmp_path):
    llm = FakeLLM("Refunds take 14 days [policy.pdf, p.1].")
    svc = make_service(tmp_path, llm)
    svc.add_pdf("policy.pdf", make_pdf_bytes(tmp_path, ["Refunds take 14 days."]))
    answer = svc.ask("How many days do refunds take")
    assert "14 days" in answer.text
    assert answer.sources[0].source == "policy.pdf"


def test_ask_with_no_documents_says_not_found(tmp_path):
    answer = make_service(tmp_path).ask("anything at all")
    assert answer.text == NOT_FOUND


def test_delete_removes_chunks_and_file(tmp_path):
    svc = make_service(tmp_path)
    svc.add_pdf("a.pdf", make_pdf_bytes(tmp_path, ["hello world"]))
    assert svc.delete_document("a.pdf") is True
    assert svc.list_documents() == []
    assert not (tmp_path / "uploads" / "a.pdf").exists()
    assert svc.delete_document("a.pdf") is False


@pytest.mark.parametrize("filename, data, fragment", [
    ("notes.txt", b"%PDF-1.4 x", "Only PDF"),
    (None, b"%PDF-1.4 x", "Only PDF"),
    ("a.pdf", b"", "empty"),
    ("a.pdf", b"this is not a pdf", "not a valid PDF"),
])
def test_bad_uploads_are_rejected(tmp_path, filename, data, fragment):
    svc = make_service(tmp_path)
    with pytest.raises(DocumentError) as err:
        svc.add_pdf(filename, data)
    assert fragment in str(err.value)
    assert svc.list_documents() == []


def test_corrupt_pdf_is_rejected_cleanly(tmp_path):
    svc = make_service(tmp_path)
    with pytest.raises(DocumentError):
        svc.add_pdf("a.pdf", b"%PDF-1.4 this is garbage")
    assert svc.list_documents() == []
    assert not (tmp_path / "uploads" / "a.pdf").exists()


def test_pdf_without_text_is_rejected(tmp_path):
    svc = make_service(tmp_path)
    with pytest.raises(DocumentError) as err:
        svc.add_pdf("blank.pdf", make_pdf_bytes(tmp_path, [""]))
    assert "No text found" in str(err.value)


def test_oversized_file_gets_413(tmp_path):
    svc = make_service(tmp_path, max_bytes=10)
    with pytest.raises(DocumentError) as err:
        svc.add_pdf("a.pdf", b"%PDF-1.4 more than ten bytes")
    assert err.value.status == 413


def test_folder_parts_in_filename_are_dropped(tmp_path):
    svc = make_service(tmp_path)
    info = svc.add_pdf("../../evil.pdf", make_pdf_bytes(tmp_path, ["text here"]))
    assert info["filename"] == "evil.pdf"
    assert (tmp_path / "uploads" / "evil.pdf").exists()

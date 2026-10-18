import pytest
from reportlab.pdfgen import canvas

from app.ingest import chunk_text, ingest_pdf


def make_pdf(path, pages):
    c = canvas.Canvas(str(path))
    for text in pages:
        c.drawString(72, 750, text)
        c.showPage()
    c.save()


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", chunk_size=100, overlap=10) == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n  ") == []


def test_long_text_is_split_with_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # the end of one chunk is repeated at the start of the next
    tail = chunks[0].split()[-2:]
    assert all(word in chunks[1] for word in tail)
    # nothing is lost: every word appears in some chunk
    joined = " ".join(chunks)
    assert all(f"word{i}" in joined for i in range(300))


def test_invalid_settings_raise():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_ingest_pdf_keeps_page_numbers(tmp_path):
    pdf = tmp_path / "sample.pdf"
    make_pdf(pdf, ["Alpha page one text", "Beta page two text"])
    chunks = ingest_pdf(pdf)
    assert [c.page for c in chunks] == [1, 2]
    assert chunks[0].source == "sample.pdf"
    assert chunks[1].chunk_id == "sample.pdf-p2-c1"
    assert "Beta" in chunks[1].text

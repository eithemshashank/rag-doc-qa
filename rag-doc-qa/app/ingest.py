"""Step 1: read a PDF and split it into overlapping text chunks.

Every chunk remembers which file and page it came from, so later the app can
show the user exactly where an answer was found.
"""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str   # file name, e.g. "report.pdf"
    page: int     # 1-based page number
    chunk_id: str  # unique id, e.g. "report.pdf-p3-c1"


def load_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) for every page that has text."""
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into pieces of about `chunk_size` characters.

    Consecutive pieces share `overlap` characters, so a sentence cut at the end
    of one chunk still appears whole at the start of the next. We prefer to
    cut at a space instead of in the middle of a word.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = " ".join(text.split())  # collapse newlines and repeated spaces
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + chunk_size // 2, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # always move forward
    return chunks


def ingest_pdf(path: str | Path, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Read a PDF and return all its chunks with page metadata."""
    source = Path(path).name
    chunks = []
    for page_number, page_text in load_pdf(path):
        for i, piece in enumerate(chunk_text(page_text, chunk_size, overlap), start=1):
            chunks.append(Chunk(
                text=piece,
                source=source,
                page=page_number,
                chunk_id=f"{source}-p{page_number}-c{i}",
            ))
    return chunks

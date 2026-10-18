"""Step 4a: the application logic behind the API.

The FastAPI file only turns HTTP requests into calls to this class, so all the
real behaviour (validation, saving, indexing, deleting, asking) lives here and
can be tested without starting a web server.
"""
import shutil
import tempfile
from pathlib import Path

from app.embeddings import Embedder
from app.ingest import ingest_pdf
from app.llm import LLM
from app.qa import Answer, answer_question
from app.retrieval import index_chunks


class DocumentError(ValueError):
    """A problem with an uploaded file. `status` is the matching HTTP code."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class DocumentService:
    def __init__(
        self,
        store,
        embedder: Embedder,
        llm: LLM,
        upload_dir: str | Path,
        k: int = 4,
        min_score: float = 0.25,
        max_bytes: int = 20 * 1024 * 1024,
    ):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.upload_dir = Path(upload_dir)
        self.k = k
        self.min_score = min_score
        self.max_bytes = max_bytes

    def add_pdf(self, filename: str | None, data: bytes) -> dict:
        """Validate a PDF, index it, and keep a copy. Uploading the same file
        name again replaces the old version."""
        name = Path(filename or "").name  # drops any folder part, e.g. "../x.pdf"
        if not name.lower().endswith(".pdf"):
            raise DocumentError("Only PDF files are accepted.")
        if not data:
            raise DocumentError("The file is empty.")
        if len(data) > self.max_bytes:
            raise DocumentError(
                f"File is larger than {self.max_bytes // (1024 * 1024)} MB.", status=413
            )
        if not data.startswith(b"%PDF"):
            raise DocumentError("This file is not a valid PDF.")

        # Parse in a temporary folder first, so a bad upload never damages
        # the copy and index we already have.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / name
            tmp_path.write_bytes(data)
            try:
                chunks = ingest_pdf(tmp_path)
            except Exception as exc:
                raise DocumentError("Could not read this PDF.") from exc
            if not chunks:
                raise DocumentError(
                    "No text found in this PDF. Scanned PDFs need OCR, "
                    "which this app does not do."
                )
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp_path, self.upload_dir / name)

        self.store.delete_source(name)  # replace any older version
        index_chunks(self.store, self.embedder, chunks)
        return {
            "filename": name,
            "pages": len({c.page for c in chunks}),
            "chunks": len(chunks),
        }

    def list_documents(self) -> list[dict]:
        return [
            {"filename": name, "chunks": count}
            for name, count in sorted(self.store.list_sources().items())
        ]

    def delete_document(self, name: str) -> bool:
        """Remove a document. Returns False if we don't have it."""
        if name not in self.store.list_sources():
            return False
        self.store.delete_source(name)
        (self.upload_dir / Path(name).name).unlink(missing_ok=True)
        return True

    def ask(self, question: str, k: int | None = None) -> Answer:
        return answer_question(
            question, self.store, self.embedder, self.llm,
            k=k or self.k, min_score=self.min_score,
        )

    def stats(self) -> dict:
        sources = self.store.list_sources()
        return {"documents": len(sources), "chunks": sum(sources.values())}

"""Same interface as ApiClient, but talks to a DocumentService in the same
process instead of over HTTP.

Used by the public demo on Streamlit Cloud, which can run only one process.
It also caps questions per visitor so a public demo cannot burn the API key.
"""
import logging
from dataclasses import asdict

from app.service import DocumentError, DocumentService
from ui.api_client import ApiError

log = logging.getLogger("rag")


class LocalClient:
    def __init__(self, service: DocumentService, max_questions: int = 0):
        self._service = service
        self._max_questions = max_questions  # 0 means no limit
        self._asked = 0

    @property
    def questions_left(self) -> int | None:
        return None if not self._max_questions else max(self._max_questions - self._asked, 0)

    def health(self) -> dict:
        return {"status": "ok", **self._service.stats()}

    def list_documents(self) -> list[dict]:
        return self._service.list_documents()

    def upload(self, filename: str, data: bytes) -> dict:
        try:
            return self._service.add_pdf(filename, data)
        except DocumentError as err:
            raise ApiError(str(err)) from None

    def delete(self, filename: str) -> None:
        if not self._service.delete_document(filename):
            raise ApiError("Document not found.")

    def ask(self, question: str) -> dict:
        if self._max_questions and self._asked >= self._max_questions:
            raise ApiError(
                f"Demo limit reached ({self._max_questions} questions per visit). "
                "Reload the page to start again, or run the project locally."
            )
        self._asked += 1
        try:
            answer = self._service.ask(question)
        except ValueError as err:
            raise ApiError(str(err)) from None
        except Exception:
            log.exception("language model request failed")
            raise ApiError("The language model request failed.") from None
        return {"answer": answer.text, "sources": [asdict(s) for s in answer.sources]}

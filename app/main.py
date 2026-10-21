"""Step 4b: the web API.

Run it with:
    python -m uvicorn app.main:app
then open http://127.0.0.1:8000/docs to try every endpoint in the browser.
"""
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.service import DocumentError, DocumentService

log = logging.getLogger("rag")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int | None = Field(default=None, ge=1, le=10)


def build_default_service() -> DocumentService:
    """The real setup: embeddings, vector store and LLM chosen by .env."""
    from app.factory import build_default_service as build

    return build()


def create_app(service: DocumentService | None = None) -> FastAPI:
    """Build the app. Tests pass in a service with fakes; in real use we leave
    it empty and the real one is created when the server starts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            app.state.service = build_default_service()
        yield

    app = FastAPI(title="RAG Document Q&A", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    def svc(request: Request) -> DocumentService:
        return request.app.state.service

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", **svc(request).stats()}

    @app.post("/documents", status_code=201)
    def upload_document(request: Request, file: UploadFile = File(...)):
        service = svc(request)
        # read one byte more than the limit so we can tell "too big" from "just right"
        data = file.file.read(service.max_bytes + 1)
        try:
            return service.add_pdf(file.filename, data)
        except DocumentError as err:
            raise HTTPException(status_code=err.status, detail=str(err))

    @app.get("/documents")
    def list_documents(request: Request):
        return svc(request).list_documents()

    @app.delete("/documents/{filename}", status_code=204)
    def delete_document(filename: str, request: Request):
        if not svc(request).delete_document(filename):
            raise HTTPException(status_code=404, detail="Document not found.")

    @app.post("/ask")
    def ask(body: AskRequest, request: Request):
        try:
            answer = svc(request).ask(body.question, body.k)
        except ValueError as err:  # e.g. question is only spaces
            raise HTTPException(status_code=422, detail=str(err))
        except Exception:
            # Log the details for us, but never send raw errors (which could
            # include keys or internals) back to the caller.
            log.exception("language model request failed")
            raise HTTPException(
                status_code=502, detail="The language model request failed."
            )
        return {"answer": answer.text, "sources": [asdict(s) for s in answer.sources]}

    return app


app = create_app()

"""Step 3b: the full question-answering step.

question -> retrieve the best chunks -> build a prompt -> LLM -> answer + sources
"""
from dataclasses import dataclass, field

from app.embeddings import Embedder
from app.llm import LLM
from app.retrieval import retrieve
from app.vectorstore import Hit

NOT_FOUND = "I couldn't find this in the uploaded documents."

PROMPT_TEMPLATE = """You answer questions using ONLY the context below, which comes from the user's documents.

Rules:
- Use only facts stated in the context. Do not use outside knowledge.
- If the context does not contain the answer, reply exactly: {not_found}
- After each fact, cite its source like [file.pdf, p.3].
- Be concise.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class Source:
    source: str
    page: int
    score: float
    snippet: str


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)


def build_prompt(question: str, hits: list[Hit]) -> str:
    """Put the retrieved chunks, each labelled with file and page, into the prompt."""
    context = "\n\n".join(
        f"[{h.source}, p.{h.page}]\n{h.text}" for h in hits
    )
    return PROMPT_TEMPLATE.format(
        not_found=NOT_FOUND, context=context, question=question.strip()
    )


def answer_question(
    question: str,
    store,
    embedder: Embedder,
    llm: LLM,
    k: int = 4,
    min_score: float = 0.25,
) -> Answer:
    """Answer a question from the stored documents, with sources.

    If nothing in the documents is similar enough to the question, we skip the
    LLM entirely. That saves a call and stops it from inventing an answer.
    """
    if not question.strip():
        raise ValueError("question is empty")

    hits = [h for h in retrieve(store, embedder, question, k=k) if h.score >= min_score]
    if not hits:
        return Answer(NOT_FOUND, [])

    reply = llm.generate(build_prompt(question, hits))
    if reply.strip() == NOT_FOUND:
        return Answer(NOT_FOUND, [])  # the model said the context didn't help

    sources = [Source(h.source, h.page, round(h.score, 3), h.text[:200]) for h in hits]
    return Answer(reply, sources)

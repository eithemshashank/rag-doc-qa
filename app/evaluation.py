"""Step 7: measure how well the app answers.

Three numbers come out of a run:
- retrieval hit: did the right page come back among the retrieved chunks?
  (this tests the search, with no LLM involved)
- answer correct: does the answer contain the expected facts?
- refusal correct: for questions NOT in the documents, did the app say
  "I couldn't find this" instead of making something up?
"""
import json
import time
from pathlib import Path

from app.qa import NOT_FOUND, answer_question


def contains_all(answer: str, expect: list[list[str]]) -> bool:
    """Every group needs at least one of its spellings to appear in the answer."""
    low = answer.lower()
    return all(any(alt.lower() in low for alt in group) for group in expect)


def is_refusal(answer: str) -> bool:
    return NOT_FOUND.lower() in answer.lower()


def run_eval(questions, store, embedder, llm, k=4, min_score=0.25, pause=0.0):
    """Ask every question and return one result row per question."""
    rows = []
    for item in questions:
        ans = answer_question(item["q"], store, embedder, llm, k=k, min_score=min_score)
        row = {
            "question": item["q"],
            "answerable": item.get("answerable", True),
            "answer": ans.text.strip(),
            "sources": [(s.source, s.page) for s in ans.sources],
        }
        if row["answerable"]:
            row["retrieval_hit"] = (item["file"], item["page"]) in row["sources"]
            row["correct"] = contains_all(ans.text, item["expect"]) and not is_refusal(ans.text)
        else:
            row["correct"] = is_refusal(ans.text)
        rows.append(row)
        if pause:
            time.sleep(pause)  # stay under free-tier rate limits
    return rows


def summarize(rows) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    return {
        "answerable": len(ans),
        "retrieval_hits": sum(r["retrieval_hit"] for r in ans),
        "answers_correct": sum(r["correct"] for r in ans),
        "unanswerable": len(unans),
        "refusals_correct": sum(r["correct"] for r in unans),
    }


def to_markdown(rows, summary, llm_name="") -> str:
    s = summary
    lines = [
        "# Evaluation results",
        "",
        f"LLM: `{llm_name}`. Two fictional PDFs, 4 pages each (see `eval/make_docs.py`).",
        "",
        "| Measure | Result |",
        "|---|---|",
        f"| Right page retrieved | {s['retrieval_hits']} / {s['answerable']} |",
        f"| Answer contains the correct facts | {s['answers_correct']} / {s['answerable']} |",
        f"| Refused questions not in the documents | {s['refusals_correct']} / {s['unanswerable']} |",
        "",
        "| # | Question | In docs? | Page found | Pass |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        found = "yes" if r.get("retrieval_hit") else ("-" if not r["answerable"] else "no")
        lines.append(
            f"| {i} | {r['question']} | {'yes' if r['answerable'] else 'no'} "
            f"| {found} | {'PASS' if r['correct'] else 'FAIL'} |"
        )
    failed = [r for r in rows if not r["correct"]]
    if failed:
        lines += ["", "## Failures", ""]
        for r in failed:
            lines.append(f"- **{r['question']}** -> {r['answer'][:200]!r}")
    return "\n".join(lines) + "\n"


def load_questions(path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

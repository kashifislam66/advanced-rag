import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from app.evaluation.dataset import EvalItem
from app.generation.prompts import NO_ANSWER_TEXT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemOutput:
    """What any RAG variant must return: the chunks it used (best first) and its answer."""

    contexts: list[Document]
    answer_text: str
    final_status: str = "answered"
    rank_valid: bool = True
    llm_calls: int = 0


SystemFn = Callable[[str], SystemOutput]


@dataclass(frozen=True)
class EvalRecord:
    """One question's full result. Saved to disk so metrics never need a re-run."""

    id: str
    question: str
    category: str
    split: str
    answerable: bool
    reference: str
    reference_sources: list[str]
    retrieved_sources: list[str]
    contexts: list[str]
    answer_text: str
    refused: bool
    final_status: str
    latency_ms: float
    error: str | None = None
    rank_valid: bool = True
    llm_calls: int = 0


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def run_system(items: list[EvalItem], system_fn: SystemFn) -> list[EvalRecord]:
    """Run one RAG variant over every question. One failure never stops the run."""
    records: list[EvalRecord] = []
    for position, item in enumerate(items, start=1):
        start = time.perf_counter()
        try:
            output = system_fn(item.question)
            error = None
        except Exception as exc:
            logger.exception("Question %s failed", item.id)
            output = SystemOutput(contexts=[], answer_text="", final_status="error")
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - start) * 1000

        sources = [str(doc.metadata.get("source", "unknown")) for doc in output.contexts]
        records.append(
            EvalRecord(
                id=item.id,
                question=item.question,
                category=item.category,
                split=item.split,
                answerable=item.answerable,
                reference=item.reference,
                reference_sources=list(item.reference_sources),
                retrieved_sources=_unique_in_order(sources),
                contexts=[doc.page_content for doc in output.contexts],
                answer_text=output.answer_text,
                refused=NO_ANSWER_TEXT.lower() in output.answer_text.lower(),
                final_status=output.final_status,
                latency_ms=round(latency_ms, 1),
                error=error,
                rank_valid=output.rank_valid,
                llm_calls=output.llm_calls,
            )
        )
        logger.info("[%d/%d] %s done in %.0f ms", position, len(items), item.id, latency_ms)
    return records


def save_records(records: list[EvalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), path)


def load_records(path: Path) -> list[EvalRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Run file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return [EvalRecord(**json.loads(line)) for line in handle if line.strip()]
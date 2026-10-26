import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.evaluation.runner import EvalRecord

logger = logging.getLogger(__name__)


def eligible_for_ragas(records: list[EvalRecord]) -> list[EvalRecord]:
    """Answerable questions that got a real answer. Refusals and crashes are counted elsewhere."""
    return [r for r in records if r.answerable and not r.refused and r.error is None and r.contexts]


def run_ragas(
    records: list[EvalRecord],
    judge_llm: BaseChatModel,
    embeddings: Embeddings,
    workers: int,
    timeout: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Score records with RAGAS. Returns (one row per record, metric column names)."""
    # Imported here so the rest of the project (and its tests) works without ragas installed.
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    if not records:
        return [], []

    llm = LangchainLLMWrapper(judge_llm)
    emb = LangchainEmbeddingsWrapper(embeddings)
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=emb),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    names = [metric.name for metric in metrics]

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=r.question,
                retrieved_contexts=r.contexts,
                response=r.answer_text,
                reference=r.reference,
            )
            for r in records
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(max_workers=workers, timeout=timeout),
        raise_exceptions=False,
        show_progress=True,
    )
    frame = result.to_pandas()

    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise RuntimeError(f"RAGAS columns {missing} not found; available: {list(frame.columns)}")
    if len(frame) != len(records):
        raise RuntimeError(f"Expected {len(records)} RAGAS rows, got {len(frame)}")

    rows: list[dict[str, Any]] = []
    for record, (_, row) in zip(records, frame.iterrows()):
        if "user_input" in frame.columns and row["user_input"] != record.question:
            raise RuntimeError(f"RAGAS row order mismatch at question {record.id}")
        scores: dict[str, float | None] = {}
        for name in names:
            value = row[name]
            scores[name] = None if value is None or value != value else float(value)
        rows.append({"id": record.id, "category": record.category, **scores})

    logger.info("RAGAS scored %d records with metrics %s", len(rows), names)
    return rows, names
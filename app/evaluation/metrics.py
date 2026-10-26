from collections import defaultdict
from typing import Any

from app.evaluation.runner import EvalRecord


def first_hit_rank(retrieved_sources: list[str], reference_sources: list[str]) -> int | None:
    """1-based position of the first correct source, or None if none was retrieved."""
    wanted = set(reference_sources)
    for position, source in enumerate(retrieved_sources, start=1):
        if source in wanted:
            return position
    return None


def full_hit(retrieved_sources: list[str], reference_sources: list[str]) -> bool:
    """True if EVERY correct source was retrieved (matters for multi_doc questions)."""
    return set(reference_sources) <= set(retrieved_sources)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_records(records: list[EvalRecord]) -> dict[str, float | int | None]:
    """Cheap, deterministic retrieval and refusal metrics."""
    ok = [r for r in records if r.error is None]
    answerable = [r for r in ok if r.answerable]
    unanswerable = [r for r in ok if not r.answerable]
    ranks = [first_hit_rank(r.retrieved_sources, r.reference_sources) for r in answerable]

    return {
        "questions": len(records),
        "errors": len(records) - len(ok),
        "answerable": len(answerable),
        "hit_rate": _mean([1.0 if rank is not None else 0.0 for rank in ranks]),
        "full_hit_rate": _mean(
            [1.0 if full_hit(r.retrieved_sources, r.reference_sources) else 0.0 for r in answerable]
        ),
        "false_refusal_rate": _mean([1.0 if r.refused else 0.0 for r in answerable]),
        "unanswerable": len(unanswerable),
        "correct_refusal_rate": _mean([1.0 if r.refused else 0.0 for r in unanswerable]),
        "median_latency_ms": _median([r.latency_ms for r in ok]),
        "mean_llm_calls": _mean([float(r.llm_calls) for r in ok]),
        "mrr": _mean(
            [1.0 / rank if rank else 0.0 for r, rank in zip(answerable, ranks) if r.rank_valid]
        ),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def summarize_by_category(records: list[EvalRecord]) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[EvalRecord]] = defaultdict(list)
    for record in records:
        groups[record.category].append(record)
    return {category: summarize_records(group) for category, group in sorted(groups.items())}


def average_scores(rows: list[dict[str, Any]], names: list[str]) -> dict[str, dict[str, Any]]:
    """Mean per RAGAS metric, plus how many rows scored and how many failed (NaN)."""
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        values = [row[name] for row in rows if row.get(name) is not None]
        result[name] = {
            "mean": _mean(values),
            "scored": len(values),
            "failed": len(rows) - len(values),
        }
    return result
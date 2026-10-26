from app.evaluation.metrics import first_hit_rank
from app.evaluation.runner import EvalRecord

# True means a higher value is better.
DIRECTION = {
    "hit_rate": True,
    "mrr": True,
    "full_hit_rate": True,
    "false_refusal_rate": False,
    "correct_refusal_rate": True,
}


def question_changes(old: list[EvalRecord], new: list[EvalRecord]) -> list[tuple[str, str]]:
    """Per-question differences between two runs of the same questions."""
    new_by_id = {record.id: record for record in new}
    changes: list[tuple[str, str]] = []

    for before in old:
        after = new_by_id.get(before.id)
        if after is None or before.error or after.error:
            continue

        if before.answerable:
            was_hit = first_hit_rank(before.retrieved_sources, before.reference_sources) is not None
            is_hit = first_hit_rank(after.retrieved_sources, after.reference_sources) is not None
            if was_hit and not is_hit:
                changes.append((before.id, "WORSE: lost the correct source"))
            elif not was_hit and is_hit:
                changes.append((before.id, "BETTER: found the correct source"))
            if not before.refused and after.refused:
                changes.append((before.id, "WORSE: now refuses an answerable question"))
            elif before.refused and not after.refused:
                changes.append((before.id, "BETTER: stopped refusing an answerable question"))
        else:
            if before.refused and not after.refused:
                changes.append((before.id, "WORSE: now answers a question it should refuse"))
            elif not before.refused and after.refused:
                changes.append((before.id, "BETTER: now correctly refuses"))
    return changes


def regression_check(
    old_summary: dict[str, float | int | None],
    new_summary: dict[str, float | int | None],
    tolerance: float,
) -> list[str]:
    """Metrics that got worse by more than `tolerance`."""
    if tolerance < 0:
        raise ValueError("tolerance must be 0 or more")

    problems: list[str] = []
    for metric, higher_is_better in DIRECTION.items():
        before, after = old_summary.get(metric), new_summary.get(metric)
        if before is None or after is None:
            continue
        drop = (before - after) if higher_is_better else (after - before)
        if drop > tolerance:
            problems.append(f"{metric}: {before:.2f} -> {after:.2f}")
    return problems
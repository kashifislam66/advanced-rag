def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge several best-first ID lists into one list scored by RRF."""
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        seen: set[str] = set()
        for position, item_id in enumerate(ranking, start=1):
            if item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
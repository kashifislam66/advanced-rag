import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.vectorstore.chroma_store import search_children
from app.vectorstore.parent_store import ParentStore

logger = logging.getLogger(__name__)


def parents_from_ranked_children(
    parent_store: ParentStore,
    ranked_children: list[tuple[Document, float]],
    max_parents: int,
) -> list[tuple[Document, float]]:
    """Turn best-first (child, score) pairs into unique (parent, score) pairs.

    Each parent keeps the score of its best (earliest) child.
    """
    if max_parents <= 0:
        raise ValueError("max_parents must be positive")

    ordered_ids: list[str] = []
    best_score: dict[str, float] = {}
    for child, score in ranked_children:
        parent_id = child.metadata.get("parent_id")
        if parent_id is None:
            logger.warning("Child without parent_id skipped: %s", child.metadata)
            continue
        if parent_id in best_score:
            continue
        best_score[parent_id] = score
        ordered_ids.append(parent_id)
        if len(ordered_ids) == max_parents:
            break

    parents_by_id = parent_store.get_many(ordered_ids)

    results: list[tuple[Document, float]] = []
    for parent_id in ordered_ids:
        parent = parents_by_id.get(parent_id)
        if parent is None:
            logger.warning("Parent %s missing from store", parent_id)
            continue
        results.append((parent, best_score[parent_id]))

    logger.info(
        "%d children scanned -> %d parents returned",
        len(ranked_children),
        len(results),
    )
    return results


def retrieve_parents(
    vector_store: Chroma,
    parent_store: ParentStore,
    question: str,
    child_k: int,
    max_parents: int,
    metadata_filter: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """Vector-only retrieval: search children, return their unique parents."""
    child_hits = search_children(vector_store, question, child_k, metadata_filter)
    return parents_from_ranked_children(parent_store, child_hits, max_parents)
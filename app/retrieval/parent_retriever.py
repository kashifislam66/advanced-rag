import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.vectorstore.chroma_store import search_children
from app.vectorstore.parent_store import ParentStore

logger = logging.getLogger(__name__)


def retrieve_parents(
    vector_store: Chroma,
    parent_store: ParentStore,
    question: str,
    child_k: int,
    max_parents: int,
    metadata_filter: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """Search children, then return their unique parents (best match first)."""
    child_hits = search_children(vector_store, question, child_k, metadata_filter)

    best_distance: dict[str, float] = {}
    for child, distance in child_hits:
        parent_id = child.metadata.get("parent_id")
        if parent_id is None:
            logger.warning("Child without parent_id skipped: %s", child.metadata)
            continue
        if parent_id not in best_distance or distance < best_distance[parent_id]:
            best_distance[parent_id] = distance

    ranked_ids = sorted(best_distance, key=best_distance.get)[:max_parents]
    parents_by_id = parent_store.get_many(ranked_ids)

    results: list[tuple[Document, float]] = []
    for parent_id in ranked_ids:
        parent = parents_by_id.get(parent_id)
        if parent is None:
            logger.warning("Parent %s missing from store", parent_id)
            continue
        results.append((parent, best_distance[parent_id]))

    logger.info(
        "%d child hits -> %d unique parents -> %d returned",
        len(child_hits),
        len(best_distance),
        len(results),
    )
    return results
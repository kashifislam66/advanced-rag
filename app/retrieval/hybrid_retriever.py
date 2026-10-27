import logging
from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.retrieval.bm25_index import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.parent_retriever import parents_from_ranked_children
from app.vectorstore.chroma_store import make_child_id, search_children
from app.vectorstore.parent_store import ParentStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridHit:
    """One fused child chunk plus how each retriever ranked it (None = not found)."""

    chunk_id: str
    child: Document
    rrf_score: float
    bm25_rank: int | None
    vector_rank: int | None


def to_chroma_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate our simple filter into Chroma's syntax. Lists become $in."""
    if not metadata_filter:
        return None
    conditions: list[dict[str, Any]] = []
    for key, value in metadata_filter.items():
        if isinstance(value, (list, tuple, set)):
            if not value:
                raise ValueError(f"Filter '{key}' is an empty list; refusing to search")
            conditions.append({key: {"$in": list(value)}})
        else:
            conditions.append({key: value})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def hybrid_search_children(
    vector_store: Chroma,
    bm25_index: BM25Index,
    question: str,
    vector_k: int,
    bm25_k: int,
    fused_k: int,
    rrf_k: int = 60,
    metadata_filter: dict[str, Any] | None = None,
) -> list[HybridHit]:
    """Run BM25 and vector search on children, then fuse them with RRF."""
    bm25_hits = bm25_index.search(question, bm25_k, metadata_filter)
    vector_hits = search_children(
        vector_store, question, vector_k, to_chroma_filter(metadata_filter)
    )

    docs_by_id: dict[str, Document] = {}
    bm25_rank: dict[str, int] = {}
    vector_rank: dict[str, int] = {}
    bm25_ids: list[str] = []
    vector_ids: list[str] = []

    for position, (chunk_id, doc, _score) in enumerate(bm25_hits, start=1):
        docs_by_id[chunk_id] = doc
        bm25_rank[chunk_id] = position
        bm25_ids.append(chunk_id)

    for position, (doc, _distance) in enumerate(vector_hits, start=1):
        chunk_id = make_child_id(doc)
        docs_by_id.setdefault(chunk_id, doc)
        vector_rank.setdefault(chunk_id, position)
        vector_ids.append(chunk_id)

    fused = reciprocal_rank_fusion([bm25_ids, vector_ids], k=rrf_k)[:fused_k]

    logger.info(
        "Hybrid: bm25=%d vector=%d -> %d unique -> %d fused",
        len(bm25_ids),
        len(vector_ids),
        len(docs_by_id),
        len(fused),
    )
    return [
        HybridHit(
            chunk_id=chunk_id,
            child=docs_by_id[chunk_id],
            rrf_score=score,
            bm25_rank=bm25_rank.get(chunk_id),
            vector_rank=vector_rank.get(chunk_id),
        )
        for chunk_id, score in fused
    ]


def retrieve_hybrid_parents(
    vector_store: Chroma,
    bm25_index: BM25Index,
    parent_store: ParentStore,
    question: str,
    vector_k: int,
    bm25_k: int,
    fused_k: int,
    max_parents: int,
    rrf_k: int = 60,
    metadata_filter: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """Hybrid search on children, then return their unique parents."""
    hits = hybrid_search_children(
        vector_store, bm25_index, question,
        vector_k, bm25_k, fused_k, rrf_k, metadata_filter,
    )
    ranked_children = [(hit.child, hit.rrf_score) for hit in hits]
    return parents_from_ranked_children(parent_store, ranked_children, max_parents)
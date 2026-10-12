import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.reranking.context_order import order_for_context
from app.reranking.reranker import rerank_documents
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import hybrid_search_children
from app.retrieval.parent_retriever import parents_from_ranked_children
from app.vectorstore.parent_store import ParentStore

if TYPE_CHECKING:
    from app.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResources:
    """All the heavy objects the pipeline needs, created once at startup."""

    vector_store: Chroma
    bm25_index: BM25Index
    parent_store: ParentStore
    reranker: CrossEncoder


@dataclass(frozen=True)
class RetrievalParams:
    """All the numbers that control retrieval."""

    vector_k: int = 20
    bm25_k: int = 20
    fused_k: int = 20
    rrf_k: int = 60
    rerank_candidates: int = 20
    rerank_top_k: int = 8
    rerank_batch_size: int = 16
    max_parents: int = 5
    ordering: Literal["best_first", "edges"] = "edges"

    @classmethod
    def from_settings(cls, settings: "Settings") -> "RetrievalParams":
        return cls(
            vector_k=settings.retrieval_top_k,
            bm25_k=settings.bm25_top_k,
            fused_k=settings.hybrid_fused_k,
            rrf_k=settings.rrf_k,
            rerank_candidates=settings.reranker_candidates,
            rerank_top_k=settings.reranker_top_k,
            rerank_batch_size=settings.reranker_batch_size,
            max_parents=settings.retrieval_max_parents,
            ordering=settings.context_ordering,
        )


@dataclass(frozen=True)
class RetrievedContext:
    """One parent chunk ready for the LLM, with the reranker score of its best child."""

    parent: Document
    rerank_score: float


def retrieve_context(
    resources: RetrievalResources,
    question: str,
    params: RetrievalParams,
    metadata_filter: dict[str, Any] | None = None,
) -> list[RetrievedContext]:
    """Hybrid search -> rerank children -> unique parents -> context order."""
    if not question.strip():
        raise ValueError("Question must not be empty")

    hits = hybrid_search_children(
        resources.vector_store,
        resources.bm25_index,
        question,
        vector_k=params.vector_k,
        bm25_k=params.bm25_k,
        fused_k=params.fused_k,
        rrf_k=params.rrf_k,
        metadata_filter=metadata_filter,
    )
    if not hits:
        logger.info("No hybrid hits found")
        return []

    candidates = [hit.child for hit in hits[: params.rerank_candidates]]
    reranked_children = rerank_documents(
        resources.reranker,
        question,
        candidates,
        params.rerank_top_k,
        params.rerank_batch_size,
    )

    parents = parents_from_ranked_children(
        resources.parent_store, reranked_children, params.max_parents
    )
    if params.ordering == "edges":
        parents = order_for_context(parents)

    logger.info(
        "Pipeline: %d fused -> %d reranked children -> %d parents (%s order)",
        len(hits),
        len(reranked_children),
        len(parents),
        params.ordering,
    )
    return [RetrievedContext(parent=p, rerank_score=s) for p, s in parents]
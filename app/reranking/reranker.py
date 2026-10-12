import logging
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker(model_name: str, device: str, max_length: int) -> CrossEncoder:
    """Load the cross-encoder once per process and reuse it."""
    logger.info("Loading reranker: %s on %s", model_name, device)
    return CrossEncoder(model_name, device=device, max_length=max_length)


def rerank_documents(
    model: CrossEncoder,
    question: str,
    documents: list[Document],
    top_k: int,
    batch_size: int = 16,
    score_threshold: float = -5.0,
) -> list[tuple[Document, float]]:
    """Score every (question, document) pair and return the top_k, best first."""
    if not question.strip():
        raise ValueError("Question must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not documents:
        return []

    pairs = [(question, doc.page_content) for doc in documents]
    scores = np.asarray(
        model.predict(pairs, batch_size=batch_size, show_progress_bar=False),
        dtype=float,
    )

    if scores.ndim != 1 or len(scores) != len(documents):
        raise RuntimeError(
            f"Expected {len(documents)} scores, got shape {scores.shape}"
        )

    sorted_indices = np.argsort(-scores, kind="stable")
    # valid_indices = [i for i in sorted_indices if scores[i] >= score_threshold]
    final_order = sorted_indices[:top_k]
    logger.info("Reranked %d documents -> kept %d", len(documents), len(final_order))
    return [(documents[i], float(scores[i])) for i in final_order]
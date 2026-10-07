import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str, device: str, batch_size: int) -> Embeddings:
    """Load the embedding model once per process and reuse it."""
    logger.info("Loading embedding model: %s on %s", model_name, device)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _embed_batch(model: Embeddings, texts: list[str]) -> list[list[float]]:
    """Embed one batch, retrying on transient failures."""
    return model.embed_documents(texts)


def embed_texts(model: Embeddings, texts: list[str], batch_size: int) -> list[list[float]]:
    """Embed many texts in batches and return one vector per text, in order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not texts:
        return []

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(_embed_batch(model, batch))
        logger.info("Embedded %d/%d texts", min(start + batch_size, len(texts)), len(texts))

    if len(vectors) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} vectors, got {len(vectors)}")
    return vectors


def embed_query(model: Embeddings, question: str) -> list[float]:
    """Embed a user question."""
    if not question.strip():
        raise ValueError("Query must not be empty")
    return model.embed_query(question)


def get_embedding_dimension(model: Embeddings) -> int:
    """Return the vector size by embedding a probe string."""
    return len(model.embed_query("dimension probe"))
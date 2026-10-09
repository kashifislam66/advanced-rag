import hashlib
import logging
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

ALLOWED_METADATA_TYPES = (str, int, float, bool)


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only value types Chroma accepts; stringify others; drop None."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, ALLOWED_METADATA_TYPES):
            clean[key] = value
        elif value is not None:
            clean[key] = str(value)
    return clean


def make_child_id(child: Document) -> str:
    """Deterministic child ID so re-ingestion upserts instead of duplicating."""
    raw = (
        f"{child.metadata.get('parent_id')}|"
        f"{child.metadata.get('child_index')}|"
        f"{child.page_content}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def get_vector_store(
    embeddings: Embeddings,
    persist_dir: Path,
    collection_name: str,
) -> Chroma:
    """Open (or create) a persistent Chroma collection using cosine distance."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _upsert_batch(
    store: Chroma,
    texts: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str],
) -> None:
    """Embed and upsert one batch, retrying on transient failures."""
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def upsert_children(store: Chroma, children: list[Document], batch_size: int = 64) -> int:
    """Embed and store child chunks in batches. Returns the number processed."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not children:
        return 0

    for start in range(0, len(children), batch_size):
        batch = children[start : start + batch_size]
        _upsert_batch(
            store,
            texts=[c.page_content for c in batch],
            metadatas=[sanitize_metadata(c.metadata) for c in batch],
            ids=[make_child_id(c) for c in batch],
        )
        logger.info("Stored %d/%d children", min(start + batch_size, len(children)), len(children))
    return len(children)


def delete_by_source(store: Chroma, source: str) -> int:
    """Delete every child chunk that came from `source`. Returns how many."""
    existing = store.get(where={"source": source}, include=["metadatas"])
    ids: list[str] = existing["ids"]
    if ids:
        store.delete(ids=ids)
    logger.info("Deleted %d stale children for source=%s", len(ids), source)
    return len(ids)


def search_children(
    store: Chroma,
    question: str,
    k: int,
    metadata_filter: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """Return the k nearest children with their cosine DISTANCE (lower is better)."""
    if not question.strip():
        raise ValueError("Question must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    return store.similarity_search_with_score(query=question, k=k, filter=metadata_filter)


def count_children(store: Chroma) -> int:
    """Number of stored children. Uses Chroma's underlying collection."""
    return store._collection.count()  # noqa: SLF001 (no public count API)
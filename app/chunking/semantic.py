import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chunking.parent_child import SEPARATORS, link_parents_and_children

logger = logging.getLogger(__name__)

# Splits after English and Urdu sentence endings (. ? ! ۔ ؟).
SENTENCE_SPLIT_REGEX = r"(?<=[.?!۔؟])\s+"


def _merge_small_chunks(chunks: list[Document], min_size: int) -> list[Document]:
    """Merge a too-small chunk into the next chunk from the same source and page."""
    merged: list[Document] = []
    for chunk in chunks:
        if (
            merged
            and len(merged[-1].page_content) < min_size
            and merged[-1].metadata.get("source") == chunk.metadata.get("source")
            and merged[-1].metadata.get("page") == chunk.metadata.get("page")
        ):
            previous = merged[-1]
            merged[-1] = Document(
                page_content=f"{previous.page_content}\n\n{chunk.page_content}",
                metadata=previous.metadata,
            )
        else:
            merged.append(chunk)
    return merged


def build_semantic_parent_child_chunks(
    documents: list[Document],
    embeddings: Embeddings,
    parent_max_size: int,
    parent_min_size: int,
    child_size: int,
    child_overlap: int,
    breakpoint_percentile: float = 95.0,
) -> tuple[list[Document], list[Document]]:
    """Semantic parents (size-bounded) + recursive children."""
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_percentile,
        sentence_split_regex=SENTENCE_SPLIT_REGEX,
    )
    size_cap_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_max_size,
        chunk_overlap=0,
        separators=SEPARATORS,
    )

    semantic_chunks = semantic_splitter.split_documents(documents)
    merged_chunks = _merge_small_chunks(semantic_chunks, parent_min_size)
    parents = size_cap_splitter.split_documents(merged_chunks)

    children = link_parents_and_children(parents, child_size, child_overlap)

    logger.info(
        "Semantic: %d raw -> %d merged -> %d parents (after size cap), %d children",
        len(semantic_chunks),
        len(merged_chunks),
        len(parents),
        len(children),
    )
    return parents, children
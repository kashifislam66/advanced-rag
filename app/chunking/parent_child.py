import hashlib
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _make_parent_id(doc: Document, index: int) -> str:
    """Deterministic ID so re-ingesting the same file gives the same IDs."""
    raw = f"{doc.metadata.get('source')}|{doc.metadata.get('page')}|{index}|{doc.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_parent_child_chunks(
    documents: list[Document],
    parent_size: int,
    parent_overlap: int,
    child_size: int,
    child_overlap: int,
) -> tuple[list[Document], list[Document]]:
    """Split documents into large parent chunks and small searchable child chunks."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
        separators=SEPARATORS,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=SEPARATORS,
    )

    parents = parent_splitter.split_documents(documents)
    children: list[Document] = []

    for index, parent in enumerate(parents):
        parent_id = _make_parent_id(parent, index)
        parent.metadata["parent_id"] = parent_id

        for child_index, child in enumerate(child_splitter.split_documents([parent])):
            child.metadata["parent_id"] = parent_id
            child.metadata["child_index"] = child_index
            children.append(child)

    logger.info("Created %d parents and %d children", len(parents), len(children))
    return parents, children
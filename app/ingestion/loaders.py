import logging
import re
from pathlib import Path

from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".csv"}


def _build_loader(path: Path) -> BaseLoader:
    """Return the correct LangChain loader for a file type."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8")
    if suffix == ".html":
        return BSHTMLLoader(str(path), open_encoding="utf-8")
    if suffix == ".csv":
        return CSVLoader(str(path), encoding="utf-8")
    raise ValueError(f"Unsupported file type: {suffix}")


def normalize_text(text: str) -> str:
    """Clean common text noise without changing meaning."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(directory: Path) -> list[Document]:
    """Load all supported files under `directory` into normalized Documents."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            loaded = _build_loader(path).load()
        except Exception:
            logger.exception("Failed to load %s, skipping", path)
            continue

        for doc in loaded:
            doc.page_content = normalize_text(doc.page_content)
            if not doc.page_content:
                continue
            doc.metadata["source"] = str(path.relative_to(directory))
            doc.metadata["file_type"] = path.suffix.lower().lstrip(".")
            documents.append(doc)

    logger.info("Loaded %d documents from %s", len(documents), directory)
    return documents
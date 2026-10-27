import logging
import re
import unicodedata
from typing import Any

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

# Arabic yeh/kaf -> Urdu/Persian yeh/kaf, so the same word is not split in two.
_CHAR_FIXES = str.maketrans({"\u064a": "\u06cc", "\u0643": "\u06a9"})


def tokenize(text: str) -> list[str]:
    """Lowercase, normalize and split text into word tokens."""
    text = unicodedata.normalize("NFKC", text).lower().translate(_CHAR_FIXES)
    return TOKEN_PATTERN.findall(text)


def _matches(doc: Document, metadata_filter: dict[str, Any]) -> bool:
    """True if every filter condition holds. A list value means 'any of these'."""
    for key, expected in metadata_filter.items():
        actual = doc.metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True

class BM25Index:
    """In-memory BM25 keyword index over child chunks."""

    def __init__(self, records: list[tuple[str, Document]]) -> None:
        if not records:
            raise ValueError("Cannot build a BM25 index from zero chunks")
        self._ids = [chunk_id for chunk_id, _ in records]
        self._docs = [doc for _, doc in records]
        tokenized_corpus = [tokenize(doc.page_content) for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index over %d chunks", len(self._docs))

    def search(
        self,
        question: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, Document, float]]:
        """Return up to k (id, chunk, bm25_score) triples, best first."""
        if k <= 0:
            raise ValueError("k must be positive")
        tokens = tokenize(question)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        if metadata_filter:
            allowed = np.array([_matches(doc, metadata_filter) for doc in self._docs])
            scores = np.where(allowed, scores, 0.0)

        best_indices = np.argsort(scores)[::-1][:k]
        return [
            (self._ids[i], self._docs[i], float(scores[i]))
            for i in best_indices
            if scores[i] > 0
        ]
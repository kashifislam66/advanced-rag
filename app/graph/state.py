import operator
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    """Shared state passed between graph nodes. Keep it serializable."""

    question: str
    metadata_filter: dict[str, Any] | None
    parents: list[Document]
    answer_text: str
    citations: list[dict[str, Any]]
    grounded: bool
    input_tokens: int | None
    output_tokens: int | None
    trace: Annotated[list[str], operator.add]
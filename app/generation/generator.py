import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from app.generation.prompts import (
    NO_ANSWER_TEXT,
    SourceRef,
    build_messages,
    format_context,
)

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class GenerationError(RuntimeError):
    """Raised when the LLM call fails after its built-in retries."""


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[SourceRef]
    grounded: bool  # True only if the model answered AND cited at least one valid source
    input_tokens: int | None
    output_tokens: int | None


def message_text(content: object) -> str:
    """LLM content is usually a string, but can be a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def clean_citations(text: str, refs: list[SourceRef]) -> str:
    """Remove citation markers that point to sources which do not exist."""
    valid = {ref.number for ref in refs}

    def keep_or_drop(match: re.Match) -> str:
        if int(match.group(1)) in valid:
            return match.group(0)
        logger.warning("Dropped invented citation %s", match.group(0))
        return ""

    return CITATION_PATTERN.sub(keep_or_drop, text)


def extract_citations(text: str, refs: list[SourceRef]) -> list[SourceRef]:
    """Return the distinct sources the answer cites, in order of first use."""
    by_number = {ref.number: ref for ref in refs}
    used: list[SourceRef] = []
    seen: set[int] = set()
    for match in CITATION_PATTERN.finditer(text):
        number = int(match.group(1))
        if number in by_number and number not in seen:
            seen.add(number)
            used.append(by_number[number])
    return used


def generate_answer(
    llm: BaseChatModel,
    question: str,
    documents: list[Document],
    max_chars_per_source: int,
) -> Answer:
    """Answer the question from the given parent chunks, with validated citations."""
    if not question.strip():
        raise ValueError("Question must not be empty")

    if not documents:
        logger.info("No context retrieved, skipping the LLM call")
        return Answer(NO_ANSWER_TEXT, [], False, 0, 0)

    context, refs = format_context(documents, max_chars_per_source)

    try:
        response = llm.invoke(build_messages(question, context))
    except Exception as exc:
        logger.exception("LLM call failed")
        raise GenerationError("LLM call failed") from exc

    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    logger.info("LLM tokens: input=%s output=%s", input_tokens, output_tokens)

    text = message_text(response.content).strip()
    if not text or NO_ANSWER_TEXT.lower() in text.lower():
        return Answer(NO_ANSWER_TEXT, [], False, input_tokens, output_tokens)

    text = clean_citations(text, refs)
    citations = extract_citations(text, refs)
    if not citations:
        logger.warning("Answer has no valid citations")

    return Answer(text, citations, bool(citations), input_tokens, output_tokens)
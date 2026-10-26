import logging
from collections import Counter
from pathlib import Path
from typing import Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

Category = Literal[
    "direct", "exact_id", "paraphrase", "roman_urdu", "multi_doc", "unanswerable"
]
Split = Literal["dev", "test"]

MIN_QUESTIONS = 20


class EvalItem(BaseModel):
    """One evaluation question with its answer key and the file(s) holding the answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str
    reference: str = ""
    reference_sources: list[str] = Field(default_factory=list)
    category: Category
    split: Split = "dev"

    @field_validator("id", "question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"

    @model_validator(mode="after")
    def check_consistency(self) -> "EvalItem":
        if self.answerable:
            if not self.reference.strip():
                raise ValueError("answerable questions need a reference answer")
            if not self.reference_sources:
                raise ValueError("answerable questions need reference_sources")
            if self.category == "multi_doc" and len(set(self.reference_sources)) < 2:
                raise ValueError("multi_doc questions need at least 2 distinct sources")
        elif self.reference_sources:
            raise ValueError("unanswerable questions must not list reference_sources")
        return self


def load_eval_set(path: Path) -> list[EvalItem]:
    """Load and validate a JSONL evaluation set, reporting every problem at once."""
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation set not found: {path}")

    items: list[EvalItem] = []
    seen_ids: set[str] = set()
    problems: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = EvalItem.model_validate_json(line)
            except ValidationError as exc:
                details = "; ".join(
                    f"{'.'.join(str(part) for part in err['loc']) or 'row'}: {err['msg']}"
                    for err in exc.errors()
                )
                problems.append(f"line {line_number}: {details}")
                continue
            if item.id in seen_ids:
                problems.append(f"line {line_number}: duplicate id '{item.id}'")
                continue
            seen_ids.add(item.id)
            items.append(item)

    if problems:
        raise ValueError("Invalid evaluation set:\n" + "\n".join(problems))
    if not items:
        raise ValueError(f"Evaluation set is empty: {path}")

    logger.info("Loaded %d evaluation questions from %s", len(items), path)
    return items


def find_unknown_sources(
    items: list[EvalItem], known_sources: set[str]
) -> dict[str, list[str]]:
    """Map question id -> reference sources that are not in the ingested corpus."""
    unknown: dict[str, list[str]] = {}
    for item in items:
        missing = [source for source in item.reference_sources if source not in known_sources]
        if missing:
            unknown[item.id] = missing
    return unknown


def summarize(items: list[EvalItem]) -> dict[str, object]:
    """Counts by category and split."""
    return {
        "total": len(items),
        "by_category": dict(Counter(item.category for item in items)),
        "by_split": dict(Counter(item.split for item in items)),
    }


def coverage_warnings(items: list[EvalItem]) -> list[str]:
    """Soft warnings about a weak evaluation set (not errors)."""
    warnings: list[str] = []
    if len(items) < MIN_QUESTIONS:
        warnings.append(f"Only {len(items)} questions; aim for at least {MIN_QUESTIONS}.")
    present = {item.category for item in items}
    for category in get_args(Category):
        if category not in present:
            warnings.append(f"No '{category}' questions.")
    if not any(item.split == "test" for item in items):
        warnings.append("No questions in the 'test' split yet.")
    return warnings
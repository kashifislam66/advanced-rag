from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config.settings import settings


class AskRequest(BaseModel):
    """The ONLY thing a client may send. No filters, no departments."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=settings.max_question_chars)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class CitationOut(BaseModel):
    number: int
    source: str
    page: int | None = None


class UsageOut(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    note: str = "answer-generation tokens only; grader and rewrite calls are not included"


class AskResponse(BaseModel):
    request_id: str
    answer: str
    grounded: bool
    status: str
    citations: list[CitationOut]
    usage: UsageOut
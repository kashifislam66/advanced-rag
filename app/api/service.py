import logging
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator

from app.api.security import User
from app.generation.prompts import NO_ANSWER_TEXT

logger = logging.getLogger(__name__)


class ServiceBusy(RuntimeError):
    """All processing slots are taken and the wait time ran out."""


@dataclass(frozen=True)
class AskResult:
    answer: str
    citations: list[dict[str, Any]]
    grounded: bool
    status: str
    input_tokens: int | None
    output_tokens: int | None
    trace: list[str]


def _to_result(state: dict[str, Any]) -> AskResult:
    grounded = bool(state.get("grounded"))
    return AskResult(
        answer=state.get("answer_text") or NO_ANSWER_TEXT,
        citations=list(state.get("citations", [])),
        grounded=grounded,
        status=state.get("final_status") or ("answered" if grounded else "no_answer"),
        input_tokens=state.get("input_tokens"),
        output_tokens=state.get("output_tokens"),
        trace=list(state.get("trace", [])),
    )


def _merge(state: dict[str, Any], changes: dict[str, Any]) -> None:
    """Fold one graph update into the running state (trace is appended, the rest replaced)."""
    for key, value in changes.items():
        if key == "trace":
            state.setdefault("trace", []).extend(value)
        else:
            state[key] = value


class RagService:
    """Runs the RAG graph for one authenticated user, with a cap on parallel work."""

    def __init__(
        self,
        graph: Any,
        max_concurrent: int,
        wait_seconds: float,
        ready_check: Callable[[], bool],
        recursion_limit: int = 25,
    ) -> None:
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        self._graph = graph
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._wait = wait_seconds
        self._ready_check = ready_check
        self._config = {"recursion_limit": recursion_limit}

    def is_ready(self) -> bool:
        try:
            return bool(self._ready_check())
        except Exception:
            logger.exception("Readiness check failed")
            return False

    @contextmanager
    def _slot(self) -> Iterator[None]:
        if not self._slots.acquire(timeout=self._wait):
            raise ServiceBusy("all processing slots are busy")
        try:
            yield
        finally:
            self._slots.release()

    @staticmethod
    def _initial(question: str, user: User) -> dict[str, Any]:
        return {"question": question, "metadata_filter": user.metadata_filter()}

    def ask(self, question: str, user: User) -> AskResult:
        with self._slot():
            state = self._graph.invoke(self._initial(question, user), config=self._config)
        return _to_result(state)

    def stream(self, question: str, user: User) -> Iterator[dict[str, Any]]:
        state: dict[str, Any] = {}
        with self._slot():
            for update in self._graph.stream(
                self._initial(question, user), config=self._config, stream_mode="updates"
            ):
                for node, changes in update.items():
                    _merge(state, changes or {})
                    yield {"event": "progress", "node": node}
        yield {"event": "final", **asdict(_to_result(state))}
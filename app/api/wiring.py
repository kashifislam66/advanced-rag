import logging
from typing import TYPE_CHECKING

from app.api.deps import Deps
from app.api.security import Authenticator, RateLimiter
from app.api.service import RagService
from app.bootstrap import AppContainer, build_container
from app.generation.generator import generate_answer
from app.graph.crag_graph import build_crag_graph
from app.graph.graders import check_answer, grade_documents, rewrite_query
from app.graph.rag_graph import build_rag_graph
from app.retrieval.pipeline import retrieve_context
from app.vectorstore.chroma_store import count_children

if TYPE_CHECKING:
    from app.config.settings import Settings

logger = logging.getLogger(__name__)


def build_deps(settings: "Settings") -> tuple[Deps, AppContainer]:
    """Create the real objects. Auth is checked first so a missing key file fails fast."""
    authenticator = Authenticator.from_file(settings.api_keys_file)
    app = build_container(settings)

    def retrieve_fn(query, metadata_filter):
        contexts = retrieve_context(app.resources, query, app.params, metadata_filter)
        return [c.parent for c in contexts]

    def generate_fn(question, parents):
        return generate_answer(app.answer_llm, question, parents, settings.max_chars_per_source)

    if settings.rag_mode == "crag":
        graph = build_crag_graph(
            retrieve_fn,
            lambda q, docs: grade_documents(app.judge_llm, q, docs, settings.grader_chars_per_source),
            generate_fn,
            lambda q, previous: rewrite_query(app.judge_llm, q, previous),
            lambda text, docs: check_answer(app.judge_llm, text, docs, settings.max_chars_per_source),
            max_retries=settings.crag_max_retries,
        )
    else:
        graph = build_rag_graph(retrieve_fn, generate_fn)

    service = RagService(
        graph,
        max_concurrent=settings.max_concurrent_requests,
        wait_seconds=settings.queue_wait_seconds,
        ready_check=lambda: count_children(app.resources.vector_store) > 0,
    )
    limiter = RateLimiter(settings.rate_limit_per_minute)
    logger.info("Service ready (mode=%s)", settings.rag_mode)
    return Deps(service=service, authenticator=authenticator, limiter=limiter), app
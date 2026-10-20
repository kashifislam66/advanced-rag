import logging
from dataclasses import asdict
from typing import Any, Callable, Literal

from langchain_core.documents import Document
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.generation.generator import Answer, GenerationError
from app.generation.prompts import NO_ANSWER_TEXT
from app.graph.graders import AnswerCheck
from app.graph.state import CRAGState

logger = logging.getLogger(__name__)

RetrieveFn = Callable[[str, dict[str, Any] | None], list[Document]]
GradeFn = Callable[[str, list[Document]], list[Document]]
GenerateFn = Callable[[str, list[Document]], Answer]
RewriteFn = Callable[[str, str], str]
CheckFn = Callable[[str, list[Document]], AnswerCheck]


def decide_after_failure(retry_count: int, max_retries: int) -> Literal["rewrite", "give_up"]:
    """Pure rule: try again only while retries remain."""
    return "rewrite" if retry_count < max_retries else "give_up"


def build_crag_graph(
    retrieve_fn: RetrieveFn,
    grade_fn: GradeFn,
    generate_fn: GenerateFn,
    rewrite_fn: RewriteFn,
    check_fn: CheckFn,
    max_retries: int = 2,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Build the corrective RAG graph. All heavy resources are injected as functions."""
    if max_retries < 0:
        raise ValueError("max_retries must be 0 or more")

    # ---------- nodes ----------
    def retrieve_node(state: CRAGState) -> dict[str, Any]:
        attempt = state.get("retry_count", 0) + 1
        query = state.get("search_query") or state["question"]
        parents = retrieve_fn(query, state.get("metadata_filter"))
        return {"parents": parents, "trace": [f"retrieve (attempt {attempt}): {len(parents)} parents"]}

    def grade_node(state: CRAGState) -> dict[str, Any]:
        parents = state.get("parents", [])
        relevant = grade_fn(state["question"], parents) if parents else []
        return {
            "graded_parents": relevant,
            "trace": [f"grade: {len(relevant)}/{len(parents)} relevant"],
        }

    def generate_node(state: CRAGState) -> dict[str, Any]:
        answer = generate_fn(state["question"], state["graded_parents"])
        return {
            "answer_text": answer.text,
            "citations": [asdict(c) for c in answer.citations],
            "grounded": answer.grounded,
            "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens,
            "trace": [f"generate: grounded={answer.grounded}"],
        }

    def check_answer_node(state: CRAGState) -> dict[str, Any]:
        verdict = check_fn(state["answer_text"], state["graded_parents"])
        update: dict[str, Any] = {
            "answer_supported": verdict.supported,
            "trace": [
                f"check: supported={verdict.supported}, "
                f"unsupported_claims={len(verdict.unsupported_claims)}"
            ],
        }
        if verdict.supported:
            update["final_status"] = "answered"
        return update

    def rewrite_node(state: CRAGState) -> dict[str, Any]:
        count = state.get("retry_count", 0)
        previous = state.get("search_query") or state["question"]
        new_query = rewrite_fn(state["question"], previous)
        return {
            "search_query": new_query,
            "retry_count": count + 1,
            # reset everything from the failed attempt so nothing stale survives
            "parents": [],
            "graded_parents": [],
            "answer_text": "",
            "citations": [],
            "grounded": False,
            "answer_supported": None,
            "trace": [f"rewrite #{count + 1}"],
        }

    def give_up_node(state: CRAGState) -> dict[str, Any]:
        if state.get("answer_supported") is False:
            status = "unsupported_answer"
        elif not state.get("graded_parents"):
            status = "no_relevant_documents"
        else:
            status = "no_grounded_answer"
        return {
            "answer_text": NO_ANSWER_TEXT,
            "citations": [],
            "grounded": False,
            "final_status": status,
            "trace": [f"give_up: {status}"],
        }

    # ---------- routers (traffic police) ----------
    def route_after_grade(state: CRAGState) -> Literal["generate", "rewrite", "give_up"]:
        if state.get("graded_parents"):
            return "generate"
        return decide_after_failure(state.get("retry_count", 0), max_retries)

    def route_after_generate(state: CRAGState) -> Literal["check_answer", "rewrite", "give_up"]:
        if state.get("grounded"):
            return "check_answer"
        return decide_after_failure(state.get("retry_count", 0), max_retries)

    def route_after_check(state: CRAGState) -> Literal["done", "rewrite", "give_up"]:
        if state.get("answer_supported"):
            return "done"
        return decide_after_failure(state.get("retry_count", 0), max_retries)

    # ---------- wiring ----------
    builder = StateGraph(CRAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node(
        "generate",
        generate_node,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_interval=1.0,
            backoff_factor=2.0,
            retry_on=GenerationError,
        ),
    )
    builder.add_node("check_answer", check_answer_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("give_up", give_up_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade", route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "give_up": "give_up"},
    )
    builder.add_conditional_edges(
        "generate", route_after_generate,
        {"check_answer": "check_answer", "rewrite": "rewrite", "give_up": "give_up"},
    )
    builder.add_conditional_edges(
        "check_answer", route_after_check,
        {"done": END, "rewrite": "rewrite", "give_up": "give_up"},
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("give_up", END)

    return builder.compile(checkpointer=checkpointer)
import logging
from dataclasses import asdict
from typing import Any, Callable, Literal

from langchain_core.documents import Document
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.generation.generator import Answer, GenerationError
from app.generation.prompts import NO_ANSWER_TEXT
from app.graph.state import RAGState

logger = logging.getLogger(__name__)

RetrieveFn = Callable[[str, dict[str, Any] | None], list[Document]]
GenerateFn = Callable[[str, list[Document]], Answer]


def route_after_retrieve(state: RAGState) -> Literal["generate", "no_context"]:
    """Traffic police: call the LLM only if retrieval found something."""
    return "generate" if state.get("parents") else "no_context"


def build_rag_graph(
    retrieve_fn: RetrieveFn,
    generate_fn: GenerateFn,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Build and compile the RAG graph. Heavy resources are injected as functions."""

    def retrieve_node(state: RAGState) -> dict[str, Any]:
        parents = retrieve_fn(state["question"], state.get("metadata_filter"))
        logger.info("retrieve node: %d parents", len(parents))
        return {"parents": parents, "trace": [f"retrieve: {len(parents)} parents"]}

    def generate_node(state: RAGState) -> dict[str, Any]:
        answer = generate_fn(state["question"], state["parents"])
        return {
            "answer_text": answer.text,
            "citations": [asdict(c) for c in answer.citations],
            "grounded": answer.grounded,
            "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens,
            "trace": [
                f"generate: grounded={answer.grounded}, citations={len(answer.citations)}"
            ],
        }

    def no_context_node(state: RAGState) -> dict[str, Any]:
        return {
            "answer_text": NO_ANSWER_TEXT,
            "citations": [],
            "grounded": False,
            "trace": ["no_context: nothing retrieved, LLM skipped"],
        }

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
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
    builder.add_node("no_context", no_context_node)

    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "no_context": "no_context"},
    )
    builder.add_edge("generate", END)
    builder.add_edge("no_context", END)

    return builder.compile(checkpointer=checkpointer)
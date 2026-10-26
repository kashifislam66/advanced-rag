import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from app.bootstrap import AppContainer
from app.evaluation.runner import SystemFn, SystemOutput
from app.generation.generator import generate_answer
from app.generation.prompts import NO_ANSWER_TEXT
from app.graph.crag_graph import build_crag_graph
from app.graph.graders import check_answer, grade_documents, rewrite_query
from app.retrieval.hybrid_retriever import hybrid_search_children
from app.retrieval.parent_retriever import parents_from_ranked_children
from app.retrieval.pipeline import retrieve_context
from app.vectorstore.chroma_store import search_children

if TYPE_CHECKING:
    from app.config.settings import Settings

logger = logging.getLogger(__name__)

VARIANT_NAMES = [
    "vector_only",
    "bm25_only",
    "hybrid",
    "hybrid_rerank",
    "hybrid_rerank_edges",
    "crag",
]


def build_variants(app: AppContainer, settings: "Settings") -> dict[str, SystemFn]:
    """Every variant uses the same generator, prompt and max_parents; only retrieval differs."""
    res = app.resources
    params = app.params
    best_first = replace(params, ordering="best_first")
    edges = replace(params, ordering="edges")

    def finish(
        question: str,
        parents: list[Document],
        rank_valid: bool = True,
        extra_calls: int = 0,
    ) -> SystemOutput:
        answer = generate_answer(app.answer_llm, question, parents, settings.max_chars_per_source)
        status = "answered" if answer.grounded else "no_grounded_answer"
        return SystemOutput(
            contexts=parents,
            answer_text=answer.text,
            final_status=status,
            rank_valid=rank_valid,
            llm_calls=(1 if parents else 0) + extra_calls,
        )

    def vector_only(question: str) -> SystemOutput:
        hits = search_children(res.vector_store, question, params.vector_k)
        ranked = parents_from_ranked_children(res.parent_store, hits, params.max_parents)
        return finish(question, [parent for parent, _ in ranked])

    def bm25_only(question: str) -> SystemOutput:
        hits = res.bm25_index.search(question, params.bm25_k)
        ranked_children = [(doc, score) for _, doc, score in hits]
        ranked = parents_from_ranked_children(res.parent_store, ranked_children, params.max_parents)
        return finish(question, [parent for parent, _ in ranked])

    def hybrid(question: str) -> SystemOutput:
        hits = hybrid_search_children(
            res.vector_store, res.bm25_index, question,
            vector_k=params.vector_k, bm25_k=params.bm25_k,
            fused_k=params.fused_k, rrf_k=params.rrf_k,
        )
        ranked_children = [(hit.child, hit.rrf_score) for hit in hits]
        ranked = parents_from_ranked_children(res.parent_store, ranked_children, params.max_parents)
        return finish(question, [parent for parent, _ in ranked])

    def hybrid_rerank(question: str) -> SystemOutput:
        contexts = retrieve_context(res, question, best_first)
        return finish(question, [c.parent for c in contexts])

    def hybrid_rerank_edges(question: str) -> SystemOutput:
        contexts = retrieve_context(res, question, edges)
        return finish(question, [c.parent for c in contexts], rank_valid=False)

    def crag(question: str) -> SystemOutput:
        last_retrieved: list[Document] = []
        calls = [0]  # a one-item list so the small functions below can change it

        def retrieve_fn(query, metadata_filter):
            parents = [c.parent for c in retrieve_context(res, query, best_first, metadata_filter)]
            last_retrieved[:] = parents
            return parents

        def grade_fn(q, docs):
            calls[0] += 1
            return grade_documents(app.judge_llm, q, docs, settings.grader_chars_per_source)

        def generate_fn(q, docs):
            calls[0] += 1
            return generate_answer(app.answer_llm, q, docs, settings.max_chars_per_source)

        def rewrite_fn(q, previous):
            calls[0] += 1
            return rewrite_query(app.judge_llm, q, previous)

        def check_fn(text, docs):
            calls[0] += 1
            return check_answer(app.judge_llm, text, docs, settings.max_chars_per_source)

        graph = build_crag_graph(
            retrieve_fn, grade_fn, generate_fn, rewrite_fn, check_fn,
            max_retries=settings.crag_max_retries,
        )
        final = graph.invoke({"question": question}, config={"recursion_limit": 25})

        status = final.get("final_status", "unknown")
        contexts = final.get("graded_parents") if status == "answered" else last_retrieved
        return SystemOutput(
            contexts=list(contexts or []),
            answer_text=final.get("answer_text", NO_ANSWER_TEXT),
            final_status=status,
            rank_valid=True,
            llm_calls=calls[0],
        )

    return {
        "vector_only": vector_only,
        "bm25_only": bm25_only,
        "hybrid": hybrid,
        "hybrid_rerank": hybrid_rerank,
        "hybrid_rerank_edges": hybrid_rerank_edges,
        "crag": crag,
    }
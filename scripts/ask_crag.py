import sys
import uuid

from langgraph.checkpoint.memory import InMemorySaver

from app.config.settings import settings
from app.embeddings.embedder import get_embedding_model
from app.generation.generator import generate_answer
from app.generation.llm import get_llm
from app.graph.crag_graph import build_crag_graph
from app.graph.graders import check_answer, grade_documents, rewrite_query
from app.reranking.reranker import get_reranker
from app.retrieval.bm25_index import BM25Index
from app.retrieval.pipeline import RetrievalParams, RetrievalResources, retrieve_context
from app.utils.logger import setup_logging
from app.vectorstore.chroma_store import get_all_children, get_vector_store
from app.vectorstore.parent_store import ParentStore

setup_logging(settings.log_level)
question = " ".join(sys.argv[1:]) or "How many days of annual leave do employees get?"

embedder = get_embedding_model(
    settings.embedding_model_name, settings.embedding_device, settings.embedding_batch_size
)
store = get_vector_store(embedder, settings.chroma_persist_dir, settings.chroma_collection_name)
parent_store = ParentStore(settings.parent_store_path)
resources = RetrievalResources(
    vector_store=store,
    bm25_index=BM25Index(get_all_children(store)),
    parent_store=parent_store,
    reranker=get_reranker(
        settings.reranker_model_name, settings.reranker_device, settings.reranker_max_length
    ),
)
params = RetrievalParams.from_settings(settings)

api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
answer_llm = get_llm(
    settings.llm_provider, settings.llm_model, api_key, settings.llm_temperature,
    settings.llm_max_tokens, settings.llm_timeout_seconds, settings.llm_max_retries,
)
judge_llm = get_llm(
    settings.llm_provider, settings.grader_model or settings.llm_model, api_key, 0.0,
    settings.grader_max_tokens, settings.llm_timeout_seconds, settings.llm_max_retries,
)


def retrieve_fn(query, metadata_filter):
    return [c.parent for c in retrieve_context(resources, query, params, metadata_filter)]


def grade_fn(q, docs):
    return grade_documents(judge_llm, q, docs, settings.grader_chars_per_source)


def generate_fn(q, docs):
    return generate_answer(answer_llm, q, docs, settings.max_chars_per_source)


def rewrite_fn(q, previous):
    return rewrite_query(judge_llm, q, previous)


def check_fn(text, docs):
    return check_answer(judge_llm, text, docs, settings.max_chars_per_source)


graph = build_crag_graph(
    retrieve_fn, grade_fn, generate_fn, rewrite_fn, check_fn,
    max_retries=settings.crag_max_retries,
    checkpointer=InMemorySaver(),
)
config = {"configurable": {"thread_id": f"cli-{uuid.uuid4()}"}, "recursion_limit": 25}

print(f"\nQuestion: {question}\n")
for update in graph.stream({"question": question}, config=config, stream_mode="updates"):
    for node_name, changes in update.items():
        print(f"[{node_name}] {changes.get('trace')}")

final = graph.get_state(config).values
print(f"\n{final['answer_text']}")
print(f"\nStatus: {final.get('final_status')} | grounded: {final['grounded']} "
      f"| retries used: {final.get('retry_count', 0)}")
for ref in final["citations"]:
    page = f", page {ref['page']}" if ref["page"] is not None else ""
    print(f"  [{ref['number']}] {ref['source']}{page}")
print(f"\nTrace: {final['trace']}")

parent_store.close()
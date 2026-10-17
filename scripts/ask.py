import sys
import time

from app.config.settings import settings
from app.embeddings.embedder import get_embedding_model
from app.generation.generator import generate_answer
from app.generation.llm import get_llm
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

llm = get_llm(
    settings.llm_provider,
    settings.llm_model,
    settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
    settings.llm_temperature,
    settings.llm_max_tokens,
    settings.llm_timeout_seconds,
    settings.llm_max_retries,
)

start = time.perf_counter()
contexts = retrieve_context(resources, question, params)
retrieval_ms = (time.perf_counter() - start) * 1000

start = time.perf_counter()
answer = generate_answer(
    llm, question, [c.parent for c in contexts], settings.max_chars_per_source
)
llm_ms = (time.perf_counter() - start) * 1000

print(f"\nQuestion: {question}\n")
print(answer.text)
print(f"\nGrounded: {answer.grounded}")
for ref in answer.citations:
    page = f", page {ref.page}" if ref.page is not None else ""
    print(f"  [{ref.number}] {ref.source}{page}")
print(
    f"\nretrieval={retrieval_ms:.0f} ms  llm={llm_ms:.0f} ms  "
    f"tokens in/out={answer.input_tokens}/{answer.output_tokens}"
)

parent_store.close()
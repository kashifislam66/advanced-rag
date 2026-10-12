import sys
import time

from app.config.settings import settings
from app.embeddings.embedder import get_embedding_model
from app.reranking.reranker import get_reranker
from app.retrieval.bm25_index import BM25Index
from app.retrieval.pipeline import RetrievalParams, RetrievalResources, retrieve_context
from app.utils.logger import setup_logging
from app.vectorstore.chroma_store import get_all_children, get_vector_store
from app.vectorstore.parent_store import ParentStore

setup_logging(settings.log_level)
question = " ".join(sys.argv[1:]) or "leave policy"

embedder = get_embedding_model(
    settings.embedding_model_name,
    settings.embedding_device,
    settings.embedding_batch_size,
)
store = get_vector_store(embedder, settings.chroma_persist_dir, settings.chroma_collection_name)
parent_store = ParentStore(settings.parent_store_path)

resources = RetrievalResources(
    vector_store=store,
    bm25_index=BM25Index(get_all_children(store)),
    parent_store=parent_store,
    reranker=get_reranker(
        settings.reranker_model_name,
        settings.reranker_device,
        settings.reranker_max_length,
    ),
)
params = RetrievalParams.from_settings(settings)

start = time.perf_counter()
results = retrieve_context(resources, question, params)
elapsed_ms = (time.perf_counter() - start) * 1000

print(f"\nQuestion: {question}")
print(f"Ordering: {params.ordering} | total time: {elapsed_ms:.0f} ms\n")
for position, item in enumerate(results, start=1):
    meta = item.parent.metadata
    print(
        f"[{position}] rerank={item.rerank_score:6.2f}  "
        f"{meta.get('source')}  page={meta.get('page')}  "
        f"({len(item.parent.page_content)} chars)"
    )
    print("    ", item.parent.page_content[:120].replace("\n", " "), "...")

parent_store.close()
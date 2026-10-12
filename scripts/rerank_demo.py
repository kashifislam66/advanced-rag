import sys
import time

from app.config.settings import settings
from app.embeddings.embedder import get_embedding_model
from app.reranking.reranker import get_reranker, rerank_documents
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import hybrid_search_children
from app.utils.logger import setup_logging
from app.vectorstore.chroma_store import get_all_children, get_vector_store

setup_logging(settings.log_level)
question = " ".join(sys.argv[1:]) or "leave policy"

embedder = get_embedding_model(
    settings.embedding_model_name,
    settings.embedding_device,
    settings.embedding_batch_size,
)
store = get_vector_store(embedder, settings.chroma_persist_dir, settings.chroma_collection_name)
index = BM25Index(get_all_children(store))

hits = hybrid_search_children(
    store, index, question,
    vector_k=settings.retrieval_top_k,
    bm25_k=settings.bm25_top_k,
    fused_k=settings.hybrid_fused_k,
    rrf_k=settings.rrf_k,
)
candidates = [hit.child for hit in hits[: settings.reranker_candidates]]
original_rank = {id(doc): position for position, doc in enumerate(candidates, start=1)}

reranker = get_reranker(
    settings.reranker_model_name, settings.reranker_device, settings.reranker_max_length
)

start = time.perf_counter()
reranked = rerank_documents(
    reranker, question, candidates, settings.reranker_top_k, settings.reranker_batch_size
)
elapsed_ms = (time.perf_counter() - start) * 1000

print(f"\nReranked {len(candidates)} candidates in {elapsed_ms:.0f} ms\n")
print("naya  pehle  score   text")
for new_position, (doc, score) in enumerate(reranked, start=1):
    print(
        f"{new_position:>4}  {original_rank[id(doc)]:>5}  {score:6.2f}  "
        f"{doc.page_content[:80]!r}"
    )
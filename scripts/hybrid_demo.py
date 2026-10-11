import sys

from app.config.settings import settings
from app.embeddings.embedder import get_embedding_model
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import hybrid_search_children, retrieve_hybrid_parents
from app.utils.logger import setup_logging
from app.vectorstore.chroma_store import get_all_children, get_vector_store
from app.vectorstore.parent_store import ParentStore

setup_logging(settings.log_level)
question = " ".join(sys.argv[1:]) or "leave policy"

model = get_embedding_model(
    settings.embedding_model_name,
    settings.embedding_device,
    settings.embedding_batch_size,
)
store = get_vector_store(model, settings.chroma_persist_dir, settings.chroma_collection_name)
parent_store = ParentStore(settings.parent_store_path)
index = BM25Index(get_all_children(store))

print("\n=== Fused children (b = BM25 rank, v = vector rank) ===")
hits = hybrid_search_children(
    store, index, question,
    vector_k=settings.retrieval_top_k,
    bm25_k=settings.bm25_top_k,
    fused_k=settings.hybrid_fused_k,
    rrf_k=settings.rrf_k,
)
for hit in hits[:8]:
    b = hit.bm25_rank if hit.bm25_rank is not None else "-"
    v = hit.vector_rank if hit.vector_rank is not None else "-"
    print(f"rrf={hit.rrf_score:.5f}  b={b!s:>2}  v={v!s:>2}  {hit.child.page_content[:80]!r}")

print("\n=== Parents ===")
parents = retrieve_hybrid_parents(
    store, index, parent_store, question,
    vector_k=settings.retrieval_top_k,
    bm25_k=settings.bm25_top_k,
    fused_k=settings.hybrid_fused_k,
    max_parents=settings.retrieval_max_parents,
    rrf_k=settings.rrf_k,
)
for parent, score in parents:
    print(f"rrf={score:.5f}  {parent.metadata.get('source')}  ({len(parent.page_content)} chars)")

parent_store.close()
import sys

from app.config.settings import settings
from app.embeddings.embedder import get_embedding_model
from app.retrieval.bm25_index import BM25Index, tokenize
from app.utils.logger import setup_logging
from app.vectorstore.chroma_store import get_all_children, get_vector_store, search_children

setup_logging(settings.log_level)
question = " ".join(sys.argv[1:]) or "leave policy"

print("Tokens:", tokenize(question))

model = get_embedding_model(
    settings.embedding_model_name,
    settings.embedding_device,
    settings.embedding_batch_size,
)
store = get_vector_store(model, settings.chroma_persist_dir, settings.chroma_collection_name)

index = BM25Index(get_all_children(store))

print("\n--- BM25 (lafz wala dost) ---")
for chunk_id, doc, score in index.search(question, settings.bm25_top_k)[:5]:
    print(f"score={score:6.2f}  id={chunk_id}  {doc.page_content[:90]!r}")

print("\n--- Vector (matlab wala dost) ---")
for doc, distance in search_children(store, question, settings.retrieval_top_k)[:5]:
    print(f"distance={distance:5.3f}  {doc.page_content[:90]!r}")
import sys

from app.config.settings import settings
from app.embeddings.embedder import get_embedding_model
from app.utils.logger import setup_logging
from app.vectorstore.chroma_store import get_vector_store, search_children

setup_logging(settings.log_level)
question = " ".join(sys.argv[1:]) or "How many days of annual leave do employees get?"

model = get_embedding_model(
    settings.embedding_model_name,
    settings.embedding_device,
    settings.embedding_batch_size,
)
store = get_vector_store(model, settings.chroma_persist_dir, settings.chroma_collection_name)

for doc, distance in search_children(store, question, settings.retrieval_top_k)[:5]:
    print(f"distance={distance:.3f} similarity={1 - distance:.3f}")
    print(f"  {doc.metadata.get('source')} page={doc.metadata.get('page')}")
    print(f"  {doc.page_content[:150]!r}")
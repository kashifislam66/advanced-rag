from app.config.settings import settings
from app.embeddings.embedder import get_embedding_model
from app.utils.logger import setup_logging
from app.vectorstore.chroma_store import count_children, get_vector_store

setup_logging(settings.log_level)
model = get_embedding_model(
    settings.embedding_model_name,
    settings.embedding_device,
    settings.embedding_batch_size,
)
store = get_vector_store(model, settings.chroma_persist_dir, settings.chroma_collection_name)

print("Count:", count_children(store))
print("Collection metadata:", store._collection.metadata)
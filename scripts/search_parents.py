import sys

from app.config.settings import settings
from app.embeddings.embedder import get_embedding_model
from app.retrieval.parent_retriever import retrieve_parents
from app.utils.logger import setup_logging
from app.vectorstore.chroma_store import get_vector_store
from app.vectorstore.parent_store import ParentStore

setup_logging(settings.log_level)
question = " ".join(sys.argv[1:]) or "How many days of annual leave do employees get?"

model = get_embedding_model(
    settings.embedding_model_name,
    settings.embedding_device,
    settings.embedding_batch_size,
)
store = get_vector_store(model, settings.chroma_persist_dir, settings.chroma_collection_name)
parent_store = ParentStore(settings.parent_store_path)

results = retrieve_parents(
    store,
    parent_store,
    question,
    child_k=settings.retrieval_top_k,
    max_parents=settings.retrieval_max_parents,
)

for parent, distance in results:
    print(f"\nbest distance={distance:.3f}  source={parent.metadata.get('source')}")
    print(f"parent length: {len(parent.page_content)} characters")
    print(parent.page_content[:300].replace("\n", " "), "...")

parent_store.close()
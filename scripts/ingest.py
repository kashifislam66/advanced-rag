from app.chunking.parent_child import build_parent_child_chunks
from app.chunking.semantic import build_semantic_parent_child_chunks
from app.config.settings import settings
from app.embeddings.embedder import get_embedding_model
from app.ingestion.loaders import load_documents
from app.utils.logger import setup_logging
from app.vectorstore.parent_store import ParentStore
from app.vectorstore.chroma_store import (
    count_children,
    delete_by_source,
    get_vector_store,
    upsert_children,
)

setup_logging(settings.log_level)

model = get_embedding_model(
    settings.embedding_model_name,
    settings.embedding_device,
    settings.embedding_batch_size,
)
docs = load_documents(settings.raw_data_dir)

if settings.chunking_strategy == "semantic":
    parents, children = build_semantic_parent_child_chunks(
        docs,
        model,
        settings.parent_chunk_size,
        settings.parent_min_size,
        settings.child_chunk_size,
        settings.child_chunk_overlap,
        settings.semantic_breakpoint_percentile,
    )
else:
    parents, children = build_parent_child_chunks(
        docs,
        settings.parent_chunk_size,
        settings.parent_chunk_overlap,
        settings.child_chunk_size,
        settings.child_chunk_overlap,
    )

store = get_vector_store(model, settings.chroma_persist_dir, settings.chroma_collection_name)
parent_store = ParentStore(settings.parent_store_path)

sources = sorted({c.metadata["source"] for c in children})
for source in sources:
    delete_by_source(store, source)
    parent_store.delete_by_source(source)

parent_store.upsert(parents)
upsert_children(store, children, settings.embedding_batch_size)

print(f"Children in Chroma: {count_children(store)}")
print(f"Parents in store:   {parent_store.count()}")
parent_store.close()
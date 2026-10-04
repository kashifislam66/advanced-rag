from app.chunking.parent_child import build_parent_child_chunks
from app.config.settings import settings
from app.ingestion.loaders import load_documents
from app.utils.logger import setup_logging

setup_logging(settings.log_level)

docs = load_documents(settings.raw_data_dir)
parents, children = build_parent_child_chunks(
    docs,
    settings.parent_chunk_size,
    settings.parent_chunk_overlap,
    settings.child_chunk_size,
    settings.child_chunk_overlap,
)

print(f"Docs={len(docs)} Parents={len(parents)} Children={len(children)}")
if children:
    print(children[0].metadata)
    print(children[0].page_content[:300])
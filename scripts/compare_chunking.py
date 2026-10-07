import statistics

from langchain_core.documents import Document

from app.chunking.parent_child import build_parent_child_chunks
from app.chunking.semantic import build_semantic_parent_child_chunks
from app.config.settings import settings
from app.embeddings.embedder import get_embedding_model
from app.ingestion.loaders import load_documents
from app.utils.logger import setup_logging


def describe(name: str, chunks: list[Document]) -> None:
    sizes = [len(c.page_content) for c in chunks]
    print(
        f"{name:<22} count={len(sizes):<5} "
        f"min={min(sizes):<5} median={int(statistics.median(sizes)):<5} max={max(sizes)}"
    )


setup_logging(settings.log_level)
docs = load_documents(settings.raw_data_dir)
model = get_embedding_model(
    settings.embedding_model_name,
    settings.embedding_device,
    settings.embedding_batch_size,
)

rec_parents, rec_children = build_parent_child_chunks(
    docs,
    settings.parent_chunk_size,
    settings.parent_chunk_overlap,
    settings.child_chunk_size,
    settings.child_chunk_overlap,
)
sem_parents, sem_children = build_semantic_parent_child_chunks(
    docs,
    model,
    settings.parent_chunk_size,
    settings.parent_min_size,
    settings.child_chunk_size,
    settings.child_chunk_overlap,
    settings.semantic_breakpoint_percentile,
)

print()
describe("recursive parents", rec_parents)
describe("semantic parents", sem_parents)
describe("recursive children", rec_children)
describe("semantic children", sem_children)
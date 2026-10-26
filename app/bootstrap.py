from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.embeddings.embedder import get_embedding_model
from app.generation.llm import get_llm
from app.reranking.reranker import get_reranker
from app.retrieval.bm25_index import BM25Index
from app.retrieval.pipeline import RetrievalParams, RetrievalResources
from app.vectorstore.chroma_store import get_all_children, get_vector_store
from app.vectorstore.parent_store import ParentStore

if TYPE_CHECKING:
    from app.config.settings import Settings


@dataclass(frozen=True)
class AppContainer:
    """Everything heavy, created once. Close it when finished."""

    embedder: Embeddings
    resources: RetrievalResources
    params: RetrievalParams
    answer_llm: BaseChatModel
    judge_llm: BaseChatModel
    eval_judge_llm: BaseChatModel

    def close(self) -> None:
        self.resources.parent_store.close()


def build_container(settings: "Settings") -> AppContainer:
    embedder = get_embedding_model(
        settings.embedding_model_name, settings.embedding_device, settings.embedding_batch_size
    )
    store = get_vector_store(embedder, settings.chroma_persist_dir, settings.chroma_collection_name)
    resources = RetrievalResources(
        vector_store=store,
        bm25_index=BM25Index(get_all_children(store)),
        parent_store=ParentStore(settings.parent_store_path),
        reranker=get_reranker(
            settings.reranker_model_name, settings.reranker_device, settings.reranker_max_length
        ),
    )

    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    common = (settings.llm_timeout_seconds, settings.llm_max_retries)
    return AppContainer(
        embedder=embedder,
        resources=resources,
        params=RetrievalParams.from_settings(settings),
        answer_llm=get_llm(
            settings.llm_provider, settings.llm_model, api_key,
            settings.llm_temperature, settings.llm_max_tokens, *common,
        ),
        judge_llm=get_llm(
            settings.llm_provider, settings.grader_model or settings.llm_model, api_key,
            0.0, settings.grader_max_tokens, *common,
        ),
        eval_judge_llm=get_llm(
            settings.llm_provider, settings.eval_judge_model or settings.llm_model, api_key,
            0.0, settings.eval_judge_max_tokens, *common,
        ),
    )
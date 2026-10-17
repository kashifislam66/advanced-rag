import logging
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm(
    provider: str,
    model: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> BaseChatModel:
    """Create the chat model once per process, with provider-independent settings."""
    logger.info("Creating LLM: provider=%s model=%s", provider, model)
    kwargs: dict = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if api_key:
        kwargs["api_key"] = api_key
    
    if provider == "openai" and "openrouter" in model.lower():
        kwargs["base_url"] = "https://openrouter.ai/api/v1"
        
    return init_chat_model(model, model_provider=provider, **kwargs)
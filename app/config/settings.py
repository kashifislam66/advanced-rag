from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    raw_data_dir: Path = Path("data/raw")
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50
    semantic_breakpoint_percentile: float = 95.0
    parent_min_size: int = 200
    log_level: str = "INFO"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        if self.parent_chunk_overlap >= self.parent_chunk_size:
            raise ValueError("parent_chunk_overlap must be smaller than parent_chunk_size")
        if self.child_chunk_overlap >= self.child_chunk_size:
            raise ValueError("child_chunk_overlap must be smaller than child_chunk_size")
        if self.child_chunk_size >= self.parent_chunk_size:
            raise ValueError("child_chunk_size must be smaller than parent_chunk_size")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")
        if not 0 < self.semantic_breakpoint_percentile < 100:
            raise ValueError("semantic_breakpoint_percentile must be between 0 and 100")
        if self.parent_min_size >= self.parent_chunk_size:
            raise ValueError("parent_min_size must be smaller than parent_chunk_size")
        return self


settings = Settings()
"""Centralized application configuration.

All environment-driven settings are declared here using pydantic-settings so
that the rest of the codebase never reads `os.environ` directly. This keeps
secret handling auditable in a single place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables / `.env`."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider
    llm_provider: str = Field(default="demo", description="anthropic|openai|ollama|demo")
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Paths
    data_dir: str = "data"
    models_dir: str = "models"
    knowledge_base_dir: str = "knowledge_base"
    vector_store_dir: str = "data/processed/chroma_db"

    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"

    @property
    def data_path(self) -> Path:
        return PROJECT_ROOT / self.data_dir

    @property
    def models_path(self) -> Path:
        return PROJECT_ROOT / self.models_dir

    @property
    def knowledge_base_path(self) -> Path:
        return PROJECT_ROOT / self.knowledge_base_dir

    @property
    def vector_store_path(self) -> Path:
        return PROJECT_ROOT / self.vector_store_dir

    def redacted(self) -> dict[str, str]:
        """Return settings safe to log (secrets masked)."""

        def mask(value: str | None) -> str:
            if not value:
                return "<not set>"
            return f"{value[:4]}{'*' * 8}"

        return {
            "llm_provider": self.llm_provider,
            "anthropic_api_key": mask(self.anthropic_api_key),
            "openai_api_key": mask(self.openai_api_key),
            "app_env": self.app_env,
        }


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()

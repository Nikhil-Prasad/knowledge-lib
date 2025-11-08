from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel


class IngestDefaults(BaseModel):
    """Default tunables for text ingestion/segmentation."""
    sentence_window_k: int = 3
    sentence_window_overlap: int = 1
    short_text_threshold_chars: int = 240
    window_soft_max_chars: int = 1200


class Settings(BaseSettings):
    """Application settings loaded from environment/.env.

    Maps common env vars used in this project and provides a computed
    SQLAlchemy URL fallback when `DATABASE_URL` is not set.
    """

    # Providers / API keys
    openai_api_key: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key")
    )

    # SQLAlchemy DSN (preferred)
    database_url: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("DATABASE_URL", "database_url")
    )

    # Discrete Postgres envs (helpful for tooling and DSN construction)
    pg_host: Optional[str] = Field(default=None, validation_alias=AliasChoices("PGHOST", "pg_host"))
    pg_port: Optional[int] = Field(default=None, validation_alias=AliasChoices("PGPORT", "pg_port"))
    pg_user: Optional[str] = Field(default=None, validation_alias=AliasChoices("PGUSER", "pg_user"))
    pg_password: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("PGPASSWORD", "pg_password")
    )
    pg_database: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("PGDATABASE", "pg_database")
    )

    # Docker Compose variables (not required at runtime but kept for completeness)
    postgres_user: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("POSTGRES_USER", "postgres_user")
    )
    postgres_password: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("POSTGRES_PASSWORD", "postgres_password")
    )
    postgres_db: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("POSTGRES_DB", "postgres_db")
    )
    postgres_port: Optional[int] = Field(
        default=None, validation_alias=AliasChoices("POSTGRES_PORT", "postgres_port")
    )

    # Ingest defaults (override via env if needed)
    ingest_sentence_window_k: int = Field(
        default=4, validation_alias=AliasChoices("INGEST_SENTENCE_WINDOW_K", "ingest_sentence_window_k")
    )
    ingest_sentence_window_overlap: int = Field(
        default=1,
        validation_alias=AliasChoices(
            "INGEST_SENTENCE_WINDOW_OVERLAP", "ingest_sentence_window_overlap"
        ),
    )
    ingest_short_text_threshold_chars: int = Field(
        default=240,
        validation_alias=AliasChoices(
            "INGEST_SHORT_TEXT_THRESHOLD_CHARS", "ingest_short_text_threshold_chars"
        ),
    )
    ingest_window_soft_max_chars: int = Field(
        default=1200,
        validation_alias=AliasChoices(
            "INGEST_WINDOW_SOFT_MAX_CHARS", "ingest_window_soft_max_chars"
        ),
    )

    # Embedding configuration
    ingest_embed_on_ingest: bool = Field(
        default=True,
        validation_alias=AliasChoices("INGEST_EMBED_ON_INGEST", "ingest_embed_on_ingest"),
    )
    ingest_embed_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("INGEST_EMBED_MODEL", "ingest_embed_model"),
    )
    ingest_embed_version: str = Field(
        default="v1",
        validation_alias=AliasChoices("INGEST_EMBED_VERSION", "ingest_embed_version"),
    )

    # pydantic-settings configuration
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @property
    def sqlalchemy_url(self) -> str:
        """Effective SQLAlchemy URL.

        Priority:
        1) DATABASE_URL
        2) Construct from PG* variables
        3) Fallback to local dev default
        """
        if self.database_url:
            return self.database_url
        if all([self.pg_host, self.pg_user, self.pg_password, self.pg_database]):
            host = self.pg_host
            port = self.pg_port or 5432
            return (
                f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}@{host}:{port}/{self.pg_database}"
            )
        # Default local dev DSN (matches db/session default)
        return "postgresql+psycopg2://kl:klpass@localhost:5432/knowledge"

    @property
    def ingest_defaults(self) -> IngestDefaults:
        return IngestDefaults(
            sentence_window_k=self.ingest_sentence_window_k,
            sentence_window_overlap=self.ingest_sentence_window_overlap,
            short_text_threshold_chars=self.ingest_short_text_threshold_chars,
            window_soft_max_chars=self.ingest_window_soft_max_chars,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

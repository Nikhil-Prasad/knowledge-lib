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
    # Optional async SQLAlchemy URL for app runtime (e.g., postgresql+asyncpg://...)
    async_database_url: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("ASYNC_DATABASE_URL", "async_database_url")
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
    ingest_embed_batch_size: int = Field(
        default=256,
        validation_alias=AliasChoices("INGEST_EMBED_BATCH_SIZE", "ingest_embed_batch_size"),
    )
    ingest_embed_concurrency: int = Field(
        default=2,
        validation_alias=AliasChoices("INGEST_EMBED_CONCURRENCY", "ingest_embed_concurrency"),
    )

    # Reranker configuration (scaffold)
    rerank_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("RERANK_ENABLED", "rerank_enabled"),
    )
    rerank_model: str = Field(
        default="BAAI/bge-reranker-base",
        validation_alias=AliasChoices("RERANK_MODEL", "rerank_model"),
    )
    rerank_max_length: int = Field(
        default=384,
        validation_alias=AliasChoices("RERANK_MAX_LENGTH", "rerank_max_length"),
    )
    rerank_batch_size: int = Field(
        default=32,
        validation_alias=AliasChoices("RERANK_BATCH_SIZE", "rerank_batch_size"),
    )
    rerank_device: str = Field(
        default="cpu",
        validation_alias=AliasChoices("RERANK_DEVICE", "rerank_device"),
    )

    # Hybrid search tunables
    hybrid_n_sem: int = Field(
        default=400,
        validation_alias=AliasChoices("HYBRID_N_SEM", "hybrid_n_sem"),
    )
    hybrid_n_lex: int = Field(
        default=200,
        validation_alias=AliasChoices("HYBRID_N_LEX", "hybrid_n_lex"),
    )
    hybrid_rerank_pool: int = Field(
        default=256,
        validation_alias=AliasChoices("HYBRID_RERANK_POOL", "hybrid_rerank_pool"),
    )
    hybrid_ann_ef_search: int = Field(
        default=96,
        validation_alias=AliasChoices("HYBRID_ANN_EF_SEARCH", "hybrid_ann_ef_search"),
    )
    hybrid_per_container_limit: int = Field(
        default=1,
        validation_alias=AliasChoices("HYBRID_PER_CONTAINER_LIMIT", "hybrid_per_container_limit"),
    )

    # PDF ingestion settings
    pdf_layout_provider: str = Field(
        default="hf",  # options: noop | hf (DocLayNet detector)
        validation_alias=AliasChoices("PDF_LAYOUT_PROVIDER", "pdf_layout_provider"),
    )
    pdf_ocr_provider: str = Field(
        default="got",  # options: noop | hf | deepseek | got
        validation_alias=AliasChoices("PDF_OCR_PROVIDER", "pdf_ocr_provider"),
    )
    # Optional model overrides (HF repo IDs)
    pdf_layout_model: str | None = Field(
        default="cmarkea/detr-layout-detection",
        validation_alias=AliasChoices("PDF_LAYOUT_MODEL", "pdf_layout_model"),
    )
    pdf_ocr_model: str | None = Field(
        default="stepfun-ai/GOT-OCR-2.0-hf",
        validation_alias=AliasChoices("PDF_OCR_MODEL", "pdf_ocr_model"),
    )
    pdf_table_det_model: str | None = Field(
        default="microsoft/table-transformer-detection",
        validation_alias=AliasChoices("PDF_TABLE_DET_MODEL", "pdf_table_det_model"),
    )
    pdf_table_struct_model: str | None = Field(
        default="microsoft/table-transformer-structure-recognition",
        validation_alias=AliasChoices("PDF_TABLE_STRUCT_MODEL", "pdf_table_struct_model"),
    )

    # Artifacts storage (local file URIs for now)
    artifacts_base_dir: str = Field(
        default="data/artifacts",
        validation_alias=AliasChoices("ARTIFACTS_BASE_DIR", "artifacts_base_dir"),
    )
    artifacts_image_format: str = Field(
        default="webp",  # webp|png|jpg
        validation_alias=AliasChoices("ARTIFACTS_IMAGE_FORMAT", "artifacts_image_format"),
    )
    artifacts_image_quality: int = Field(
        default=85,
        validation_alias=AliasChoices("ARTIFACTS_IMAGE_QUALITY", "artifacts_image_quality"),
    )
    pdf_render_dpi: int = Field(
        default=200,
        validation_alias=AliasChoices("PDF_RENDER_DPI", "pdf_render_dpi"),
    )
    pdf_max_pages: int = Field(
        default=200,
        validation_alias=AliasChoices("PDF_MAX_PAGES", "pdf_max_pages"),
    )

    # Page routing thresholds (0..1 floats)
    route_text_coverage_high: float = Field(
        default=0.2,
        validation_alias=AliasChoices("ROUTE_TEXT_COVERAGE_HIGH", "route_text_coverage_high"),
    )
    route_text_coverage_low: float = Field(
        default=0.02,
        validation_alias=AliasChoices("ROUTE_TEXT_COVERAGE_LOW", "route_text_coverage_low"),
    )
    route_image_coverage_high: float = Field(
        default=0.6,
        validation_alias=AliasChoices("ROUTE_IMAGE_COVERAGE_HIGH", "route_image_coverage_high"),
    )
    route_sandwich_threshold: float = Field(
        default=0.5,
        validation_alias=AliasChoices("ROUTE_SANDWICH_THRESHOLD", "route_sandwich_threshold"),
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
    def async_sqlalchemy_url(self) -> str:
        """Effective async SQLAlchemy URL for app runtime.

        Priority:
        1) ASYNC_DATABASE_URL (if provided)
        2) Derive from `sqlalchemy_url` by switching to asyncpg
        """
        if self.async_database_url:
            return self.async_database_url
        url = self.sqlalchemy_url
        # Already async
        if url.startswith("postgresql+asyncpg://"):
            return url
        # Transform common sync forms
        if url.startswith("postgresql+psycopg2://"):
            return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        # Fallback (return as-is)
        return url

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

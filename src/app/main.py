from __future__ import annotations

from fastapi import FastAPI

from src.app.api.health.health import router as health_router
from src.app.api.v1.ingest.ingest import router as ingest_router
from src.app.api.v1.search.search import router as search_router


def create_app() -> FastAPI:
    app = FastAPI(title="knowledge-lib", version="0.1.0")
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(search_router)
    return app


app = create_app()

# If you prefer running directly: `uv run uvicorn src.app.main:app --reload`

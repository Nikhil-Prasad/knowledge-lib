from __future__ import annotations

"""Ingest service package.

Provides a stable entrypoint `ingest(session, req)` that routes sources
to the appropriate pipeline (text vs container) after a resolve/identify
phase. The API layer should import only this function.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from src.app.api.v1.ingest.schemas import IngestRequest, IngestResponse
from . import orchestrator


async def ingest(session: AsyncSession, req: IngestRequest) -> IngestResponse:
    """Route an `IngestRequest` to the appropriate pipeline.

    High level steps (implemented in `orchestrator.ingest`):
    - Resolve the source (download/read/parse data_uri; compute sha256; derive source_uri).
    - Identify mime type (use hint → headers/extension).
    - Choose a pipeline (text vs container), optionally honoring `modality_hint`.
    - Run the selected pipeline which persists a container, a page list, and segments.
    - Return counts suitable for the API response.

    This function is a thin delegator to keep the route import stable.
    """
    return await orchestrator.ingest(session, req)

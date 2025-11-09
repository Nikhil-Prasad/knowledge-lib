from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.db.session_async import get_async_db
from src.app.services.ingest import ingest as ingest_service
from src.app.services.embeddings.embed_runner_async import embed_container_segments_async
from src.app.api.v1.ingest.schemas import IngestRequest, IngestResponse


router = APIRouter(prefix="/v1", tags=["v1"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_async_db)) -> IngestResponse:
    """Delegate ingest to the services router for resolve/identify/pipeline routing."""
    try:
        resp = await ingest_service(db, req)
        background_tasks.add_task(embed_container_segments_async, resp.container_id)
        return resp
    except NotImplementedError as e:
        # Until router/pipelines are fully implemented, surface a 501 for unhandled cases
        raise HTTPException(status_code=501, detail=str(e))

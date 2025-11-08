from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from src.app.api.deps import get_db
from src.app.services.ingest import ingest as ingest_service
from src.app.services.embeddings.embed_runner import embed_container_segments
from src.app.api.v1.ingest.schemas import IngestRequest, IngestResponse


router = APIRouter(prefix="/v1", tags=["v1"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> IngestResponse:
    """Delegate ingest to the services router for resolve/identify/pipeline routing."""
    try:
        resp = ingest_service(db, req)
        background_tasks.add_task(embed_container_segments, resp.container_id)
        return resp
    except NotImplementedError as e:
        # Until router/pipelines are fully implemented, surface a 501 for unhandled cases
        raise HTTPException(status_code=501, detail=str(e))

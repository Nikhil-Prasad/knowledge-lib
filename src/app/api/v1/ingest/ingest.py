from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.app.api.deps import get_db
from src.app.services.ingest_service import ingest_text_segment
from src.app.api.v1.ingest.schemas import IngestRequest, IngestResponse, RawTextSource


router = APIRouter(prefix="/v1", tags=["v1"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, db: Session = Depends(get_db)) -> IngestResponse:
    """Auto-routing ingest endpoint using unified request schema.

    For now, only raw_text is implemented; other source types will be wired later.
    """
    src = req.source
    if isinstance(src, RawTextSource):
        return ingest_text_segment(db, src)
    raise HTTPException(status_code=501, detail="Ingest source type not implemented yet")

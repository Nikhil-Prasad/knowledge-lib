from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.db.session.session_async import get_async_db
from src.app.services.ingest import ingest as ingest_service
from src.app.services.embeddings.embed_runner_async import embed_container_segments_async
from src.app.services.ingest.container_pipeline import process_pdf_container_async
from src.app.settings import get_settings
from src.app.api.v1.ingest.schemas import IngestRequest, IngestResponse


router = APIRouter(prefix="/v1", tags=["v1"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_async_db)) -> IngestResponse:
    """Delegate ingest to the services router for resolve/identify/pipeline routing."""
    try:
        resp = await ingest_service(db, req)
        # If the source is a PDF upload_ref, schedule the PDF processing job instead of immediate embeddings.
        is_pdf = False
        try:
            src = req.source
            # Narrow only to UploadRefSource to avoid routing changes for other types
            from src.app.api.v1.ingest.schemas import UploadRefSource
            if isinstance(src, UploadRefSource):
                uri = str(src.upload_uri)
                cth = (src.content_type_hint or "").lower()
                if uri.lower().endswith(".pdf") or cth == "application/pdf":
                    # Resolve file path for the background job
                    if uri.startswith("file://"):
                        from urllib.parse import urlparse, unquote
                        p = urlparse(uri)
                        abs_path = unquote(p.path)
                    else:
                        from pathlib import Path
                        abs_path = str(Path(uri).expanduser().resolve())
                    background_tasks.add_task(process_pdf_container_async, container_id=resp.container_id, pdf_path=abs_path)
                    is_pdf = True
        except Exception:
            # If detection fails, fall back to default behavior
            is_pdf = False

        # Honor settings flag to enqueue embeddings on ingest (skip for PDFs; job embeds later)
        settings = get_settings()
        if settings.ingest_embed_on_ingest and not is_pdf:
            background_tasks.add_task(embed_container_segments_async, resp.container_id)
        return resp
    except NotImplementedError as e:
        # Until router/pipelines are fully implemented, surface a 501 for unhandled cases
        raise HTTPException(status_code=501, detail=str(e))

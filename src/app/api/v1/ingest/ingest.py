from __future__ import annotations

import hashlib
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.app.api.deps import get_db
from src.app.db.models.models import Document, Page, Chunk
from src.app.schemas.ingest import IngestTextRequest


router = APIRouter(prefix="/v1", tags=["v1"])


class IngestTextResponse(BaseModel):
    doc_id: uuid.UUID
    pages_created: int
    chunks_created: int


@router.post("/ingest/text", response_model=IngestTextResponse)
def ingest_text(payload: IngestTextRequest, db: Session = Depends(get_db)) -> IngestTextResponse:
    sha = hashlib.sha256(payload.text.encode("utf-8")).digest()
    # Pre-generate doc_id so related rows can reference it before flush
    doc_id = uuid.uuid4()
    doc = Document(
        doc_id=doc_id,
        source_uri="inline:text",
        mime_type="text/plain",
        sha256=sha,
        title=payload.title or None,
    )
    db.add(doc)

    page = Page(doc_id=doc_id, page_no=1, text=payload.text, image_uri=None)
    db.add(page)

    chunk = Chunk(
        doc_id=doc_id,
        page_no=1,
        object_type="text",
        section_path=None,
        bbox=None,
        text=payload.text,
        emb_v1=None,
    )
    db.add(chunk)

    db.commit()
    return IngestTextResponse(doc_id=doc_id, pages_created=1, chunks_created=1)


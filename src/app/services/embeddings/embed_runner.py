from __future__ import annotations

from typing import List
from uuid import UUID

from sqlalchemy.orm import Session

from src.app.db.session import SessionLocal
from src.app.db.models.models import TextSegment
from src.app.settings import get_settings
from src.app.services.embeddings.oai_embeddings import embed_many
import asyncio


def embed_container_segments(container_id: UUID) -> None:
    """Background task: embed all text segments for a container that lack embeddings.

    Note: Uses a synchronous SQLAlchemy session inside an async function. This is
    acceptable for BackgroundTasks in our current setup, but consider switching
    to an async DB session or a separate worker for heavy loads.
    """
    settings = get_settings()
    model = settings.ingest_embed_model

    with SessionLocal() as session:  
        segs: List[TextSegment] = (
            session.query(TextSegment)
            .filter(TextSegment.container_id == container_id, TextSegment.emb_v1 == None)
            .order_by(TextSegment.segment_id)
            .all()
        )
        if not segs:
            return

        BATCH = 256
        for i in range(0, len(segs), BATCH):
            chunk = segs[i : i + BATCH]
            texts = [s.text for s in chunk]
            vectors = asyncio.run(embed_many(texts, model=model))
            for s, vec in zip(chunk, vectors):
                s.emb_v1 = vec
                s.emb_model = model
                s.emb_version = settings.ingest_embed_version
            session.commit()

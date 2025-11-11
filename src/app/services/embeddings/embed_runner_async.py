from __future__ import annotations

import asyncio
from typing import List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.db.session.session_async import AsyncSessionLocal
from src.app.db.models.models import TextSegment
from src.app.settings import get_settings
from src.app.services.embeddings.oai_embeddings import embed_many


async def embed_container_segments_async(container_id: UUID) -> None:
    """Async embedding runner with bounded worker pool (per-container job).

    Batches segments and processes them with N workers. Each worker embeds one
    chunk then writes with its own short-lived AsyncSession (commit per chunk).
    Selection is idempotent: only rows with emb_v1 IS NULL are processed.
    """
    settings = get_settings()
    model = settings.ingest_embed_model
    batch_size = settings.ingest_embed_batch_size
    concurrency = max(1, settings.ingest_embed_concurrency)

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TextSegment.segment_id, TextSegment.text)
            .where(TextSegment.container_id == container_id, TextSegment.emb_v1 == None)  # noqa: E711
            .order_by(TextSegment.segment_id)
        )
        rows = list(result.all())
    if not rows:
        return

    # Prepare batches of (segment_id, text)
    batches: List[List[tuple[UUID, str]]] = [
        rows[i : i + batch_size] for i in range(0, len(rows), batch_size)
    ]

    q: asyncio.Queue[List[tuple[UUID, str]]] = asyncio.Queue(maxsize=max(1, concurrency * 4))
    for b in batches:
        await q.put(b)
    for _ in range(concurrency):
        await q.put([])  # sentinel

    async def worker() -> None:
        while True:
            chunk = await q.get()
            try:
                if not chunk:
                    return
                texts = [t for _, t in chunk]
                vectors = await embed_many(texts, model=model)
                async with AsyncSessionLocal() as write_sess:
                    for (seg_id, _), vec in zip(chunk, vectors):
                        seg = await write_sess.get(TextSegment, seg_id)
                        if seg is None:
                            continue
                        if seg.emb_v1 is None:  # idempotent guard
                            seg.emb_v1 = vec
                            seg.emb_model = model
                            seg.emb_version = settings.ingest_embed_version
                    await write_sess.commit()
            finally:
                q.task_done()

    async with asyncio.TaskGroup() as tg:
        for _ in range(concurrency):
            tg.create_task(worker())
        await q.join()

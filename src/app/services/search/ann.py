from __future__ import annotations

from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import text as sql_text, bindparam
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.services.embeddings.oai_embeddings import embed_one
from src.app.settings import get_settings
from pgvector.sqlalchemy import Vector


async def ann_search(
    db: AsyncSession,
    *,
    query: str,
    k: int,
    collection_id: Optional[UUID] = None,
    ef_search: Optional[int] = None,
) -> List[Dict[str, Any]]:
    q = query.strip()
    if not q:
        return []

    settings = get_settings()
    qvec = await embed_one(q, model=settings.ingest_embed_model)

    # Optionally set HNSW ef_search for this transaction
    if ef_search is not None:
        try:
            # Parameter binding is not supported in SET; interpolate trusted int
            await db.execute(sql_text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
        except Exception:
            # Clear aborted transaction state and continue without ef_search
            try:
                await db.rollback()
            except Exception:
                pass

    if collection_id is not None:
        sql = sql_text(
            """
            SELECT ts.container_id, ts.page_no, ts.segment_id,
                   1 - (ts.emb_v1 <=> :qvec) AS score,
                   left(ts.text, 200) AS snippet,
                   left(ts.text, 1500) AS text
            FROM text_segments ts
            JOIN containers_collections cc ON cc.container_id = ts.container_id
            WHERE cc.collection_id = :collection_id AND ts.emb_v1 IS NOT NULL
            ORDER BY ts.emb_v1 <=> :qvec
            LIMIT :k
            """
        ).bindparams(bindparam("qvec", type_=Vector(1536)))
        params = {"qvec": qvec, "k": k, "collection_id": collection_id}
    else:
        sql = sql_text(
            """
            SELECT ts.container_id, ts.page_no, ts.segment_id,
                   1 - (ts.emb_v1 <=> :qvec) AS score,
                   left(ts.text, 200) AS snippet,
                   left(ts.text, 1500) AS text
            FROM text_segments ts
            WHERE ts.emb_v1 IS NOT NULL
            ORDER BY ts.emb_v1 <=> :qvec
            LIMIT :k
            """
        ).bindparams(bindparam("qvec", type_=Vector(1536)))
        params = {"qvec": qvec, "k": k}

    result = await db.execute(sql, params)
    rows = result.mappings().all()
    return [
        {
            "modality": "text",
            "segment_id": row["segment_id"],
            "container_id": row["container_id"],
            "page_no": row["page_no"],
            "score": float(row["score"] or 0.0),
            "snippet": row["snippet"],
        }
        for row in rows
    ]

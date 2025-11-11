from __future__ import annotations

from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession


async def fts_search(
    db: AsyncSession,
    *,
    query: str,
    k: int,
    collection_id: Optional[UUID] = None,
) -> List[Dict[str, Any]]:
    q = query.strip()
    if not q:
        return []

    if collection_id is not None:
        sql = sql_text(
            """
            SELECT ts.container_id, ts.page_no, ts.segment_id,
                   ts_rank_cd(to_tsvector('english', ts.text), plainto_tsquery('english', :q)) AS score,
                   left(ts.text, 200) AS snippet
            FROM text_segments ts
            JOIN containers_collections cc ON cc.container_id = ts.container_id
            WHERE cc.collection_id = :collection_id
              AND to_tsvector('english', ts.text) @@ plainto_tsquery('english', :q)
            ORDER BY score DESC
            LIMIT :k
            """
        )
        params = {"q": q, "k": k, "collection_id": collection_id}
    else:
        sql = sql_text(
            """
            SELECT ts.container_id, ts.page_no, ts.segment_id,
                   ts_rank_cd(to_tsvector('english', ts.text), plainto_tsquery('english', :q)) AS score,
                   left(ts.text, 200) AS snippet
            FROM text_segments ts
            WHERE to_tsvector('english', ts.text) @@ plainto_tsquery('english', :q)
            ORDER BY score DESC
            LIMIT :k
            """
        )
        params = {"q": q, "k": k}

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


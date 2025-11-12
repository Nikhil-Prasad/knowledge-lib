from __future__ import annotations

from typing import List, Optional, Dict, Any
from uuid import UUID
import re

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession


def _build_or_prefix_tsquery(query: str, max_terms: int = 3) -> Optional[str]:
    # Extract simple word tokens, drop very short and numeric-only ones, dedupe, and build OR with prefix
    tokens = re.findall(r"[A-Za-z0-9_]+", query.lower())
    tokens = [t for t in tokens if len(t) >= 5 and not t.isdigit()]
    if not tokens:
        return None
    # Deduplicate preserving order
    seen = set()
    uniq: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    uniq = uniq[:max_terms]
    if not uniq:
        return None
    # Add prefix wildcard
    return " | ".join(f"{t}:*" for t in uniq)


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

    # Primary: flexible websearch query (Google-like)
    if collection_id is not None:
        sql_primary = sql_text(
            """
            WITH tsq AS (
              SELECT websearch_to_tsquery('pg_catalog.english', unaccent(:q)) AS q
            )
            SELECT ts.container_id, ts.page_no, ts.segment_id,
                   ts_rank_cd(ts.text_fts, (SELECT q FROM tsq)) AS score,
                   left(ts.text, 200) AS snippet
            FROM text_segments ts
            JOIN containers_collections cc ON cc.container_id = ts.container_id
            WHERE cc.collection_id = :collection_id
              AND ts.text_fts @@ (SELECT q FROM tsq)
            ORDER BY score DESC
            LIMIT :k
            """
        )
        params = {"q": q, "k": k, "collection_id": collection_id}
    else:
        sql_primary = sql_text(
            """
            WITH tsq AS (
              SELECT websearch_to_tsquery('pg_catalog.english', unaccent(:q)) AS q
            )
            SELECT ts.container_id, ts.page_no, ts.segment_id,
                   ts_rank_cd(ts.text_fts, (SELECT q FROM tsq)) AS score,
                   left(ts.text, 200) AS snippet
            FROM text_segments ts
            WHERE ts.text_fts @@ (SELECT q FROM tsq)
            ORDER BY score DESC
            LIMIT :k
            """
        )
        params = {"q": q, "k": k}

    result = await db.execute(sql_primary, params)
    rows = result.mappings().all()
    if rows:
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

    # Fallback: OR-of-top tokens with prefix matching
    or_query = _build_or_prefix_tsquery(q)
    if not or_query:
        return []

    if collection_id is not None:
        sql_fallback = sql_text(
            """
            SELECT ts.container_id, ts.page_no, ts.segment_id,
                   ts_rank_cd(ts.text_fts, to_tsquery('pg_catalog.english', :orq)) AS score,
                   left(ts.text, 200) AS snippet
            FROM text_segments ts
            JOIN containers_collections cc ON cc.container_id = ts.container_id
            WHERE cc.collection_id = :collection_id
              AND ts.text_fts @@ to_tsquery('pg_catalog.english', :orq)
              AND ts_rank_cd(ts.text_fts, to_tsquery('pg_catalog.english', :orq)) >= :min_score
            ORDER BY score DESC
            LIMIT :k
            """
        )
        params_fb = {"orq": or_query, "k": k, "collection_id": collection_id, "min_score": 0.10}
    else:
        sql_fallback = sql_text(
            """
            SELECT ts.container_id, ts.page_no, ts.segment_id,
                   ts_rank_cd(ts.text_fts, to_tsquery('pg_catalog.english', :orq)) AS score,
                   left(ts.text, 200) AS snippet
            FROM text_segments ts
            WHERE ts.text_fts @@ to_tsquery('pg_catalog.english', :orq)
              AND ts_rank_cd(ts.text_fts, to_tsquery('pg_catalog.english', :orq)) >= :min_score
            ORDER BY score DESC
            LIMIT :k
            """
        )
        params_fb = {"orq": or_query, "k": k, "min_score": 0.10}

    result_fb = await db.execute(sql_fallback, params_fb)
    rows_fb = result_fb.mappings().all()
    return [
        {
            "modality": "text",
            "segment_id": row["segment_id"],
            "container_id": row["container_id"],
            "page_no": row["page_no"],
            "score": float(row["score"] or 0.0),
            "snippet": row["snippet"],
        }
        for row in rows_fb
    ]

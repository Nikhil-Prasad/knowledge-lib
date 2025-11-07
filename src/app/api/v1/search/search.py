from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from src.app.api.deps import get_db
from src.app.api.v1.search.schemas import SearchRequest, SearchResponse, SearchHit


router = APIRouter(prefix="/v1", tags=["v1"])


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    q = req.query.strip()
    if not q:
        return SearchResponse(results=[])

    sql = sql_text(
        """
        SELECT c.container_id, c.page_no, c.segment_id,
               ts_rank_cd(c.text_fts, plainto_tsquery('simple', :q)) AS score,
               left(c.text, 200) AS snippet
        FROM text_segments c
        WHERE c.text_fts @@ plainto_tsquery('simple', :q)
        ORDER BY score DESC
        LIMIT :k
        """
    )
    rows = db.execute(sql, {"q": q, "k": req.k}).mappings().all()

    hits: List[SearchHit] = [
        SearchHit(
            modality="text",
            segment_id=row["segment_id"],
            container_id=row["container_id"],
            page_no=row["page_no"],
            score=float(row["score"] or 0.0),
            snippet=row["snippet"],
        )
        for row in rows
    ]
    return SearchResponse(results=hits)

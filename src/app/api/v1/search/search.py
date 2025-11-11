from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.db.session.session_async import get_async_db
from src.app.api.v1.search.schemas import (
    SearchResponse,
    SearchHit,
    FtsSearchRequest,
    AnnSearchRequest,
    HybridSearchRequest,
)
from src.app.services.search.fts import fts_search
from src.app.services.search.ann import ann_search
from src.app.services.search.hybrid import hybrid_search


router = APIRouter(prefix="/v1", tags=["v1"])


@router.post("/search/fts", response_model=SearchResponse)
async def search_fts(req: FtsSearchRequest, db: AsyncSession = Depends(get_async_db)) -> SearchResponse:
    rows = await fts_search(db, query=req.query, k=req.k, collection_id=req.collection_id)
    hits: List[SearchHit] = [
        SearchHit(
            modality=row.get("modality", "text"),
            segment_id=row["segment_id"],
            container_id=row["container_id"],
            page_no=row.get("page_no"),
            score=row.get("score", 0.0),
            snippet=row.get("snippet"),
        )
        for row in rows
    ]
    return SearchResponse(results=hits)


@router.post("/search/ann", response_model=SearchResponse)
async def search_ann(req: AnnSearchRequest, db: AsyncSession = Depends(get_async_db)) -> SearchResponse:
    try:
        rows = await ann_search(db, query=req.query, k=req.k, collection_id=req.collection_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ANN failed: {e}")

    hits: List[SearchHit] = [
        SearchHit(
            modality=row.get("modality", "text"),
            segment_id=row["segment_id"],
            container_id=row["container_id"],
            page_no=row.get("page_no"),
            score=row.get("score", 0.0),
            snippet=row.get("snippet"),
        )
        for row in rows
    ]
    return SearchResponse(results=hits)


@router.post("/search/hybrid", response_model=SearchResponse)
async def search_hybrid(req: HybridSearchRequest, db: AsyncSession = Depends(get_async_db)) -> SearchResponse:
    try:
        rows = await hybrid_search(db, query=req.query, k=req.k)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="Hybrid search not implemented yet")
    hits: List[SearchHit] = [
        SearchHit(
            modality=row.get("modality", "text"),
            segment_id=row["segment_id"],
            container_id=row["container_id"],
            page_no=row.get("page_no"),
            score=row.get("score", 0.0),
            snippet=row.get("snippet"),
        )
        for row in rows
    ]
    return SearchResponse(results=hits)

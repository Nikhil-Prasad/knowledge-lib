from __future__ import annotations

from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from .fts import fts_search
from .ann import ann_search
from .utils import rrf_fuse


async def hybrid_search(
    db: AsyncSession,
    *,
    query: str,
    k: int,
    n_lex: int = 200,
    n_sem: int = 200,
    rrf_k: int = 60,
    collection_id: Optional[UUID] = None,
    use_reranker: bool = False,
    use_mmr: bool = False,
) -> List[Dict[str, Any]]:
    # Gather candidates from each leg
    fts_hits = await fts_search(db, query=query, k=n_lex, collection_id=collection_id)
    ann_hits = await ann_search(db, query=query, k=n_sem, collection_id=collection_id)

    # Fuse via Reciprocal Rank Fusion
    fused = rrf_fuse([fts_hits, ann_hits], k=rrf_k)

    # Take top-M (here directly slice to requested k after fusion; reranker/MMR could be applied here later)
    top = fused[:k]

    results: List[Dict[str, Any]] = []
    for hit, fused_score in top:
        # Preserve fields and replace score with fused score (float)
        results.append({
            "modality": hit.get("modality", "text"),
            "segment_id": hit.get("segment_id"),
            "container_id": hit.get("container_id"),
            "page_no": hit.get("page_no"),
            "score": float(fused_score),
            "snippet": hit.get("snippet"),
        })

    return results

from __future__ import annotations

from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

from .fts import fts_search
from .ann import ann_search
from .utils import rrf_fuse

# Optional reranker import (scaffold)
try:  # pragma: no cover - optional
    from src.app.services.rerank.bge_reranker import BgeReranker, RerankConfig
    from src.app.services.rerank.text_builders import build_rerank_text
except Exception:  # pragma: no cover - optional
    BgeReranker = None  # type: ignore
    RerankConfig = None  # type: ignore
    def build_rerank_text(candidate):  # type: ignore
        return str(candidate.get("snippet") or candidate.get("text") or "")


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
    rerank_pool: int = 256,
    per_container_limit: int = 1,
    ann_ef_search: Optional[int] = None,
) -> List[Dict[str, Any]]:
    # Gather candidates from each leg
    fts_hits = await fts_search(db, query=query, k=n_lex, collection_id=collection_id)
    ann_hits = await ann_search(db, query=query, k=n_sem, collection_id=collection_id, ef_search=ann_ef_search)

    # Fuse via Reciprocal Rank Fusion
    fused = rrf_fuse([fts_hits, ann_hits], k=rrf_k)

    # Container-level dedup to diversify (keep up to per_container_limit hits per container)
    if per_container_limit and per_container_limit > 0:
        counts: Dict[Any, int] = {}
        deduped: List[tuple[Dict[str, Any], float]] = []
        for hit, score in fused:
            cid = hit.get("container_id")
            c = counts.get(cid, 0)
            if c < per_container_limit:
                deduped.append((hit, score))
                counts[cid] = c + 1
        fused = deduped

    # Optionally rerank top-M via BGE cross-encoder (scaffold only; requires sentence-transformers)
    top_m = min(len(fused), max(k, rerank_pool))
    top = fused[:top_m]
    if use_reranker and BgeReranker is not None:
        # Prepare candidates with text for reranking
        cand_hits = [h for (h, _score) in top]
        # Build reranker input texts
        texts = [build_rerank_text(h) for h in cand_hits]

        reranker = BgeReranker(RerankConfig())

        def _sync_rerank() -> List[Dict[str, Any]]:
            # Attach text for rerank, call reranker, drop temporary field
            tmp = [{**h, "text": t} for h, t in zip(cand_hits, texts)]
            rr = reranker.rerank(query, tmp, text_key="text")
            for h in rr:
                h.pop("text", None)
            return rr

        try:
            reranked = await asyncio.to_thread(_sync_rerank)
            # Replace top with reranked (trim to k)
            top = [(h, float(h.get("rerank_score", 0.0))) for h in reranked][:k]
        except Exception:
            # If reranker not available or fails, keep fused ordering
            top = fused[:k]
    else:
        # No reranker; trim fused top-k
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

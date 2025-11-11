from __future__ import annotations

from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession


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
    """Placeholder for hybrid search; to be implemented.

    Should fuse FTS and ANN candidates (e.g., via RRF), then optionally rerank
    and apply MMR, returning top-k.
    """
    raise NotImplementedError("Hybrid search not implemented yet")


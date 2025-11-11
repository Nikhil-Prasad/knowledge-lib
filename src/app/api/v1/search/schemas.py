from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel

from src.app.domain.common import Modality


class BaseSearchRequest(BaseModel):
    query: str
    k: int = 20
    collection_id: Optional[UUID] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 20


class SearchHit(BaseModel):
    modality: Modality
    segment_id: UUID
    container_id: UUID
    page_no: Optional[int] = None
    score: float
    snippet: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchHit]


# Skeletonized request models for specific endpoints
class FtsSearchRequest(BaseSearchRequest):
    pass


class AnnSearchRequest(BaseSearchRequest):
    # Placeholder for ANN-specific params (e.g., vector model/version)
    pass


class HybridSearchRequest(BaseSearchRequest):
    # Placeholder for hybrid-specific params (e.g., rrf_k, weights)
    pass

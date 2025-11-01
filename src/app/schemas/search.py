from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID

class SearchRequest(BaseModel):
    query: str
    k: int = 20

class SearchHit(BaseModel):
    doc_id: UUID
    page_no: int
    chunk_id: UUID
    score: float
    snippet: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchHit]


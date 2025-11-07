from __future__ import annotations

from typing import List, Optional, Literal, Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from .segments import SegmentRef
from .anchors import Anchor


Relation = Literal[
    "supports",
    "contradicts",
    "refers_to",
    "caption_of",
    "mentions_entity",
    "same_as",
    "similar_to",
]

LinkScope = Literal["document", "collection", "global"]


class Link(BaseModel):
    link_id: UUID
    src: SegmentRef
    dst: SegmentRef
    relation: Relation
    scope: LinkScope
    scope_id: Optional[UUID] = None
    anchors: List[Anchor] = Field(default_factory=list)
    weight: Optional[float] = None
    confidence: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None
    created_by: Optional[str] = None
    method: Optional[str] = None
    model_version: Optional[str] = None
    # created_at is DB-managed; omitted in domain


CandidateStatus = Literal["proposed", "accepted", "rejected"]


class LinkCandidate(BaseModel):
    candidate_id: UUID
    src: SegmentRef
    dst: SegmentRef
    relation: Optional[Relation] = None
    anchors: List[Anchor] = Field(default_factory=list)
    score: Optional[float] = None
    confidence: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None
    batch_id: Optional[UUID] = None
    status: CandidateStatus = "proposed"
    created_by: Optional[str] = None
    method: Optional[str] = None
    model_version: Optional[str] = None
    # created_at is DB-managed; omitted in domain

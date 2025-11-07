from __future__ import annotations

from typing import Annotated, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from .common import BBox


AnchorType = Literal["text_span", "bbox", "table_cell", "av_window", "citation_ref"]


class TextAnchor(BaseModel):
    atype: Literal["text_span"] = "text_span"
    segment_id: UUID
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def _check_span(self) -> "TextAnchor":
        if self.end <= self.start:
            raise ValueError("Invalid TextAnchor: end must be > start")
        return self


class BBoxAnchor(BaseModel):
    atype: Literal["bbox"] = "bbox"
    segment_id: UUID  # typically a Figure segment
    bbox: BBox


class TableAnchor(BaseModel):
    atype: Literal["table_cell"] = "table_cell"
    table_id: UUID
    row_index: Annotated[int, Field(ge=0)]
    col_index: Optional[Annotated[int, Field(ge=0)]] = None


class AVAnchor(BaseModel):
    atype: Literal["av_window"] = "av_window"
    segment_id: UUID  # audio/video segment
    t0_ms: Annotated[int, Field(ge=0)]
    t1_ms: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def _check_window(self) -> "AVAnchor":
        if self.t1_ms <= self.t0_ms:
            raise ValueError("Invalid AVAnchor: t1_ms must be > t0_ms")
        return self


class CitationRef(BaseModel):
    atype: Literal["citation_ref"] = "citation_ref"
    segment_id: UUID  # references an existing CitationSegment id


Anchor = Annotated[
    Union[TextAnchor, BBoxAnchor, TableAnchor, AVAnchor, CitationRef],
    Field(discriminator="atype"),
]


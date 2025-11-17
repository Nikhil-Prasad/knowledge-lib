from __future__ import annotations

from typing import Optional, Literal, Annotated, Union
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from .common import (
    Modality,
    TextSegmentType,
    BBox,
    SegmentBase,
)


# ----- text modality segments ----- #

class TextSegment(SegmentBase):
    modality: Literal["text"] = "text"
    container_id: UUID
    page_no: Annotated[int, Field(ge=1)]
    object_type: TextSegmentType
    section_path: Optional[str] = None
    bbox: Optional[BBox] = None
    text: str
    text_source: Optional[Literal["vector", "ocr", "fused"]] = None
    pre_score: Optional[float] = None
    post_score: Optional[float] = None
    emb_model: Optional[str] = None
    emb_version: Optional[str] = None
    chunk_version: Optional[str] = None


# ----- table segments ----- #

class TableRow(SegmentBase):
    modality: Literal["table"] = "table"
    table_id: UUID
    row_index: Annotated[int, Field(ge=0)]
    row_json: dict
    row_text: str
    pre_score: Optional[float] = None
    post_score: Optional[float] = None
    emb_model: Optional[str] = None
    emb_version: Optional[str] = None


# ----- citation segments ----- #

class CitationSegment(SegmentBase):
    modality: Literal["citation"] = "citation"
    container_id: UUID
    page_no: Annotated[int, Field(ge=1)]
    char_offset: Optional[int] = None
    marker: str
    target_bib: Optional[UUID] = None


class BibliographyEntry(BaseModel):
    bib_id: UUID
    container_id: UUID
    label: Optional[str] = None
    raw_text: str
    parsed: Optional[dict] = None
    # created_at is DB-managed; omitted in domain


# ----- image / audio / video segments ----- #

class Figure(SegmentBase):
    modality: Literal["image"] = "image"
    container_id: UUID
    page_no: Annotated[int, Field(ge=1)]
    bbox: Optional[BBox] = None
    caption_segment_id: Optional[UUID] = None
    image_uri: Optional[str] = None
    emb_model: Optional[str] = None
    emb_version: Optional[str] = None


class AudioSegment(SegmentBase):
    modality: Literal["audio"] = "audio"
    t0_ms: Annotated[int, Field(ge=0)]
    t1_ms: Annotated[int, Field(gt=0)]
    transcript_segment_id: Optional[UUID] = None
    emb_model: Optional[str] = None
    emb_version: Optional[str] = None

    @model_validator(mode="after")
    def _check_window(self) -> "AudioSegment":
        if self.t1_ms <= self.t0_ms:
            raise ValueError("t1_ms must be > t0_ms")
        return self


class VideoSegment(SegmentBase):
    modality: Literal["video"] = "video"
    shot_id: Optional[str] = None
    frame_idx: Optional[int] = None
    t0_ms: Optional[int] = None
    t1_ms: Optional[int] = None
    keyframe_uri: Optional[str] = None
    emb_model: Optional[str] = None
    emb_version: Optional[str] = None

    @model_validator(mode="after")
    def _check_time(self) -> "VideoSegment":
        if self.t0_ms is not None and self.t1_ms is not None and self.t1_ms <= self.t0_ms:
            raise ValueError("t1_ms must be > t0_ms")
        return self


# ----- discriminated unions ----- #

Segment = Annotated[
    Union[TextSegment, TableRow, CitationSegment, Figure, AudioSegment, VideoSegment],
    Field(discriminator="modality"),
]


class SegmentRef(BaseModel):
    segment_id: UUID
    modality: Modality
    container_id: Optional[UUID] = None

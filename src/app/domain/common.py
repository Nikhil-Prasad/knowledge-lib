from __future__ import annotations

from typing import Literal, Annotated, Tuple
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


# Modalities supported across the system
Modality = Literal["text", "table", "citation", "image", "video", "audio", "container"]

# Within-text segment classification (used by text segments)
TextSegmentType = Literal[
    "title",
    "heading",
    "paragraph",
    "caption",
    "footnote",
    "sentence_window",
    "blob",
]

class BBox(BaseModel):
    x0: Annotated[float, Field(ge=0, le=1)]
    y0: Annotated[float, Field(ge=0, le=1)]
    x1: Annotated[float, Field(ge=0, le=1)]
    y1: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def _check_order(self) -> "BBox":
        if (self.x1 - self.x0) <= 0 or (self.y1 - self.y0) <= 0:
            raise ValueError("Invalid BBox: require x0 < x1 and y0 < y1 with non-zero area")
        return self

    def to_px(self, width_px: int, height_px: int) -> Tuple[int, int, int, int]:
        x0 = int(round(self.x0 * width_px))
        y0 = int(round(self.y0 * height_px))
        x1 = int(round(self.x1 * width_px))
        y1 = int(round(self.y1 * height_px))
        return x0, y0, x1, y1

    @classmethod
    def from_px(
        cls,
        x0_px: float,
        y0_px: float,
        x1_px: float,
        y1_px: float,
        width_px: int,
        height_px: int,
        *,
        origin_top_left: bool = True,
        clip: bool = True,
    ) -> "BBox":
        if not origin_top_left:
            y0_px, y1_px = height_px - y1_px, height_px - y0_px

        x0 = x0_px / width_px
        y0 = y0_px / height_px
        x1 = x1_px / width_px
        y1 = y1_px / height_px

        if clip:
            x0 = min(max(x0, 0.0), 1.0)
            y0 = min(max(y0, 0.0), 1.0)
            x1 = min(max(x1, 0.0), 1.0)
            y1 = min(max(y1, 0.0), 1.0)

        return cls(x0=x0, y0=y0, x1=x1, y1=y1)


class SegmentBase(BaseModel):
    segment_id: UUID
    modality: Modality

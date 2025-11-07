from __future__ import annotations

from typing import Optional, List, Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from .common import Modality, BBox


class Asset(BaseModel):
    asset_id: UUID
    source_uri: str
    mime_type: str
    modality: Modality
    bytes: Optional[int] = None


class Document(BaseModel):
    container_id: UUID
    asset_id: UUID
    title: Optional[str] = None
    language: Optional[str] = None
    is_scanned: bool = False


class Page(BaseModel):
    container_id: UUID
    page_no: Annotated[int, Field(ge=1)]
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    text: Optional[str] = None
    image_uri: Optional[str] = None


# ----- table containers ----- #

class TableSchemaCol(BaseModel):
    name: str
    dtype: str                          # e.g., "string","float64","int64","date"
    unit: Optional[str] = None
    description: Optional[str] = None


class TableSet(BaseModel):
    table_id: UUID
    asset_id: UUID
    name: Optional[str] = None          # filename, caption, detected title
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    schema: List[TableSchemaCol] = Field(default_factory=list)
    page_no: Optional[Annotated[int, Field(ge=1)]] = None       # for tables extracted from containers
    bbox: Optional[BBox] = None         # same
    # created_at is DB-managed; omitted in domain

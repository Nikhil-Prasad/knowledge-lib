from __future__ import annotations

from typing import Optional, Literal, Annotated, Union
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl
from src.app.domain.common import Modality


class RawTextSource(BaseModel):
    source_type: Literal["raw_text"] = "raw_text"
    title: Optional[str] = None
    text: str


class RemoteUriSource(BaseModel):
    source_type: Literal["remote_uri"] = "remote_uri"
    uri: HttpUrl | str  # allow non-HTTP schemes if needed
    content_type_hint: Optional[str] = None


class DataUriSource(BaseModel):
    source_type: Literal["data_uri"] = "data_uri"
    data_uri: str  # RFC 2397 data URI (e.g., data:...;base64,...)
    content_type_hint: Optional[str] = None


class UploadRefSource(BaseModel):
    source_type: Literal["upload_ref"] = "upload_ref"
    upload_uri: str  # e.g., s3://, file://, gs://, or internal handle
    content_type_hint: Optional[str] = None


IngestSource = Annotated[
    Union[RawTextSource, RemoteUriSource, DataUriSource, UploadRefSource],
    Field(discriminator="source_type"),
]


class IngestOptions(BaseModel):
    dedupe: bool = True
    modality_hint: Optional[Modality] = None


class IngestRequest(BaseModel):
    source: IngestSource
    options: IngestOptions = Field(default_factory=IngestOptions)
    collection_id: Optional[UUID] = None


class IngestResponse(BaseModel):
    container_id: UUID
    pages_created: int
    segments_created: int

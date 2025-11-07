from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import (
    String, Text, Integer, Boolean, TIMESTAMP, JSON, ForeignKey,
    Index, text, Computed, Float, Enum as SAEnum
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, BYTEA, TSVECTOR
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Container(Base):
    __tablename__ = "containers"

    container_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_uri: Mapped[str]   = mapped_column(Text, nullable=False)
    mime_type: Mapped[str]    = mapped_column(Text, nullable=False)
    sha256: Mapped[bytes]     = mapped_column(BYTEA, nullable=False)
    title: Mapped[Optional[str]]    = mapped_column(Text)
    language: Mapped[Optional[str]] = mapped_column(Text)
    is_scanned: Mapped[bool]  = mapped_column(Boolean, server_default=text("FALSE"), nullable=False)

    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False)

    pages:  Mapped[List["Page"]]         = relationship(back_populates="container", cascade="all, delete-orphan")
    text_segments: Mapped[List["TextSegment"]] = relationship(back_populates="container", cascade="all, delete-orphan")


class Page(Base):
    __tablename__ = "pages"

    container_id:  Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("containers.container_id", ondelete="CASCADE"), primary_key=True
    )
    page_no: Mapped[int] = mapped_column(Integer, primary_key=True)

    text:      Mapped[Optional[str]] = mapped_column(Text)
    image_uri: Mapped[Optional[str]] = mapped_column(Text)  # path to page render (webp/png)
    width_px:  Mapped[Optional[int]] = mapped_column(Integer)
    height_px: Mapped[Optional[int]] = mapped_column(Integer)

    container: Mapped["Container"] = relationship(back_populates="pages")
    text_segments:   Mapped[List["TextSegment"]] = relationship(back_populates="page", cascade="all, delete-orphan")


class TextSegment(Base):
    __tablename__ = "text_segments"

    segment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    container_id:   Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("containers.container_id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_no:  Mapped[int]       = mapped_column(Integer, nullable=False)

    object_type:  Mapped[str]          = mapped_column(
        SAEnum(
            'title', 'heading', 'paragraph', 'caption', 'footnote', 'sentence_window', 'blob',
            name='text_object_type',
            native_enum=True,
        ),
        nullable=False,
    )
    section_path: Mapped[Optional[str]] = mapped_column(Text)
    bbox:         Mapped[Optional[dict]] = mapped_column(JSON)                   
    text:         Mapped[str]           = mapped_column(Text, nullable=False)

    text_fts: Mapped[str] = mapped_column(
        TSVECTOR,
        nullable=False,
    )

    emb_v1:        Mapped[Optional[List[float]]] = mapped_column(Vector(1536)) #openAI text-embedder-3-small for 1536 dim. 
    emb_model:     Mapped[Optional[str]] = mapped_column(Text)
    emb_version:   Mapped[Optional[str]] = mapped_column(Text)
    chunk_version: Mapped[Optional[str]] = mapped_column(Text)

    container: Mapped["Container"] = relationship(back_populates="text_segments")
    page:     Mapped["Page"] = relationship(
        back_populates="text_segments",
        primaryjoin="and_(TextSegment.container_id==Page.container_id, TextSegment.page_no==Page.page_no)",
        viewonly=True,
    )

    __table_args__ = (
        Index("idx_text_segments_doc_page", "container_id", "page_no"),
        Index("idx_text_segments_fts", "text_fts", postgresql_using="gin"),
        Index(
            "idx_text_segments_emb_v1",
            "emb_v1",
            postgresql_using="hnsw",
            postgresql_ops={"emb_v1": "vector_cosine_ops"},
        ),
    )


# ---------------- additional segment tables to mirror schemas ---------------- #

class Figure(Base):
    __tablename__ = "figures"

    figure_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    container_id:    Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("containers.container_id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_no:   Mapped[int] = mapped_column(Integer, nullable=False)
    bbox:      Mapped[Optional[dict]] = mapped_column(JSON)
    caption_segment_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("text_segments.segment_id"))
    image_uri: Mapped[Optional[str]] = mapped_column(Text)

    emb_v1:      Mapped[Optional[List[float]]] = mapped_column(Vector(1536))
    emb_model:   Mapped[Optional[str]] = mapped_column(Text)
    emb_version: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("idx_figures_doc_page", "container_id", "page_no"),
        Index(
            "idx_figures_emb_v1",
            "emb_v1",
            postgresql_using="hnsw",
            postgresql_ops={"emb_v1": "vector_cosine_ops"},
        ),
    )


class TableSet(Base):
    __tablename__ = "table_sets"

    table_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    container_id:   Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("containers.container_id", ondelete="CASCADE"), nullable=False, index=True
    )
    name:     Mapped[Optional[str]] = mapped_column(Text)
    n_rows:   Mapped[Optional[int]] = mapped_column(Integer)
    n_cols:   Mapped[Optional[int]] = mapped_column(Integer)
    schema:   Mapped[Optional[dict]] = mapped_column(JSON)
    page_no:  Mapped[Optional[int]] = mapped_column(Integer)
    bbox:     Mapped[Optional[dict]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False)

    rows: Mapped[List["TableRow"]] = relationship(back_populates="table", cascade="all, delete-orphan")


class TableRow(Base):
    __tablename__ = "table_rows"

    row_id:   Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    table_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("table_sets.table_id", ondelete="CASCADE"), nullable=False, index=True
    )
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    row_json:  Mapped[dict] = mapped_column(JSON, nullable=False)
    row_text:  Mapped[str] = mapped_column(Text, nullable=False)
    row_text_fts: Mapped[str] = mapped_column(TSVECTOR, nullable=False)

    emb_v1:      Mapped[Optional[List[float]]] = mapped_column(Vector(1536))
    emb_model:   Mapped[Optional[str]] = mapped_column(Text)
    emb_version: Mapped[Optional[str]] = mapped_column(Text)

    table: Mapped["TableSet"] = relationship(back_populates="rows")

    __table_args__ = (
        Index("idx_table_rows_table_idx", "table_id", "row_index"),
        Index("idx_table_rows_fts", "row_text_fts", postgresql_using="gin"),
        Index(
            "idx_table_rows_emb_v1",
            "emb_v1",
            postgresql_using="hnsw",
            postgresql_ops={"emb_v1": "vector_cosine_ops"},
        ),
    )


class BibliographyEntry(Base):
    __tablename__ = "bibliography_entries"

    bib_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    container_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("containers.container_id", ondelete="CASCADE"), nullable=False, index=True
    )
    label:    Mapped[Optional[str]] = mapped_column(Text)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    parsed:   Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False)


class CitationAnchor(Base):
    __tablename__ = "citation_anchors"

    anchor_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    container_id:    Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("containers.container_id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_no:     Mapped[int] = mapped_column(Integer, nullable=False)
    char_offset: Mapped[Optional[int]] = mapped_column(Integer)
    marker:      Mapped[str] = mapped_column(String, nullable=False)
    target_bib:  Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("bibliography_entries.bib_id"))

    __table_args__ = (
        Index("idx_citation_anchors_doc_page", "container_id", "page_no"),
    )


class AudioSegment(Base):
    __tablename__ = "audio_segments"

    segment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    container_id:     Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("containers.container_id", ondelete="CASCADE"), nullable=False, index=True
    )
    t0_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    t1_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    transcript_segment_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("text_segments.segment_id"))

    emb_v1:      Mapped[Optional[List[float]]] = mapped_column(Vector(1536))
    emb_model:   Mapped[Optional[str]] = mapped_column(Text)
    emb_version: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("idx_audio_segments_doc", "container_id"),
        Index(
            "idx_audio_segments_emb_v1",
            "emb_v1",
            postgresql_using="hnsw",
            postgresql_ops={"emb_v1": "vector_cosine_ops"},
        ),
    )


class VideoSegment(Base):
    __tablename__ = "video_segments"

    segment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    container_id:     Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("containers.container_id", ondelete="CASCADE"), nullable=False, index=True
    )
    shot_id:    Mapped[Optional[str]] = mapped_column(Text)
    frame_idx:  Mapped[Optional[int]] = mapped_column(Integer)
    t0_ms:      Mapped[Optional[int]] = mapped_column(Integer)
    t1_ms:      Mapped[Optional[int]] = mapped_column(Integer)
    keyframe_uri: Mapped[Optional[str]] = mapped_column(Text)

    emb_v1:      Mapped[Optional[List[float]]] = mapped_column(Vector(1536))
    emb_model:   Mapped[Optional[str]] = mapped_column(Text)
    emb_version: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("idx_video_segments_doc", "container_id"),
        Index(
            "idx_video_segments_emb_v1",
            "emb_v1",
            postgresql_using="hnsw",
            postgresql_ops={"emb_v1": "vector_cosine_ops"},
        ),
    )


# ---------------- link overlay tables ---------------- #

class Link(Base):
    __tablename__ = "links"

    link_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    src_segment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    src_modality:   Mapped[str] = mapped_column(String, nullable=False)
    dst_segment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    dst_modality:   Mapped[str] = mapped_column(String, nullable=False)
    relation:       Mapped[str] = mapped_column(String, nullable=False)
    scope:          Mapped[str] = mapped_column(String, nullable=False)
    scope_id:       Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))

    weight:     Mapped[Optional[float]] = mapped_column(Float)
    confidence: Mapped[Optional[float]] = mapped_column(Float)

    created_by:   Mapped[Optional[str]] = mapped_column(Text)
    method:       Mapped[Optional[str]] = mapped_column(Text)
    model_version:Mapped[Optional[str]] = mapped_column(Text)
    created_at:   Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False)

    __table_args__ = (
        Index("idx_links_src", "src_segment_id"),
        Index("idx_links_dst", "dst_segment_id"),
        Index("idx_links_scope", "scope", "scope_id"),
    )


class LinkAnchor(Base):
    __tablename__ = "link_anchors"

    link_id:    Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("links.link_id", ondelete="CASCADE"), primary_key=True
    )
    atype:      Mapped[str] = mapped_column(String, primary_key=True)
    anchor:     Mapped[dict] = mapped_column(JSON, nullable=False)

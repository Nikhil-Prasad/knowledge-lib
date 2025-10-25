from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import (
    String, Text, Integer, Boolean, TIMESTAMP, JSON, ForeignKey,
    Index, text, Computed
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, BYTEA, TSVECTOR
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_uri: Mapped[str]   = mapped_column(Text, nullable=False)      
    mime_type: Mapped[str]    = mapped_column(Text, nullable=False)
    sha256: Mapped[bytes]     = mapped_column(BYTEA, nullable=False)
    title: Mapped[Optional[str]]    = mapped_column(Text)
    language: Mapped[Optional[str]] = mapped_column(Text)
    is_scanned: Mapped[bool]  = mapped_column(Boolean, server_default=text("FALSE"), nullable=False)

    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False)

    pages:  Mapped[List["Page"]]  = relationship(back_populates="document", cascade="all, delete-orphan")
    chunks: Mapped[List["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Page(Base):
    __tablename__ = "pages"

    doc_id:  Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True
    )
    page_no: Mapped[int] = mapped_column(Integer, primary_key=True)

    text:      Mapped[Optional[str]] = mapped_column(Text)
    image_uri: Mapped[Optional[str]] = mapped_column(Text)  # path to page render (webp/png)

    document: Mapped["Document"] = relationship(back_populates="pages")
    chunks:   Mapped[List["Chunk"]] = relationship(back_populates="page", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id:   Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_no:  Mapped[int]       = mapped_column(Integer, nullable=False)

    object_type:  Mapped[str]          = mapped_column(String, nullable=False)   
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

    document: Mapped["Document"] = relationship(back_populates="chunks")
    page:     Mapped["Page"] = relationship(
        back_populates="chunks",
        primaryjoin="and_(Chunk.doc_id==Page.doc_id, Chunk.page_no==Page.page_no)",
        viewonly=True,
    )

    __table_args__ = (
        Index("idx_chunks_doc_page", "doc_id", "page_no"),
        Index("idx_chunks_fts", "text_fts", postgresql_using="gin"),
        Index(
            "idx_chunks_emb_v1",
            "emb_v1",
            postgresql_using="hnsw",
            postgresql_ops={"emb_v1": "vector_cosine_ops"},
        ),
    )

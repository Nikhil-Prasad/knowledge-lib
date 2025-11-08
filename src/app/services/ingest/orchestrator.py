"""Orchestrator for the unified ingest flow.

Responsibilities:
- Resolve transport (raw_text, remote_uri, data_uri, upload_ref) to concrete bytes/text
  and provenance (source_uri, sha256, filename?).
- Identify mime type and choose a pipeline (text vs container).
- Enforce dedupe (optional) and call the chosen pipeline.

Current scope: keep it simple and focus on text inputs (raw text, local files, remote URIs).
"""

from __future__ import annotations

from sqlalchemy.orm import Session
from typing import Optional
from pathlib import Path
from urllib.parse import urlparse, unquote

from src.app.api.v1.ingest.schemas import (
    IngestRequest,
    IngestResponse,
    RawTextSource,
    UploadRefSource,
)
from . import text_pipeline


# ---------------- Identify/Resolve helpers (text-only focus for now) ---------------- #

def is_text_mime(mime: Optional[str]) -> bool:
    """True if the mime is a supported text type (plain/markdown)."""
    if not mime:
        return False
    base = mime.split(";")[0].strip().lower()
    return base in {"text/plain", "text/markdown", "text/x-markdown", "text/md"}


def infer_mime_from_filename(name: Optional[str]) -> Optional[str]:
    """Infer text mime from filename extension (.txt/.md)."""
    if not name:
        return None
    ext = Path(name).suffix.lower()
    if ext == ".txt":
        return "text/plain"
    if ext in {".md", ".markdown"}:
        return "text/markdown"
    return None


def decode_bytes_to_text(data: bytes, charset_hint: Optional[str] = None) -> str:
    """Decode bytes into text using charset_hint or UTF-8 with BOM handling and fallback."""
    if charset_hint:
        try:
            return data.decode(charset_hint, errors="strict")
        except Exception:
            pass
    try:
        return data.decode("utf-8-sig", errors="strict")
    except Exception:
        txt = data.decode("utf-8", errors="replace")
        if txt.count("�") > max(3, len(txt) // 200):
            return data.decode("latin-1", errors="replace")
        return txt


# ---------------- Public entrypoint ---------------- #

def ingest(session: Session, req: IngestRequest) -> IngestResponse:
    """Entry point for the unified ingest flow (text-focused scaffold).

    Flow per source:
    - raw_text: already identified/resolved → call text pipeline directly.
    - upload_ref: treat as local file or file:// → read bytes → decode → call text pipeline.
    - remote_uri: (intentionally not implemented here)

    Note: This is a scaffold. Implement branches and call the text pipeline accordingly.
    """
    src = req.source

    if isinstance(src, RawTextSource):
        return text_pipeline.ingest_raw_text_pipeline(
            session,
            text=src.text,
            title_hint=src.title,
            source_uri="raw:text",
            mime_type="text/plain",
            options=req.options,
            collection_id=req.collection_id,
        )

    if isinstance(src, UploadRefSource):
        uri = str(src.upload_uri)
        if uri.startswith("file://"):
            p = urlparse(uri)
            abs_path = unquote(p.path)
            source_uri = uri
        else:
            abs_path = str(Path(uri).expanduser().resolve())
            source_uri = f"file://{abs_path}"

        path = Path(abs_path)
        filename = path.name
        title_hint = path.stem

        mime = src.content_type_hint or infer_mime_from_filename(filename) or "text/plain"
        if not is_text_mime(mime):
            raise NotImplementedError(f"upload_ref non-text not supported yet (mime={mime})")

        data = path.read_bytes()
        text = decode_bytes_to_text(data)

        return text_pipeline.ingest_raw_text_pipeline(
            session,
            text=text,
            title_hint=title_hint,
            source_uri=source_uri,
            mime_type=mime,
            options=req.options,
            collection_id=req.collection_id,
        )

    raise NotImplementedError("ingest: source type not supported yet")

from __future__ import annotations

from sqlalchemy.orm import Session

from src.app.api.v1.ingest.schemas import RawTextSource, IngestResponse


def ingest_text_segment(session: Session, src: RawTextSource) -> IngestResponse:
    """Plan: Ingest a raw text payload into the uniform retrieval layer.

    Overview
    - Treat the input as a lightweight container (container_id) with a single page (page_no=1).
    - Split text per policy (paragraph/sentence/sentence_window) into TextSegments and persist.
    - FTS trigger populates text_fts; embeddings are optional and filled asynchronously.

    Steps (to implement)
    1) Compute `sha256` of `src.text` for dedupe; if options.dedupe, short‑circuit to existing container.
    2) Create a new row in `containers` with:
       - source_uri = "raw:text"
       - mime_type   = "text/plain"
       - sha256, title (optional)
       Capture `container_id`.
    3) Insert one `pages` row: (container_id, page_no=1, text=full body).
    4) Segment the text into units per policy:
       - If no segmenter yet, create a single segment with object_type="blob".
       - Else, for each unit, insert into `text_segments`:
         { container_id, page_no=1, object_type (TextSegmentType), section_path?, bbox=None, text }
    5) Return IngestResponse(container_id, pages_created, segments_created).

    Notes
    - All writes happen in a single transaction.
    - FTS: `text_fts` is trigger‑maintained; no client computation needed.
    - Embeddings: leave `emb_v1` null; queue a background job if enabled later.
    - The API DTO already exposes `container_id` in the response.
    """
    raise NotImplementedError("ingest_text_segment is a planned pipeline; implement per the docstring plan")

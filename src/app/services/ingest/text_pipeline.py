"""Text-only ingestion pipeline (raw text and text/* sources).

This pipeline normalizes input text, optionally extracts a title, segments the
body according to a chosen policy (sentence_window/paragraph/blob), persists a
container + single page, and inserts text segments. Embeddings may be populated
on-demand or via a background job.
"""

from typing import Iterable, Optional, List
from uuid import UUID
from sqlalchemy import func
import hashlib
import re
import unicodedata
from sqlalchemy.orm import Session

from src.app.api.v1.ingest.schemas import IngestOptions, IngestResponse
from src.app.domain.common import TextSegmentType
from src.app.db.models.models import TextSegment, Container, Page
from src.app.settings import get_settings


def ingest_raw_text_pipeline(
    session: Session,
    text: str,
    title_hint: Optional[str],
    source_uri: str,
    mime_type: str,
    options: IngestOptions,
    collection_id: Optional[UUID] = None,
) -> IngestResponse:
    """Raw-text ingestion pipeline: create container + page then insert text segments.

    Steps:
    1) Normalize text and extract a title if obvious (prefer title_hint).
    2) Dedupe by sha256 of normalized body if enabled.
    3) Create a container and one page (page_no=1).
    4) Segment the body (blob for very short, else sentence windows) and insert segments.
    5) Return counts.
    """
    try:
        # 1) Normalize input and extract title
        norm = normalize_text(text)
        if not norm.strip():
            raise ValueError("Empty input after normalization")

        title, body = extract_title_and_body(norm, hint=title_hint)

        # 2) Dedupe on normalized body
        body_bytes = body.encode("utf-8")
        digest = hashlib.sha256(body_bytes).digest()
        if options.dedupe:
            existing = session.query(Container).filter(Container.sha256 == digest).first()
            if existing is not None:
                return IngestResponse(
                    container_id=existing.container_id,
                    pages_created=0,
                    segments_created=0,
                )

        # 3) Create container and single page
        title_final = title or title_hint or f"untitled-{hashlib.sha256(body_bytes).hexdigest()[:8]}"
        container = Container(
            source_uri=source_uri,
            mime_type=mime_type,
            sha256=digest,
            title=title_final,
        )
        session.add(container)
        session.flush()

        page = Page(container_id=container.container_id, page_no=1, text=body)
        session.add(page)

        # 4) Segment body and insert segments
        segments_created = 0

        inserted: List[tuple[UUID, str]] = []
        if title and title.strip():
            seg_id = ingest_text_segment(
                session,
                container_id=container.container_id,
                page_no=1,
                object_type="title",
                text=title.strip(),
            )
            segments_created += 1
            inserted.append((seg_id, title.strip()))

        settings = get_settings()
        short_thresh = settings.ingest_defaults.short_text_threshold_chars
        if len(body) < short_thresh:
            seg_id = ingest_text_segment(
                session,
                container_id=container.container_id,
                page_no=1,
                object_type="blob",
                text=body.strip(),
            )
            segments_created += 1
            inserted.append((seg_id, body.strip()))
        else:
            sentences = split_sentences(body)
            k = settings.ingest_defaults.sentence_window_k
            ov = settings.ingest_defaults.sentence_window_overlap
            windows = make_sentence_windows(sentences, k=k, overlap=ov)
            for w in windows:
                if not w.strip():
                    continue
                seg_id = ingest_text_segment(
                    session,
                    container_id=container.container_id,
                    page_no=1,
                    object_type="sentence_window",
                    text=w.strip(),
                )
                segments_created += 1
                inserted.append((seg_id, w.strip()))

        # 5) Associate to collection if provided
        if collection_id is not None:
            from src.app.db.models.models import Collection
            coll = session.get(Collection, collection_id)
            if coll is None:
                coll = Collection(collection_id=collection_id)
                session.add(coll)
                session.flush()
            container.collections.append(coll)

        # Embeddings are computed in a background task after commit.

        session.commit()

        return IngestResponse(
            container_id=container.container_id,
            pages_created=1,
            segments_created=segments_created,
        )
    except Exception:
        session.rollback()
        raise


def ingest_text_segment(
    session: Session,
    *,
    container_id: UUID,
    page_no: int,
    object_type: TextSegmentType,
    text: str,
    section_path: Optional[str] = None,
    bbox: Optional[dict] = None,
) -> UUID:
    """Insert a single TextSegment row and return its segment_id.

    Notes:
    - Computes FTS value via `to_tsvector('simple', :text)` using a SQL function call.
    - Embeddings are left null; a background job or on-demand call can populate them.
    """

    seg = TextSegment(
        container_id=container_id,
        page_no=page_no,
        object_type=object_type,  # must be one of TextSegmentType literals
        section_path=section_path,
        bbox=bbox,
        text=text,
        text_fts=func.to_tsvector('simple', text),
    )
    session.add(seg)
    # Flush to obtain the generated segment_id without committing the transaction
    session.flush()
    return seg.segment_id


def normalize_text(raw: str) -> str:
    """Normalize whitespace/newlines and strip control characters.

    Suggested behavior:
    - Unicode NFKC; keep tabs/newlines; remove other control chars.
    - Convert CRLF/CR to LF; collapse multiple blank lines.
    - Trim trailing spaces; dedent if needed.
    """
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "".join(ch for ch in s if (ch == "\t" or ch == "\n" or ord(ch) >= 32))
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_title_and_body(text: str, hint: Optional[str] = None) -> tuple[Optional[str], str]:
    """Extract a title (if obvious) and return the remaining body.

    Heuristic for plain text:
    - If `hint` provided, prefer it.
    - Else if first non-empty line is 5–200 chars, low digit ratio, not all caps, and followed by a blank line,
      treat it as title and remove it from body. Otherwise return (None, text).
    """
    if hint and hint.strip():
        return hint.strip(), text

    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return None, text

    candidate = lines[i].strip()
    next_is_blank = (i + 1 < len(lines)) and (lines[i + 1].strip() == "")
    length_ok = 5 <= len(candidate) <= 200
    digits = sum(c.isdigit() for c in candidate)
    digit_ratio = digits / max(1, len(candidate))
    letters = [c for c in candidate if c.isalpha()]
    upper_ratio = (sum(c.isupper() for c in letters) / max(1, len(letters))) if letters else 0.0
    not_all_caps = upper_ratio < 0.9

    if length_ok and digit_ratio < 0.2 and not_all_caps and next_is_blank:
        body_lines = lines[i + 2 :]
        return candidate, "\n".join(body_lines).lstrip()

    return None, text


def split_paragraphs(text: str) -> list[str]:
    """Split body into paragraphs using blank lines as separators.

    Filter very short paragraphs (e.g., < 40 chars) if desired, or keep them for recall.
    """
    parts = re.split(r"\n\s*\n+", text.strip())
    paras = [re.sub(r"[ \t]{2,}", " ", p.strip()) for p in parts if p.strip()]
    return paras


def split_sentences(text: str) -> list[str]:
    """Split body into sentences.

    Use a lightweight rule-based splitter or a library if available. Ensure consistent handling of abbreviations.
    """
    text = text.strip()
    if not text:
        return []

    chunks = re.split(r"(?<=[.!?])\s+", text)
    chunks = [c.strip() for c in chunks if c and c.strip()]
    if not chunks:
        return []

    abbreviations = {
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.",
        "vs.", "etc.", "e.g.", "i.e.", "Fig.", "Eq.", "No.", "St.", "Mt.",
    }

    sentences: List[str] = []
    for c in chunks:
        if not sentences:
            sentences.append(c)
            continue
        prev = sentences[-1]
        tail = prev.split()[-1] if prev.split() else ""
        if tail in abbreviations or prev.endswith("..."):
            sentences[-1] = prev + " " + c
        else:
            sentences.append(c)

    cleaned: List[str] = []
    for s in sentences:
        if len(s) < 2 and cleaned:
            cleaned[-1] = cleaned[-1] + " " + s
        else:
            cleaned.append(s)
    return cleaned


def make_sentence_windows(sentences: Iterable[str], k: int = 3, overlap: int = 1) -> list[str]:
    """Build sliding windows of K contiguous sentences with the specified overlap.

    Example: k=3, overlap=1 → windows [s0+s1+s2], [s2+s3+s4], ... (stride = k - overlap).
    """
    sents = [s.strip() for s in sentences if s and s.strip()]
    if not sents:
        return []
    stride = max(1, k - overlap)
    windows: List[str] = []
    from src.app.settings import get_settings
    soft_max = get_settings().ingest_defaults.window_soft_max_chars

    for i in range(0, len(sents), stride):
        chunk = sents[i : i + k]
        if not chunk:
            continue
        joined = " ".join(chunk).strip()
        while len(joined) > soft_max and len(chunk) > 1:
            chunk = chunk[:-1]
            joined = " ".join(chunk).strip()
        if len(joined) > soft_max:
            joined = joined[:soft_max].rstrip()
        windows.append(joined)

    deduped: List[str] = []
    for w in windows:
        if not deduped or deduped[-1] != w:
            deduped.append(w)
    return deduped

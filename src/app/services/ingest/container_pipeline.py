"""Container (PDF) ingestion pipeline scaffold.

Goal: Support application/pdf inputs, including image-based (scanned) PDFs.

High-level flow (image-heavy PDFs):
1) Resolve/identify: local file/URI → bytes/path, title hint, sha256.
2) Create container row; mark `mime_type='application/pdf'`.
3) Page discovery: extract page count + dimensions; create `pages` rows with
   `(container_id, page_no, width_px, height_px)`; optionally persist page renders.
4) Layout detection: detect regions (text/table/figure) per page.
5) Region processing:
   - text regions → OCR to lines/words; group → paragraphs; insert TextSegments with bbox.
   - table regions → table extraction → TableSet/TableRow rows (+embeddings later).
   - figure regions → create Figure rows with bbox; optional caption linking.
6) Enqueue text embedding task (existing embed runner) after commit.

This module provides contracts for layout/OCR providers and persists results
using domain and DB models. Heavy dependencies (OCR/layout) are not bundled;
providers are selected via settings and injected at runtime.
"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Iterable, List, Optional, Sequence, Tuple
import re
from uuid import UUID
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.app.api.v1.ingest.schemas import IngestOptions, IngestResponse
from src.app.db.models.models import Container, Page, Figure, TableSet, PageAnalysis, BibliographyEntry, CitationAnchor
from src.app.domain.common import BBox
from src.app.services.embeddings.embed_runner_async import embed_container_segments_async
from .providers import resolve_providers
from .providers.base import PDFPageInfo, LayoutRegion
from .providers.pymupdf_pager import extract_text_spans
from .text_pipeline import ingest_text_segment
from .page_router import route_page, PageSignals
from src.app.settings import get_settings
from pathlib import Path as _Path
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------- Pipeline Orchestrator ---------------- #

async def ingest_pdf_container_pipeline(
    session: AsyncSession,
    *,
    pdf_path: Path,
    options: IngestOptions,
    source_uri: str,
    title_hint: Optional[str] = None,
    collection_id: Optional[UUID] = None,
) -> IngestResponse:
    """Ingest a PDF container (scaffold; provider-driven parsing runs separately).

    Current behavior:
    - Dedup by sha256 of file bytes against `containers.sha256`.
    - Create a container row with mime_type=application/pdf and title from filename/hint.
    - Does not parse pages yet (no pages/segments inserted). Future work wires page
      enumeration, layout detection, region OCR, and persistence.
    """
    data = pdf_path.read_bytes()
    digest = hashlib.sha256(data).digest()

    # Dedupe check
    if options.dedupe:
        existing = (await session.execute(select(Container).where(Container.sha256 == digest))).scalar_one_or_none()
        if existing is not None:
            # Associate to collection if provided
            if collection_id is not None:
                from src.app.db.models.models import Collection, containers_collections
                coll = await session.get(Collection, collection_id)
                if coll is None:
                    coll = Collection(collection_id=collection_id)
                    session.add(coll)
                    await session.flush()
                stmt = (
                    pg_insert(containers_collections)
                    .values(collection_id=collection_id, container_id=existing.container_id)
                    .on_conflict_do_nothing(index_elements=["collection_id", "container_id"])
                )
                await session.execute(stmt)
                await session.commit()
            return IngestResponse(container_id=existing.container_id, pages_created=0, segments_created=0)

    # Create container
    title = title_hint or pdf_path.stem
    container = Container(
        source_uri=source_uri,
        mime_type="application/pdf",
        sha256=digest,
        title=title,
        is_scanned=False,  # TODO: detect scanned vs digital text
    )
    session.add(container)
    await session.flush()

    # Note: page enumeration, layout, and OCR happen in a background job via
    # `process_pdf_container_async`. This call only records the container.

    # Link to collection if requested
    if collection_id is not None:
        from src.app.db.models.models import Collection, containers_collections
        coll = await session.get(Collection, collection_id)
        if coll is None:
            coll = Collection(collection_id=collection_id)
            session.add(coll)
            await session.flush()
        stmt = (
            pg_insert(containers_collections)
            .values(collection_id=collection_id, container_id=container.container_id)
            .on_conflict_do_nothing(index_elements=["collection_id", "container_id"])
        )
        await session.execute(stmt)

    await session.commit()

    logger.info(f"Created PDF container %s for %s", container.container_id, pdf_path)
    return IngestResponse(container_id=container.container_id, pages_created=0, segments_created=0)


# ---------------- Future worker entry (stub) ---------------- #

async def process_pdf_container_async(
    *,
    container_id: UUID,
    pdf_path: Path,
) -> None:
    """Parse pages, run layout + OCR, and persist segments using providers.

    Uses provider resolution from settings to obtain pager/layout/ocr. Inserts
    `pages`, `text_segments` (paragraphs with bbox), and `figures` for figure
    regions; stubs `table_sets` for table regions. Enqueues embeddings for new
    text segments at the end.
    """
    from src.app.db.session.session_async import AsyncSessionLocal
    from src.app.settings import get_settings

    # Normalize input
    pdf_path = Path(pdf_path)
    pager, layout, ocr = resolve_providers()
    settings = get_settings()

    # 1) Enumerate pages
    page_infos: List[PDFPageInfo] = await pager.pages(pdf_path=pdf_path, max_pages=settings.pdf_max_pages)
    logger.info("Processing container %s: %d pages detected", container_id, len(page_infos))

    # 2) Persist pages if not present and export page images
    artifacts_dir = _Path(get_settings().artifacts_base_dir) / "containers" / str(container_id) / "pages"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fmt = settings.artifacts_image_format.lower()
    qual = settings.artifacts_image_quality
    dpi = settings.pdf_render_dpi

    def _render_page_image(pdf_path: Path, page_no: int, dpi: int) -> Image.Image:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        page = doc[page_no - 1]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

    async with AsyncSessionLocal() as session:
        for p in page_infos:
            page = await session.get(Page, (container_id, p.page_no))
            if page is None:
                page = Page(container_id=container_id, page_no=p.page_no, width_px=p.width_px, height_px=p.height_px)
                session.add(page)
                await session.flush()
            # Export page image
            try:
                img = _render_page_image(pdf_path, p.page_no, dpi)
                filename = f"page-{p.page_no}-{dpi}dpi.{fmt}"
                out_path = artifacts_dir / filename
                save_kwargs = {}
                if fmt == "webp":
                    save_kwargs = {"quality": qual, "method": 6}
                elif fmt in ("jpeg", "jpg"):
                    save_kwargs = {"quality": qual}
                img.save(str(out_path), format=fmt.upper() if fmt != "jpg" else "JPEG", **save_kwargs)
                page.image_uri = f"file://{out_path}"
                logger.debug("Saved page %d render → %s", p.page_no, page.image_uri)
            except Exception:
                # Ignore rendering failures; keep dims only
                logger.warning("Failed to render page %d at %ddpi", p.page_no, dpi, exc_info=True)
        await session.commit()

    # 3) Layout + OCR/vector-fusion per page, then persist segments
    segments_created = 0
    figures_created = 0
    tables_created = 0
    page_text_map: dict[int, List[str]] = {}
    async with AsyncSessionLocal() as session:
        for p in page_infos:
            regions: List[LayoutRegion] = await layout.detect(pdf_path=pdf_path, page_no=p.page_no)
            # Sort regions roughly by reading order (top-to-bottom, then left-to-right)
            try:
                regions.sort(key=lambda rr: (rr.bbox.y0, rr.bbox.x0))
            except Exception:
                pass
            logger.debug("Page %d: %d regions detected (sample classes: %s)", p.page_no, len(regions), list({r.rtype for r in regions})[:5])
            # Compute simple signals to route (placeholder heuristic for now)
            # Debug: compare raw vs filtered spans
            raw_span_count = None
            try:
                import fitz  # type: ignore
                _doc = fitz.open(str(pdf_path))
                _page = _doc[p.page_no - 1]
                _rect = _page.rect
                _raw = _page.get_text("rawdict") or {}
                _blocks = _raw.get("blocks", [])
                _raw_spans = []
                for _b in _blocks:
                    if _b.get("type", 0) != 0:
                        continue
                    for _ln in _b.get("lines", []):
                        for _sp in _ln.get("spans", []):
                            # keep only spans that have any visible text (for signal)
                            if (_sp.get("text") or "").strip():
                                _raw_spans.append(_sp)
                raw_span_count = len(_raw_spans)
                logger.debug(
                    "Page %d debug: raw_spans=%d text_sample=%r",
                    p.page_no,
                    raw_span_count,
                    (_page.get_text("text") or "")[:200].replace("\n", " "),
                )
            except Exception:
                logger.debug("Page %d debug: raw span inspection failed", p.page_no, exc_info=True)

            try:
                spans = await extract_text_spans(pdf_path, p.page_no)
                logger.debug(
                    "Page %d debug: filtered_spans=%d norm_samples=%s",
                    p.page_no,
                    len(spans),
                    [
                        (
                            round(s.bbox.x0, 3),
                            round(s.bbox.y0, 3),
                            round(s.bbox.x1, 3),
                            round(s.bbox.y1, 3),
                            (s.text or "")[:40],
                        )
                        for s in spans[:3]
                    ],
                )
                # Approximate text coverage by union-less sum of span areas (overestimates)
                text_cov = 0.0
                for s in spans:
                    text_cov += max(0.0, (s.bbox.x1 - s.bbox.x0) * (s.bbox.y1 - s.bbox.y0))
                text_cov = min(1.0, text_cov)
            except Exception:
                spans = []
                text_cov = 0.0
            # image coverage unknown without XObject parsing; default from regions
            img_cov = 0.0
            for r in regions:
                if r.rtype == "figure":
                    img_cov += max(0.0, (r.bbox.x1 - r.bbox.x0) * (r.bbox.y1 - r.bbox.y0))
            img_cov = min(1.0, img_cov)
            # Optional warning if raw spans exist but our filter produced none
            if raw_span_count is not None and raw_span_count > 0 and not spans:
                logger.warning(
                    "Page %d: PyMuPDF raw spans=%d but filtered_spans=0 (check normalization/filters)",
                    p.page_no,
                    raw_span_count,
                )
            sandwich = img_cov * (1.0 - min(1.0, text_cov * 2))
            route = route_page(PageSignals(text_coverage=text_cov, image_coverage=img_cov, sandwich_score=sandwich))
            logger.info(
                "Page %d: route=%s text_cov=%.3f image_cov=%.3f sandwich=%.3f spans=%d regions=%d",
                p.page_no, route, text_cov, img_cov, sandwich, len(spans), len(regions)
            )

            # Persist per-page analysis/routing
            try:
                pa = await session.get(PageAnalysis, (container_id, p.page_no))
                if pa is None:
                    pa = PageAnalysis(
                        container_id=container_id,
                        page_no=p.page_no,
                        route=route,
                        text_coverage=text_cov,
                        image_coverage=img_cov,
                        sandwich_score=sandwich,
                        version="v1",
                    )
                    session.add(pa)
                else:
                    pa.route = route
                    pa.text_coverage = text_cov
                    pa.image_coverage = img_cov
                    pa.sandwich_score = sandwich
                    pa.version = "v1"
            except Exception:
                pass

            page_parts: List[str] = []
            for idx, r in enumerate(regions):
                rtype = (r.rtype or "").lower()
                if rtype == "text":
                    text = ""
                    source_kind: Optional[str] = None
                    # Prefer vector text on digital/hybrid pages
                    if spans and route in ("digital", "hybrid"):
                        # simple spatial join: take spans whose centers lie within region bbox
                        cx0, cy0, cx1, cy1 = r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1
                        selected = []
                        for s in spans:
                            cx = (s.bbox.x0 + s.bbox.x1) / 2.0
                            cy = (s.bbox.y0 + s.bbox.y1) / 2.0
                            if cx0 <= cx <= cx1 and cy0 <= cy <= cy1:
                                selected.append(s)
                        if selected:
                            # order by top-to-bottom, then left-to-right
                            selected.sort(key=lambda s: (s.bbox.y0, s.bbox.x0))
                            text = " ".join(s.text for s in selected).strip()
                            source_kind = "vector"
                            logger.debug("Page %d zone %d(text): using vector spans=%d len=%d", p.page_no, idx, len(selected), len(text))
                    # Fallback to OCR if no vector text or OCR path
                    if not text and route in ("ocr", "hybrid"):
                        text = await ocr.ocr_region(pdf_path=pdf_path, page_no=p.page_no, region=r)
                        if text:
                            source_kind = "ocr"
                            logger.debug("Page %d zone %d(text): using OCR len=%d", p.page_no, idx, len(text))
                    if text and text.strip():
                        bbox = r.bbox.model_dump()  # JSON for DB
                        await ingest_text_segment(
                            session,
                            container_id=container_id,
                            page_no=p.page_no,
                            object_type="paragraph",
                            text=text.strip(),
                            bbox=bbox,
                            text_source=source_kind,
                        )
                        segments_created += 1
                        page_parts.append(text.strip())
                elif rtype == "figure":
                    # Create Figure and crop image region for artifacts
                    fig = Figure(container_id=container_id, page_no=p.page_no, bbox=r.bbox.model_dump(), image_uri=None)
                    session.add(fig)
                    await session.flush()
                    try:
                        # Render page (or open saved) and crop bbox
                        img = _render_page_image(pdf_path, p.page_no, dpi)
                        W, H = img.size
                        x0 = int(max(0, min(W, r.bbox.x0 * W)))
                        y0 = int(max(0, min(H, r.bbox.y0 * H)))
                        x1 = int(max(0, min(W, r.bbox.x1 * W)))
                        y1 = int(max(0, min(H, r.bbox.y1 * H)))
                        if x1 > x0 and y1 > y0:
                            fig_dir = artifacts_dir / f"page-{p.page_no}" / "figures"
                            fig_dir.mkdir(parents=True, exist_ok=True)
                            fname = f"figure-{fig.figure_id}.{fmt}"
                            out_path = fig_dir / fname
                            crop = img.crop((x0, y0, x1, y1))
                            save_kwargs = {}
                            if fmt == "webp":
                                save_kwargs = {"quality": qual, "method": 6}
                            elif fmt in ("jpeg", "jpg"):
                                save_kwargs = {"quality": qual}
                            crop.save(str(out_path), format=fmt.upper() if fmt != "jpg" else "JPEG", **save_kwargs)
                            fig.image_uri = f"file://{out_path}"
                            logger.debug("Page %d zone %d(figure): saved crop → %s", p.page_no, idx, fig.image_uri)
                    except Exception:
                        logger.warning("Page %d zone %d(figure): crop failed", p.page_no, idx, exc_info=True)
                    figures_created += 1
                elif rtype == "table":
                    # Create a stub TableSet; row extraction not yet implemented
                    session.add(TableSet(container_id=container_id, name=f"table-{p.page_no}-{idx}", page_no=p.page_no, bbox=r.bbox.model_dump()))
                    tables_created += 1
                else:
                    continue
            page_text_map[p.page_no] = page_parts
        await session.commit()

    # 4) Update pages.text with fused reading-order text per page
    async with AsyncSessionLocal() as session:
        for p in page_infos:
            parts = page_text_map.get(p.page_no, [])
            try:
                page = await session.get(Page, (container_id, p.page_no))
                if page is not None:
                    page.text = "\n\n".join([s for s in parts if s]) or None
            except Exception:
                logger.warning("Failed to update pages.text for page %d", p.page_no, exc_info=True)
        await session.commit()

    # 5) Detect references and populate bibliography_entries + citation_anchors
    try:
        # Determine first page containing a References heading
        ref_start: Optional[int] = None
        for p in page_infos:
            parts = page_text_map.get(p.page_no, [])
            joined = "\n".join(parts)
            # Heuristic: a line equals 'references' (case-insensitive)
            lines = [ln.strip() for ln in joined.splitlines()]
            if any(ln.lower() == "references" for ln in lines):
                ref_start = p.page_no
                break
        bib_map: dict[str, UUID] = {}

        async with AsyncSessionLocal() as session:
            # Parse bibliography entries
            if ref_start is not None:
                logger.info("Detected References section starting at page %d", ref_start)
                entry_pat = re.compile(r"^\s*\[(\d+)\]\s*(.+)")
                current_label: Optional[str] = None
                current_text: List[str] = []
                def flush_entry():
                    nonlocal current_label, current_text
                    if current_label and current_text:
                        raw = " ".join([t.strip() for t in current_text]).strip()
                        be = BibliographyEntry(container_id=container_id, label=current_label, raw_text=raw)
                        session.add(be)
                        # Need to flush to get bib_id for mapping
                        return be
                    return None

                for p in page_infos:
                    if p.page_no < ref_start:
                        continue
                    parts = page_text_map.get(p.page_no, [])
                    for ln in "\n".join(parts).splitlines():
                        m = entry_pat.match(ln)
                        if m:
                            # New entry begins; flush previous
                            be = flush_entry()
                            if be is not None:
                                await session.flush()
                                bib_map[current_label] = be.bib_id  # type: ignore[index]
                            current_label = m.group(1)
                            current_text = [m.group(2)]
                        else:
                            if current_label is not None:
                                current_text.append(ln)
                # Flush last
                be = flush_entry()
                if be is not None:
                    await session.flush()
                    bib_map[current_label] = be.bib_id  # type: ignore[index]
                await session.commit()

            # Create in-text citation anchors on non-reference pages
            cite_pat = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
            for p in page_infos:
                if ref_start is not None and p.page_no >= ref_start:
                    continue
                page_text = "\n".join(page_text_map.get(p.page_no, []))
                pos = 0
                for m in cite_pat.finditer(page_text):
                    marker = m.group(0)
                    nums = re.split(r"\s*,\s*", m.group(1))
                    for n in nums:
                        tgt = bib_map.get(n)
                        ca = CitationAnchor(
                            container_id=container_id,
                            page_no=p.page_no,
                            char_offset=int(m.start()),
                            marker=marker,
                            target_bib=tgt,
                        )
                        session.add(ca)
                # commit in batches per page
            await session.commit()
            logger.info("Citations: %d bibliography entries, anchors created across pages", len(bib_map))
    except Exception:
        logger.warning("Citation parsing failed", exc_info=True)

    # 6) Enqueue embeddings for any new text segments under this container (respect setting)
    if settings.ingest_embed_on_ingest:
        logger.info("Enqueuing text embeddings for container %s (settings enabled)", container_id)
        await embed_container_segments_async(container_id)
    logger.info(
        "Completed processing container %s: segments=%d figures=%d tables=%d",
        container_id, segments_created, figures_created, tables_created
    )

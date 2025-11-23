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
import time
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any
import re
from uuid import UUID
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.app.api.v1.ingest.schemas import IngestOptions, IngestResponse
from src.app.db.models.models import Container, Page, Figure, TableSet, PageAnalysis, BibliographyEntry, CitationAnchor
from src.app.domain.common import BBox
from src.app.services.embeddings.embed_runner_async import embed_container_segments_async
from .providers import resolve_providers
from .providers.base import PDFPageInfo, LayoutRegion
from .providers.pdf_pager import extract_text_spans
from .text_pipeline import ingest_text_segment
from .page_router import route_page, PageSignals
from src.app.settings import get_settings
from pathlib import Path as _Path
import json
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
    import os

    pipeline_start = time.perf_counter()
    
    # Normalize input
    pdf_path = Path(pdf_path)
    pager, layout, ocr = resolve_providers()
    settings = get_settings()
    
    # Set PyTorch thread defaults if configured (0 means use PyTorch defaults)
    if settings.torch_num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(settings.torch_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(settings.torch_num_threads)
        os.environ["TORCH_NUM_THREADS"] = str(settings.torch_num_threads)
        logger.info(f"Set PyTorch threads to {settings.torch_num_threads}")
    else:
        logger.info("Using PyTorch default thread settings")

    # 1) Enumerate pages
    stage_start = time.perf_counter()
    page_infos: List[PDFPageInfo] = await pager.pages(pdf_path=pdf_path, max_pages=settings.pdf_max_pages)
    logger.info("Processing container %s: %d pages detected", container_id, len(page_infos))
    logger.info("[PROFILE] Page enumeration: %.3fs", time.perf_counter() - stage_start)

    # 2) Prepare page image rendering outside of DB context
    artifacts_dir = _Path(get_settings().artifacts_base_dir) / "containers" / str(container_id) / "pages"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fmt = settings.artifacts_image_format.lower()
    qual = settings.artifacts_image_quality
    dpi = settings.pdf_render_dpi

    def _render_page_image(pdf_path: Path, page_no: int, dpi: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Render a single page and save it. Returns (page_no, result_dict)."""
        import fitz  # PyMuPDF
        try:
            doc = fitz.open(str(pdf_path))
            page = doc[page_no - 1]
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()  # Important: close the document
            
            # Save the image
            filename = f"page-{page_no}-{dpi}dpi.{fmt}"
            out_path = artifacts_dir / filename
            save_kwargs = {}
            if fmt == "webp":
                save_kwargs = {"quality": qual, "method": 6}
            elif fmt in ("jpeg", "jpg"):
                save_kwargs = {"quality": qual}
            img.save(str(out_path), format=fmt.upper() if fmt != "jpg" else "JPEG", **save_kwargs)
            
            result = {
                "image": img,
                "uri": f"file://{out_path}"
            }
            logger.debug("Rendered page %d → %s", page_no, result["uri"])
            return page_no, result
        except Exception:
            logger.warning("Failed to render page %d at %ddpi", page_no, dpi, exc_info=True)
            return page_no, None

    # Render all pages concurrently (outside of DB transaction)
    stage_start = time.perf_counter()
    page_images = {}
    
    # Use ThreadPoolExecutor for concurrent page rendering
    max_workers = min(settings.pdf_render_max_workers, len(page_infos))
    logger.info("Starting concurrent page rendering with %d workers for %d pages", max_workers, len(page_infos))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all page rendering tasks
        future_to_page = {
            executor.submit(_render_page_image, pdf_path, p.page_no, dpi): p
            for p in page_infos
        }
        
        # Collect results as they complete
        import concurrent.futures
        for future in concurrent.futures.as_completed(future_to_page):
            page_no, result = future.result()
            page_images[page_no] = result
    
    logger.info("[PROFILE] Concurrent page rendering (%d pages): %.3fs", len(page_infos), time.perf_counter() - stage_start)

    # 3) Persist pages with minimal DB transaction time
    async with AsyncSessionLocal() as session:
        for p in page_infos:
            page = await session.get(Page, (container_id, p.page_no))
            if page is None:
                page = Page(container_id=container_id, page_no=p.page_no, width_px=p.width_px, height_px=p.height_px)
                session.add(page)
            
            # Add image URI if rendering succeeded
            if page_images.get(p.page_no):
                page.image_uri = page_images[p.page_no]["uri"]
        
        await session.commit()

    # 4) Process layout + OCR per page (heavy processing outside DB transactions)
    stage_start = time.perf_counter()
    segments_data = []
    figures_data = []
    tables_data = []
    page_text_map: dict[int, List[str]] = {}
    segments_created = 0
    figures_created = 0
    tables_created = 0
    
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

        # Store page analysis data
        page_analysis_data = {
            "container_id": container_id,
            "page_no": p.page_no,
            "route": route,
            "text_coverage": text_cov,
            "image_coverage": img_cov,
            "sandwich_score": sandwich,
            "version": "v1",
        }

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
                    segments_data.append({
                        "container_id": container_id,
                        "page_no": p.page_no,
                        "object_type": "paragraph",
                        "text": text.strip(),
                        "bbox": bbox,
                        "text_source": source_kind,
                    })
                    page_parts.append(text.strip())
            elif rtype == "figure":
                # Process figure
                bbox_dict = r.bbox.model_dump()
                figure_data = {
                    "container_id": container_id,
                    "page_no": p.page_no,
                    "bbox": bbox_dict,
                    "idx": idx,
                }
                
                # Crop figure image if page was rendered
                if page_images.get(p.page_no):
                    try:
                        img = page_images[p.page_no]["image"]
                        W, H = img.size
                        x0 = int(max(0, min(W, r.bbox.x0 * W)))
                        y0 = int(max(0, min(H, r.bbox.y0 * H)))
                        x1 = int(max(0, min(W, r.bbox.x1 * W)))
                        y1 = int(max(0, min(H, r.bbox.y1 * H)))
                        if x1 > x0 and y1 > y0:
                            fig_dir = artifacts_dir / f"page-{p.page_no}" / "figures"
                            fig_dir.mkdir(parents=True, exist_ok=True)
                            crop = img.crop((x0, y0, x1, y1))
                            figure_data["crop_image"] = crop
                            figure_data["fig_dir"] = fig_dir
                    except Exception:
                        logger.warning("Failed to crop figure on page %d", p.page_no, exc_info=True)
                
                figures_data.append(figure_data)
            elif rtype == "table":
                # Process table - store data for later
                tables_data.append({
                    "container_id": container_id,
                    "page_no": p.page_no,
                    "idx": idx,
                    "bbox": r.bbox,
                    "page_image": page_images.get(p.page_no),
                })
            else:
                continue
        
        page_text_map[p.page_no] = page_parts
        
        # Persist page analysis
        async with AsyncSessionLocal() as session:
            try:
                pa = await session.get(PageAnalysis, (container_id, p.page_no))
                if pa is None:
                    pa = PageAnalysis(**page_analysis_data)
                    session.add(pa)
                else:
                    for k, v in page_analysis_data.items():
                        if k not in ("container_id", "page_no"):
                            setattr(pa, k, v)
                await session.commit()
            except Exception:
                logger.warning("Failed to persist page analysis for page %d", p.page_no, exc_info=True)
    
    logger.info("[PROFILE] Page processing (layout + OCR + extraction): %.3fs", time.perf_counter() - stage_start)

    # 5) Persist all segments in batches
    stage_start = time.perf_counter()
    segments_created = 0
    if segments_data:
        async with AsyncSessionLocal() as session:
            for seg_data in segments_data:
                await ingest_text_segment(session, **seg_data)
                segments_created += 1
            await session.commit()
    logger.info("[PROFILE] Segments persistence (%d segments): %.3fs", segments_created, time.perf_counter() - stage_start)
    
    # 6) Persist figures with optional DePlot processing
    stage_start = time.perf_counter()
    figures_created = 0
    if figures_data:
        async with AsyncSessionLocal() as session:
            for fig_data in figures_data:
                fig = Figure(
                    container_id=fig_data["container_id"],
                    page_no=fig_data["page_no"],
                    bbox=fig_data["bbox"],
                    image_uri=None
                )
                session.add(fig)
                await session.flush()
                
                # Save cropped figure image
                if "crop_image" in fig_data and "fig_dir" in fig_data:
                    try:
                        fname = f"figure-{fig.figure_id}.{fmt}"
                        out_path = fig_data["fig_dir"] / fname
                        save_kwargs = {}
                        if fmt == "webp":
                            save_kwargs = {"quality": qual, "method": 6}
                        elif fmt in ("jpeg", "jpg"):
                            save_kwargs = {"quality": qual}
                        fig_data["crop_image"].save(str(out_path), format=fmt.upper() if fmt != "jpg" else "JPEG", **save_kwargs)
                        fig.image_uri = f"file://{out_path}"
                        logger.debug("Saved figure crop → %s", fig.image_uri)
                        
                        # Optional DePlot processing (simplified for now)
                        st = get_settings()
                        if st.figure_enable_deplot:
                            logger.debug("DePlot processing skipped in fixed version")
                    except Exception:
                        logger.warning("Failed to save figure image", exc_info=True)
                
                figures_created += 1
            await session.commit()
    logger.info("[PROFILE] Figures persistence (%d figures): %.3fs", figures_created, time.perf_counter() - stage_start)
    
    # 7) Process tables
    stage_start = time.perf_counter()
    tables_created = 0
    if tables_data and settings.table_enable_structure:
        async with AsyncSessionLocal() as session:
            for table_data in tables_data:
                try:
                    # Process table with structure detection
                    img = None
                    if table_data["page_image"]:
                        img = table_data["page_image"]["image"]
                    
                    if img:
                        W, H = img.size
                        r = table_data["bbox"]
                        tx0 = int(max(0, min(W, r.x0 * W)))
                        ty0 = int(max(0, min(H, r.y0 * H)))
                        tx1 = int(max(0, min(W, r.x1 * W)))
                        ty1 = int(max(0, min(H, r.y1 * H)))
                        crop = img.crop((tx0, ty0, tx1, ty1))
                        
                        from .providers.cached_providers import get_table_structure_extractor
                        tstruct = get_table_structure_extractor()
                        cells = await tstruct.detect_cells(image=crop)
                        
                        tset = TableSet(
                            container_id=table_data["container_id"],
                            name=f"table-{table_data['page_no']}-{table_data['idx']}",
                            page_no=table_data["page_no"],
                            bbox=table_data["bbox"].model_dump(),
                            n_rows=0,
                            n_cols=0,
                        )
                        session.add(tset)
                        tables_created += 1
                    else:
                        # Create placeholder
                        session.add(
                            TableSet(
                                container_id=table_data["container_id"],
                                name=f"table-{table_data['page_no']}-{table_data['idx']}",
                                page_no=table_data["page_no"],
                                bbox=table_data["bbox"].model_dump(),
                            )
                        )
                        tables_created += 1
                except Exception:
                    logger.warning("Failed to process table", exc_info=True)
            await session.commit()
    logger.info("[PROFILE] Tables persistence (%d tables): %.3fs", tables_created, time.perf_counter() - stage_start)

    # 8) Update pages.text with fused reading-order text per page
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

    # 9) Detect references and populate bibliography_entries + citation_anchors
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

    # 10) Enqueue embeddings for any new text segments under this container (respect setting)
    if settings.ingest_embed_on_ingest:
        logger.info("Enqueuing text embeddings for container %s (settings enabled)", container_id)
        await embed_container_segments_async(container_id)
    
    # Log overall pipeline timing
    total_time = time.perf_counter() - pipeline_start
    logger.info(
        "Completed processing container %s: segments=%d figures=%d tables=%d",
        container_id, segments_created, figures_created, tables_created
    )
    logger.info("[PROFILE] Total pipeline execution time: %.3fs", total_time)

"""Microbatched version of container pipeline with GPU lanes for streaming."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from uuid import UUID
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from src.app.services.ingest.providers.base import PDFPageInfo, LayoutRegion
from src.app.db.models.models import Container, Page, Figure, TableSet, PageAnalysis
from src.app.domain.common import BBox
from src.app.settings import get_settings
from .gpu_lanes import LayoutGPULane, OCRGPULane, TableGPULane

logger = logging.getLogger(__name__)


async def render_page_worker(
    page_info: PDFPageInfo,
    pdf_path: Path,
    layout_queue: asyncio.Queue,
    artifacts_dir: Path,
    settings
):
    """Render a single page and send to layout queue."""
    import fitz
    from PIL import Image
    
    try:
        # Render page
        doc = fitz.open(str(pdf_path))
        page = doc[page_info.page_no - 1]
        zoom = settings.pdf_render_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        
        # Save to disk
        fmt = settings.artifacts_image_format.lower()
        filename = f"page-{page_info.page_no}-{settings.pdf_render_dpi}dpi.{fmt}"
        out_path = artifacts_dir / filename
        save_kwargs = {}
        if fmt == "webp":
            save_kwargs = {"quality": settings.artifacts_image_quality, "method": 6}
        elif fmt in ("jpeg", "jpg"):
            save_kwargs = {"quality": settings.artifacts_image_quality}
        img.save(str(out_path), format=fmt.upper() if fmt != "jpg" else "JPEG", **save_kwargs)
        
        # Send to layout queue
        await layout_queue.put({
            'image': img,
            'page_info': page_info,
            'image_uri': f"file://{out_path}",
            'metadata': {'page_no': page_info.page_no}
        })
        
        logger.debug(f"Rendered page {page_info.page_no}")
        
    except Exception as e:
        logger.error(f"Failed to render page {page_info.page_no}: {e}", exc_info=True)
        # Could put error marker in queue


async def process_pdf_container_microbatched(
    *,
    container_id: UUID,
    pdf_path: Path,
    use_microbatching: bool = True
) -> None:
    """Microbatched version of PDF processing pipeline with GPU lanes."""
    if not use_microbatching:
        # Fall back to original processing
        from .container_pipeline import process_pdf_container_async
        return await process_pdf_container_async(container_id=container_id, pdf_path=pdf_path)
    
    # Imports
    from src.app.db.session.session_async import AsyncSessionLocal
    from src.app.services.ingest.providers import resolve_providers
    from src.app.services.ingest.providers.batched_providers import get_batched_providers
    from src.app.services.ingest.page_router import route_page, PageSignals
    from src.app.services.ingest.providers.pdf_pager import extract_text_spans
    from src.app.services.ingest.text_pipeline import ingest_text_segment
    from src.app.services.embeddings.embed_runner_async import embed_container_segments_async
    import os
    
    pipeline_start = time.perf_counter()
    
    # Initialize
    pdf_path = Path(pdf_path)
    settings = get_settings()
    
    # Create container in database first
    async with AsyncSessionLocal() as session:
        import hashlib
        from src.app.db.models.models import Container
        
        # Calculate file hash
        data = pdf_path.read_bytes()
        digest = hashlib.sha256(data).digest()
        
        # Create container
        container = Container(
            container_id=container_id,
            source_uri=str(pdf_path),
            mime_type="application/pdf",
            sha256=digest,
            title=pdf_path.stem,
            is_scanned=False,
        )
        session.add(container)
        await session.commit()
        logger.info(f"Created PDF container {container_id} for {pdf_path}")
    
    # Get cached providers
    pager, _, _ = resolve_providers()
    layout_model, ocr_model, table_model = get_batched_providers()
    
    # Pre-load models
    logger.info("Pre-loading models...")
    layout_model._ensure_model()
    ocr_model._ensure_model()
    table_model._ensure_model()
    
    # Set PyTorch threads
    if settings.torch_num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(settings.torch_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(settings.torch_num_threads)
        os.environ["TORCH_NUM_THREADS"] = str(settings.torch_num_threads)
    
    # 1) Enumerate pages
    stage_start = time.perf_counter()
    page_infos: List[PDFPageInfo] = await pager.pages(pdf_path=pdf_path, max_pages=settings.pdf_max_pages)
    logger.info("Processing container %s: %d pages detected", container_id, len(page_infos))
    logger.info("[PROFILE] Page enumeration: %.3fs", time.perf_counter() - stage_start)
    
    # Setup queues
    layout_queue = asyncio.Queue(maxsize=getattr(settings, 'layout_queue_max', 64))
    ocr_queue = asyncio.Queue(maxsize=getattr(settings, 'ocr_queue_max', 64))
    results_queue = asyncio.Queue(maxsize=getattr(settings, 'results_queue_max', 128))
    
    # Setup executors
    cpu_pool = ThreadPoolExecutor(
        max_workers=settings.pdf_render_max_workers,
        thread_name_prefix="cpu"
    )
    
    # Create GPU lanes
    layout_lane = LayoutGPULane(
        model=layout_model,
        pdf_path=pdf_path,
        input_queue=layout_queue,
        output_queue=ocr_queue,
        max_items=getattr(settings, 'microbatch_max_items', 8),
        max_bytes=getattr(settings, 'microbatch_max_bytes', 256 * 1024 * 1024),
        max_latency_ms=getattr(settings, 'microbatch_max_latency_ms', 50)
    )
    
    ocr_lane = OCRGPULane(
        model=ocr_model,
        pdf_path=pdf_path,
        input_queue=ocr_queue,
        output_queue=results_queue,
        max_items=getattr(settings, 'microbatch_max_items', 8),
        max_bytes=getattr(settings, 'microbatch_max_bytes', 256 * 1024 * 1024),
        max_latency_ms=getattr(settings, 'microbatch_max_latency_ms', 50)
    )
    
    # Start GPU lanes
    await layout_lane.start()
    await ocr_lane.start()
    
    # Create artifacts directory
    artifacts_dir = Path(settings.artifacts_base_dir) / "containers" / str(container_id) / "pages"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Results collector
    page_results = defaultdict(dict)
    
    async def results_collector():
        """Collect results from the pipeline."""
        while True:
            result = await results_queue.get()
            if result is None:  # Sentinel
                results_queue.task_done()
                break
                
            page_no = result['page_no']
            page_results[page_no] = result
            results_queue.task_done()
    
    # Start results collector
    collector_task = asyncio.create_task(results_collector())
    
    # 2) Start page rendering workers
    stage_start = time.perf_counter()
    render_tasks = []
    
    for page_info in page_infos:
        task = asyncio.create_task(
            render_page_worker(
                page_info=page_info,
                pdf_path=pdf_path,
                layout_queue=layout_queue,
                artifacts_dir=artifacts_dir,
                settings=settings
            )
        )
        render_tasks.append(task)
    
    # Wait for all pages to be rendered and queued
    await asyncio.gather(*render_tasks)
    
    logger.info("[PROFILE] Concurrent page rendering: %.3fs", time.perf_counter() - stage_start)
    
    # 3) Send sentinel to indicate no more pages coming
    await layout_queue.put(None)
    
    # 4) Wait for pipeline to complete
    pipeline_start_time = time.perf_counter()
    
    # Wait for lanes to finish processing (do not call stop() yet)
    await layout_lane._task  # Wait for layout lane to finish
    await ocr_lane._task     # Wait for OCR lane to finish
    
    # Send sentinel to results collector
    await results_queue.put(None)
    await collector_task
    
    logger.info("[PROFILE] GPU pipeline execution: %.3fs", time.perf_counter() - pipeline_start_time)
    
    # 4) Process results and persist to database
    stage_start = time.perf_counter()
    
    # Persist pages
    async with AsyncSessionLocal() as session:
        for page_info in page_infos:
            page = await session.get(Page, (container_id, page_info.page_no))
            if page is None:
                page = Page(
                    container_id=container_id,
                    page_no=page_info.page_no,
                    width_px=page_info.width_px,
                    height_px=page_info.height_px
                )
                session.add(page)
            
            # Set image URI if available
            if page_info.page_no in page_results:
                result = page_results[page_info.page_no]
                if 'image_uri' in result:
                    page.image_uri = result['image_uri']
        
        await session.commit()
    
    # Process segments, figures, tables
    segments_created = 0
    figures_created = 0
    tables_created = 0
    
    async with AsyncSessionLocal() as session:
        for page_no, result in page_results.items():
            regions = result.get('regions', [])
            ocr_results = result.get('ocr_results', {})
            
            # Process text segments
            page_texts = []
            for idx, region in enumerate(regions):
                if region.rtype == "text":
                    text = ocr_results.get(idx, "")
                    if text and text.strip():
                        await ingest_text_segment(
                            session,
                            container_id=container_id,
                            page_no=page_no,
                            object_type="paragraph",
                            text=text.strip(),
                            bbox=region.bbox.model_dump(),
                            text_source="ocr",
                        )
                        segments_created += 1
                        page_texts.append((region.bbox.y0, region.bbox.x0, text.strip()))
                        
                elif region.rtype == "figure":
                    fig = Figure(
                        container_id=container_id,
                        page_no=page_no,
                        bbox=region.bbox.model_dump(),
                    )
                    session.add(fig)
                    figures_created += 1
                    
                elif region.rtype == "table":
                    tset = TableSet(
                        container_id=container_id,
                        name=f"table-{page_no}-{idx}",
                        page_no=page_no,
                        bbox=region.bbox.model_dump(),
                        n_rows=0,
                        n_cols=0,
                    )
                    session.add(tset)
                    tables_created += 1
            
            # Update page text
            if page_texts:
                page = await session.get(Page, (container_id, page_no))
                if page is not None:
                    # Sort by position
                    page_texts.sort(key=lambda x: (x[0], x[1]))
                    page.text = "\n\n".join(t[2] for t in page_texts)
        
        await session.commit()
    
    logger.info("[PROFILE] Results persistence: %.3fs", time.perf_counter() - stage_start)
    
    # 5) Enqueue embeddings if enabled
    if settings.ingest_embed_on_ingest:
        logger.info("Enqueuing text embeddings for container %s", container_id)
        await embed_container_segments_async(container_id)
    
    # Log overall timing
    total_time = time.perf_counter() - pipeline_start
    logger.info(
        "Completed MICROBATCHED processing container %s: segments=%d figures=%d tables=%d in %.3fs",
        container_id, segments_created, figures_created, tables_created, total_time
    )
    
    # Cleanup
    cpu_pool.shutdown(wait=True)

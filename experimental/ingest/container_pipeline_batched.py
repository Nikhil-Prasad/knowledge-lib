"""Batched version of container pipeline for GPU efficiency."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from uuid import UUID
from collections import defaultdict

from src.app.services.ingest.providers.base import PDFPageInfo, LayoutRegion
from src.app.db.models.models import Container, Page, Figure, TableSet, PageAnalysis
from src.app.domain.common import BBox
from src.app.settings import get_settings

logger = logging.getLogger(__name__)


class BatchedPipelineProcessor:
    """Processes PDF pages in batches for GPU efficiency."""
    
    def __init__(self):
        settings = get_settings()
        self.layout_batch_size = getattr(settings, 'layout_batch_size', 8)
        self.ocr_batch_size = getattr(settings, 'ocr_batch_size', 16)
        self.table_batch_size = getattr(settings, 'table_batch_size', 8)
        
    async def process_layouts_batched(
        self,
        layout_detector,
        pdf_path: Path,
        page_infos: List[PDFPageInfo],
        page_images: Dict[int, Dict[str, Any]]
    ) -> Dict[int, List[LayoutRegion]]:
        """Process layout detection in batches."""
        logger.info("Starting batched layout detection for %d pages (batch_size=%d)", 
                    len(page_infos), self.layout_batch_size)
        
        all_layouts = {}
        
        # Process pages in batches
        for i in range(0, len(page_infos), self.layout_batch_size):
            batch = page_infos[i:i + self.layout_batch_size]
            batch_start = time.perf_counter()
            
            # Collect images for batch
            batch_images = []
            batch_page_nos = []
            for p in batch:
                if p.page_no in page_images and page_images[p.page_no]:
                    batch_images.append(page_images[p.page_no]["image"])
                    batch_page_nos.append(p.page_no)
            
            if batch_images:
                # Call batch detection if the provider supports it
                if hasattr(layout_detector, 'detect_batch'):
                    logger.debug("Processing layout batch of %d pages", len(batch_images))
                    batch_results = await layout_detector.detect_batch(
                        images=batch_images,
                        page_nos=batch_page_nos
                    )
                    
                    # Map results back to page numbers
                    for page_no, regions in zip(batch_page_nos, batch_results):
                        all_layouts[page_no] = regions
                else:
                    # Fallback to sequential if batch not supported
                    logger.debug("Batch detection not supported, falling back to sequential")
                    for p in batch:
                        regions = await layout_detector.detect(pdf_path=pdf_path, page_no=p.page_no)
                        all_layouts[p.page_no] = regions
            else:
                # No images for this batch, use sequential
                for p in batch:
                    regions = await layout_detector.detect(pdf_path=pdf_path, page_no=p.page_no)
                    all_layouts[p.page_no] = regions
            
            batch_time = time.perf_counter() - batch_start
            logger.debug("Layout batch %d-%d completed in %.3fs", 
                        i + 1, min(i + self.layout_batch_size, len(page_infos)), batch_time)
        
        return all_layouts
    
    async def process_ocr_batched(
        self,
        ocr_provider,
        pdf_path: Path,
        ocr_tasks: List[Tuple[int, LayoutRegion, str]],  # (page_no, region, route)
    ) -> Dict[Tuple[int, int], str]:  # (page_no, region_idx) -> text
        """Process OCR in batches."""
        if not ocr_tasks:
            return {}
            
        logger.info("Starting batched OCR for %d regions (batch_size=%d)", 
                    len(ocr_tasks), self.ocr_batch_size)
        
        ocr_results = {}
        
        # Group by page for better batching
        page_groups = defaultdict(list)
        for page_no, region, route in ocr_tasks:
            page_groups[page_no].append(region)
        
        # Process in batches
        all_regions = []
        region_map = {}  # Track which region belongs to which page
        
        for page_no, regions in page_groups.items():
            for idx, region in enumerate(regions):
                all_regions.append((page_no, idx, region))
                region_map[len(all_regions) - 1] = (page_no, idx)
        
        for i in range(0, len(all_regions), self.ocr_batch_size):
            batch = all_regions[i:i + self.ocr_batch_size]
            batch_start = time.perf_counter()
            
            if hasattr(ocr_provider, 'ocr_regions_batch'):
                # Use batch OCR if available
                batch_regions = [item[2] for item in batch]
                batch_page_nos = [item[0] for item in batch]
                
                logger.debug("Processing OCR batch of %d regions", len(batch_regions))
                texts = await ocr_provider.ocr_regions_batch(
                    pdf_path=pdf_path,
                    page_nos=batch_page_nos,
                    regions=batch_regions
                )
                
                # Map results back
                for j, text in enumerate(texts):
                    if i + j < len(all_regions):
                        page_no, idx = region_map[i + j]
                        ocr_results[(page_no, idx)] = text
            else:
                # Fallback to sequential
                logger.debug("Batch OCR not supported, falling back to sequential")
                for page_no, idx, region in batch:
                    text = await ocr_provider.ocr_region(
                        pdf_path=pdf_path,
                        page_no=page_no,
                        region=region
                    )
                    ocr_results[(page_no, idx)] = text
            
            batch_time = time.perf_counter() - batch_start
            logger.debug("OCR batch %d-%d completed in %.3fs", 
                        i + 1, min(i + self.ocr_batch_size, len(all_regions)), batch_time)
        
        return ocr_results
    
    async def process_tables_batched(
        self,
        table_extractor,
        table_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process table structure detection in batches."""
        if not table_tasks:
            return []
            
        logger.info("Starting batched table processing for %d tables (batch_size=%d)", 
                    len(table_tasks), self.table_batch_size)
        
        results = []
        
        for i in range(0, len(table_tasks), self.table_batch_size):
            batch = table_tasks[i:i + self.table_batch_size]
            batch_start = time.perf_counter()
            
            # Extract crops for batch
            crops = []
            valid_indices = []
            for j, task in enumerate(batch):
                if task.get("crop_image") is not None:
                    crops.append(task["crop_image"])
                    valid_indices.append(i + j)
            
            if crops and hasattr(table_extractor, 'detect_cells_batch'):
                # Process batch
                logger.debug("Processing table batch of %d tables", len(crops))
                batch_cells = await table_extractor.detect_cells_batch(images=crops)
                
                # Create results with cells
                for idx, cells in zip(valid_indices, batch_cells):
                    task = table_tasks[idx]
                    results.append({
                        **task,
                        "cells": cells
                    })
            else:
                # Fallback to sequential or no crops
                for task in batch:
                    if task.get("crop_image") is not None:
                        cells = await table_extractor.detect_cells(image=task["crop_image"])
                        results.append({**task, "cells": cells})
                    else:
                        results.append(task)
            
            batch_time = time.perf_counter() - batch_start
            logger.debug("Table batch %d-%d completed in %.3fs", 
                        i + 1, min(i + self.table_batch_size, len(table_tasks)), batch_time)
        
        return results


# Helper function to enable batched processing
async def process_pdf_container_batched(
    *,
    container_id: UUID,
    pdf_path: Path,
    use_batching: bool = True
) -> None:
    """Batched version of PDF processing pipeline."""
    from src.app.services.ingest.container_pipeline import process_pdf_container_async
    
    if not use_batching:
        # Fall back to original sequential processing
        return await process_pdf_container_async(container_id=container_id, pdf_path=pdf_path)
    
    # Import here to avoid circular imports
    from src.app.db.session.session_async import AsyncSessionLocal
    from src.app.services.ingest.providers import resolve_providers
    from src.app.services.ingest.providers.batched_providers import get_batched_providers
    from src.app.services.ingest.page_router import route_page, PageSignals
    from src.app.services.ingest.providers.pdf_pager import extract_text_spans
    from src.app.services.ingest.text_pipeline import ingest_text_segment
    from src.app.services.embeddings.embed_runner_async import embed_container_segments_async
    from PIL import Image
    import fitz
    import os
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures
    
    pipeline_start = time.perf_counter()
    
    # Initialize
    pdf_path = Path(pdf_path)
    # Get regular pager but batched layout/ocr/table providers
    pager, _, _ = resolve_providers()
    layout, ocr, table_extractor = get_batched_providers()
    settings = get_settings()
    processor = BatchedPipelineProcessor()
    
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
    
    # 2) Render pages (same as before - already concurrent)
    artifacts_dir = Path(settings.artifacts_base_dir) / "containers" / str(container_id) / "pages"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fmt = settings.artifacts_image_format.lower()
    qual = settings.artifacts_image_quality
    dpi = settings.pdf_render_dpi
    
    def _render_page_image(pdf_path: Path, page_no: int, dpi: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Render a single page and save it. Returns (page_no, result_dict)."""
        try:
            doc = fitz.open(str(pdf_path))
            page = doc[page_no - 1]
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            
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
            return page_no, result
        except Exception:
            logger.warning("Failed to render page %d", page_no, exc_info=True)
            return page_no, None
    
    stage_start = time.perf_counter()
    page_images = {}
    
    max_workers = min(settings.pdf_render_max_workers, len(page_infos))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {
            executor.submit(_render_page_image, pdf_path, p.page_no, dpi): p
            for p in page_infos
        }
        
        for future in concurrent.futures.as_completed(future_to_page):
            page_no, result = future.result()
            page_images[page_no] = result
    
    logger.info("[PROFILE] Concurrent page rendering: %.3fs", time.perf_counter() - stage_start)
    
    # 3) Persist pages
    async with AsyncSessionLocal() as session:
        for p in page_infos:
            page = await session.get(Page, (container_id, p.page_no))
            if page is None:
                page = Page(container_id=container_id, page_no=p.page_no, width_px=p.width_px, height_px=p.height_px)
                session.add(page)
            
            if page_images.get(p.page_no):
                page.image_uri = page_images[p.page_no]["uri"]
        
        await session.commit()
    
    # 4) BATCHED layout detection
    stage_start = time.perf_counter()
    all_layouts = await processor.process_layouts_batched(layout, pdf_path, page_infos, page_images)
    logger.info("[PROFILE] Batched layout detection: %.3fs", time.perf_counter() - stage_start)
    
    # 5) Process regions and prepare tasks
    ocr_tasks = []
    figure_tasks = []
    table_tasks = []
    page_text_map = defaultdict(list)
    
    for p in page_infos:
        regions = all_layouts.get(p.page_no, [])
        
        # Sort regions by reading order
        try:
            regions.sort(key=lambda r: (r.bbox.y0, r.bbox.x0))
        except Exception:
            pass
            
        # Compute page routing
        try:
            spans = await extract_text_spans(pdf_path, p.page_no)
            text_cov = sum(max(0.0, (s.bbox.x1 - s.bbox.x0) * (s.bbox.y1 - s.bbox.y0)) for s in spans)
            text_cov = min(1.0, text_cov)
        except Exception:
            spans = []
            text_cov = 0.0
            
        img_cov = sum(max(0.0, (r.bbox.x1 - r.bbox.x0) * (r.bbox.y1 - r.bbox.y0)) 
                      for r in regions if r.rtype == "figure")
        img_cov = min(1.0, img_cov)
        sandwich = img_cov * (1.0 - min(1.0, text_cov * 2))
        route = route_page(PageSignals(text_coverage=text_cov, image_coverage=img_cov, sandwich_score=sandwich))
        
        logger.debug("Page %d: route=%s regions=%d", p.page_no, route, len(regions))
        
        # Collect tasks for batch processing
        for idx, r in enumerate(regions):
            rtype = (r.rtype or "").lower()
            
            if rtype == "text":
                # Check if we need OCR
                needs_ocr = False
                text_from_vector = ""
                
                if spans and route in ("digital", "hybrid"):
                    # Try vector text first
                    cx0, cy0, cx1, cy1 = r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1
                    selected = []
                    for s in spans:
                        cx = (s.bbox.x0 + s.bbox.x1) / 2.0
                        cy = (s.bbox.y0 + s.bbox.y1) / 2.0
                        if cx0 <= cx <= cx1 and cy0 <= cy <= cy1:
                            selected.append(s)
                    
                    if selected:
                        selected.sort(key=lambda s: (s.bbox.y0, s.bbox.x0))
                        text_from_vector = " ".join(s.text for s in selected).strip()
                
                if not text_from_vector and route in ("ocr", "hybrid"):
                    needs_ocr = True
                    ocr_tasks.append((p.page_no, r, route))
                elif text_from_vector:
                    # Store vector text directly
                    page_text_map[p.page_no].append((idx, text_from_vector, "vector", r.bbox))
                    
            elif rtype == "figure":
                if page_images.get(p.page_no):
                    img = page_images[p.page_no]["image"]
                    figure_tasks.append({
                        "page_no": p.page_no,
                        "idx": idx,
                        "bbox": r.bbox,
                        "image": img,
                    })
                    
            elif rtype == "table":
                if page_images.get(p.page_no):
                    img = page_images[p.page_no]["image"]
                    W, H = img.size
                    x0 = int(max(0, min(W, r.bbox.x0 * W)))
                    y0 = int(max(0, min(H, r.bbox.y0 * H)))
                    x1 = int(max(0, min(W, r.bbox.x1 * W)))
                    y1 = int(max(0, min(H, r.bbox.y1 * H)))
                    
                    if x1 > x0 and y1 > y0:
                        crop = img.crop((x0, y0, x1, y1))
                        table_tasks.append({
                            "page_no": p.page_no,
                            "idx": idx,
                            "bbox": r.bbox,
                            "crop_image": crop,
                        })
    
    # 6) BATCHED OCR processing
    ocr_results = {}
    if ocr_tasks:
        stage_start = time.perf_counter()
        
        # Create indexed tasks for mapping back
        indexed_ocr_tasks = []
        for i, (page_no, region, route) in enumerate(ocr_tasks):
            indexed_ocr_tasks.append((page_no, region, route))
        
        ocr_results = await processor.process_ocr_batched(ocr, pdf_path, indexed_ocr_tasks)
        
        # Add OCR results to page text map
        task_idx = 0
        for page_no, region, route in ocr_tasks:
            if (page_no, task_idx) in ocr_results:
                text = ocr_results[(page_no, task_idx)]
                if text and text.strip():
                    page_text_map[page_no].append((task_idx, text.strip(), "ocr", region.bbox))
            task_idx += 1
        
        logger.info("[PROFILE] Batched OCR processing: %.3fs", time.perf_counter() - stage_start)
    
    # 7) Persist text segments
    stage_start = time.perf_counter()
    segments_created = 0
    
    async with AsyncSessionLocal() as session:
        for page_no, segments in page_text_map.items():
            for idx, text, source, bbox in segments:
                await ingest_text_segment(
                    session,
                    container_id=container_id,
                    page_no=page_no,
                    object_type="paragraph",
                    text=text,
                    bbox=bbox.model_dump(),
                    text_source=source,
                )
                segments_created += 1
        await session.commit()
    
    logger.info("[PROFILE] Segments persistence (%d segments): %.3fs", 
                segments_created, time.perf_counter() - stage_start)
    
    # 8) Process figures
    figures_created = 0
    if figure_tasks:
        # Process figures (no batching for now, but prepared for it)
        async with AsyncSessionLocal() as session:
            for task in figure_tasks:
                fig = Figure(
                    container_id=container_id,
                    page_no=task["page_no"],
                    bbox=task["bbox"].model_dump(),
                )
                session.add(fig)
                figures_created += 1
            await session.commit()
    
    # 9) BATCHED table processing
    tables_created = 0
    if table_tasks and settings.table_enable_structure:
        stage_start = time.perf_counter()
        
        from .providers.cached_providers import get_table_structure_extractor
        table_extractor = get_table_structure_extractor()
        
        processed_tables = await processor.process_tables_batched(table_extractor, table_tasks)
        
        # Persist results
        async with AsyncSessionLocal() as session:
            for table_result in processed_tables:
                tset = TableSet(
                    container_id=container_id,
                    name=f"table-{table_result['page_no']}-{table_result['idx']}",
                    page_no=table_result["page_no"],
                    bbox=table_result["bbox"].model_dump(),
                    n_rows=0,
                    n_cols=0,
                )
                session.add(tset)
                tables_created += 1
            await session.commit()
        
        logger.info("[PROFILE] Batched table processing (%d tables): %.3fs", 
                    tables_created, time.perf_counter() - stage_start)
    
    # 10) Update page text
    async with AsyncSessionLocal() as session:
        for page_no, segments in page_text_map.items():
            # Sort by position for reading order
            segments.sort(key=lambda x: (x[3].y0, x[3].x0))
            parts = [seg[1] for seg in segments]
            
            page = await session.get(Page, (container_id, page_no))
            if page is not None and parts:
                page.text = "\n\n".join(parts)
        await session.commit()
    
    # 11) Enqueue embeddings if enabled
    if settings.ingest_embed_on_ingest:
        logger.info("Enqueuing text embeddings for container %s", container_id)
        await embed_container_segments_async(container_id)
    
    # Log overall timing
    total_time = time.perf_counter() - pipeline_start
    logger.info(
        "Completed BATCHED processing container %s: segments=%d figures=%d tables=%d in %.3fs",
        container_id, segments_created, figures_created, tables_created, total_time
    )

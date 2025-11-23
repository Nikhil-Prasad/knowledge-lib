"""GPU lanes for streaming pipeline with microbatching."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable, TypeVar
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch

from .microbatcher import MicroBatcher, MicroBatchItem
from .providers.base import LayoutRegion
from PIL import Image

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class GPULane:
    """Single GPU lane with dedicated executor and microbatcher."""
    
    def __init__(
        self,
        name: str,
        model: Any,
        process_batch_fn: Callable,
        input_queue: asyncio.Queue,
        output_queue: Optional[asyncio.Queue] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        microbatcher: Optional[MicroBatcher] = None,
        max_items: int = 8,
        max_bytes: int = 256 * 1024 * 1024,
        max_latency_ms: int = 50
    ):
        self.name = name
        self.model = model
        self.process_batch_fn = process_batch_fn
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"gpu-{name}")
        self.microbatcher = microbatcher or MicroBatcher(
            max_items=max_items,
            max_bytes=max_bytes,
            max_latency_ms=max_latency_ms
        )
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._stats = {
            "batches_processed": 0,
            "items_processed": 0,
            "total_time": 0.0,
            "gpu_time": 0.0
        }
        
    async def start(self):
        """Start the GPU lane."""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info(f"Started GPU lane: {self.name}")
        
    async def stop(self):
        """Stop the GPU lane."""
        if not self._running:
            return
            
        self._running = False
        
        # Send sentinel to stop processing
        await self.input_queue.put(None)
        
        if self._task:
            await self._task
            
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info(
            f"Stopped GPU lane: {self.name} - "
            f"batches={self._stats['batches_processed']}, "
            f"items={self._stats['items_processed']}, "
            f"gpu_time={self._stats['gpu_time']:.3f}s"
        )
        
    async def _run(self):
        """Main processing loop."""
        loop = asyncio.get_running_loop()
        last_flush_check = time.time()
        
        try:
            while self._running:
                # Try to get an item with timeout for periodic flush checks
                try:
                    item = await asyncio.wait_for(
                        self.input_queue.get(), 
                        timeout=0.01  # 10ms
                    )
                except asyncio.TimeoutError:
                    # Check if we need to flush due to time
                    if time.time() - last_flush_check > 0.01:
                        batch = self.microbatcher.maybe_flush_due_to_time()
                        if batch:
                            await self._process_batch(batch, loop)
                        last_flush_check = time.time()
                    continue
                    
                if item is None:  # Sentinel
                    # Flush any pending items
                    batch = self.microbatcher.force_flush()
                    if batch:
                        await self._process_batch(batch, loop)
                    
                    # Propagate sentinel
                    if self.output_queue:
                        await self.output_queue.put(None)
                    
                    self.input_queue.task_done()
                    break
                    
                # Add to microbatcher
                batch = self.microbatcher.add(item)
                if batch:
                    await self._process_batch(batch, loop)
                    
                self.input_queue.task_done()
                
        except Exception as e:
            logger.error(f"Error in GPU lane {self.name}: {e}", exc_info=True)
            raise
    
    async def _process_batch(
        self, 
        batch: List[MicroBatchItem], 
        loop: asyncio.EventLoop
    ):
        """Process a batch of items on the GPU."""
        start_time = time.time()
        
        try:
            # Run GPU processing in executor to avoid blocking event loop
            results = await loop.run_in_executor(
                self.executor,
                self._gpu_process,
                batch
            )
            
            # Update stats
            gpu_time = time.time() - start_time
            self._stats["batches_processed"] += 1
            self._stats["items_processed"] += len(batch)
            self._stats["gpu_time"] += gpu_time
            
            logger.debug(
                f"GPU lane {self.name}: processed batch of {len(batch)} items "
                f"in {gpu_time:.3f}s ({gpu_time/len(batch)*1000:.1f}ms/item)"
            )
            
            # Send results to output queue if specified
            if self.output_queue and results:
                for result in results:
                    await self.output_queue.put(result)
                    
        except Exception as e:
            logger.error(f"Error processing batch in {self.name}: {e}", exc_info=True)
            # Optionally re-queue items or handle error
            raise
    
    def _gpu_process(self, batch: List[MicroBatchItem]) -> List[Any]:
        """Run the actual GPU processing (in executor thread)."""
        with torch.inference_mode():
            # Extract data from batch items
            batch_data = [item.data for item in batch]
            batch_metadata = [item.metadata for item in batch]
            
            # Call the processing function
            results = self.process_batch_fn(self.model, batch_data, batch_metadata)
            
            return results


class LayoutGPULane(GPULane):
    """GPU lane specifically for layout detection."""
    
    def __init__(self, model, pdf_path: Path, *args, **kwargs):
        self.pdf_path = pdf_path
        
        def process_batch(model, items: List[Dict], metadata: List[Dict[str, Any]]):
            """Process a batch of images for layout detection."""
            images = [item['image'] for item in items]
            page_nos = [item['page_info'].page_no for item in items]
            
            # Use the batched detect method
            if hasattr(model, 'detect_batch'):
                results = asyncio.run(model.detect_batch(
                    images=images,
                    page_nos=page_nos
                ))
                
                # Process results and determine OCR needs
                from src.app.services.ingest.page_router import route_page, PageSignals
                from src.app.services.ingest.providers.pdf_pager import extract_text_spans
                output = []
                
                for regions, item in zip(results, items):
                    page_no = item['page_info'].page_no
                    
                    # Determine page routing
                    try:
                        spans = asyncio.run(extract_text_spans(self.pdf_path, page_no))
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
                    
                    # Determine which text regions need OCR
                    needs_ocr = {}
                    for idx, r in enumerate(regions):
                        if r.rtype == "text" and route in ("ocr", "hybrid"):
                            needs_ocr[idx] = True
                    
                    output.append({
                        'page_no': page_no,
                        'regions': regions,
                        'image': item['image'],
                        'needs_ocr': needs_ocr,
                        'route': route,
                        'image_uri': item.get('image_uri')
                    })
                return output
            else:
                raise ValueError("Model does not support batch processing")
        
        super().__init__(
            name="layout",
            model=model,
            process_batch_fn=process_batch,
            *args,
            **kwargs
        )


class OCRGPULane(GPULane):
    """GPU lane specifically for OCR processing."""
    
    def __init__(self, model, pdf_path, *args, **kwargs):
        self.pdf_path = pdf_path
        
        def process_batch(model, items: List[Dict], metadata: List[Dict[str, Any]]):
            """Process regions for OCR."""
            # Extract regions needing OCR
            ocr_tasks = []
            task_indices = []
            
            for i, item in enumerate(items):
                page_no = item['page_no']
                regions = item.get('regions', [])
                
                for j, region in enumerate(regions):
                    if region.rtype == "text" and item.get('needs_ocr', {}).get(j, False):
                        ocr_tasks.append((page_no, region))
                        task_indices.append((i, j))
            
            if not ocr_tasks:
                return items  # Pass through unchanged
            
            # Run OCR
            page_nos = [t[0] for t in ocr_tasks]
            regions = [t[1] for t in ocr_tasks]
            
            if hasattr(model, 'ocr_regions_batch'):
                texts = asyncio.run(model.ocr_regions_batch(
                    pdf_path=self.pdf_path,
                    page_nos=page_nos,
                    regions=regions
                ))
                
                # Map results back
                for (item_idx, region_idx), text in zip(task_indices, texts):
                    if 'ocr_results' not in items[item_idx]:
                        items[item_idx]['ocr_results'] = {}
                    items[item_idx]['ocr_results'][region_idx] = text
            
            return items
        
        super().__init__(
            name="ocr",
            model=model,
            process_batch_fn=process_batch,
            *args,
            **kwargs
        )


class TableGPULane(GPULane):
    """GPU lane specifically for table structure detection."""
    
    def __init__(self, model, *args, **kwargs):
        def process_batch(model, items: List[Dict], metadata: List[Dict[str, Any]]):
            """Process table regions."""
            # Extract table crops
            table_tasks = []
            task_indices = []
            
            for i, item in enumerate(items):
                tables = item.get('tables', [])
                
                for j, table in enumerate(tables):
                    if 'crop_image' in table:
                        table_tasks.append(table['crop_image'])
                        task_indices.append((i, j))
            
            if not table_tasks:
                return items  # Pass through unchanged
            
            # Run table structure detection
            if hasattr(model, 'detect_cells_batch'):
                cells_list = asyncio.run(model.detect_cells_batch(images=table_tasks))
                
                # Map results back
                for (item_idx, table_idx), cells in zip(task_indices, cells_list):
                    items[item_idx]['tables'][table_idx]['cells'] = cells
            
            return items
        
        super().__init__(
            name="table",
            model=model,
            process_batch_fn=process_batch,
            *args,
            **kwargs
        )

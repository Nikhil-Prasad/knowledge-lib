"""Batched versions of providers for GPU efficiency with caching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple
import asyncio
from functools import lru_cache

from PIL import Image
import torch

from .base import LayoutRegion
from .layout_detector import LayoutDetectorDetr
from .got_ocr import GOTOCRProvider
from .table_transformer import HfTableStructureExtractor
from src.app.domain.common import BBox

logger = logging.getLogger(__name__)


class BatchedLayoutDetector(LayoutDetectorDetr):
    """Batched version of layout detector."""
    
    async def detect_batch(self, *, images: List[Image.Image], page_nos: List[int]) -> List[List[LayoutRegion]]:
        """Detect layouts for multiple images in a batch."""
        import torch
        
        self._ensure_model()
        
        if not images:
            return []
        
        # Process all images at once using true batch processing
        target_sizes = [(img.height, img.width) for img in images]
        
        # Let the processor handle all images at once
        batch_inputs = self._proc(images=images, return_tensors="pt")
        
        # Move to device
        batch_inputs = {k: v.to(self._device) for k, v in batch_inputs.items()}
        
        # Run inference
        with torch.inference_mode():
            outputs = self._model(**batch_inputs)
        
        # Post-process each result
        all_regions = []
        
        # Create a simple object to hold the outputs
        class OutputHolder:
            def __init__(self, logits, pred_boxes):
                self.logits = logits
                self.pred_boxes = pred_boxes
        
        for i, (page_no, target_size) in enumerate(zip(page_nos, target_sizes)):
            # Extract outputs for this image
            single_output = OutputHolder(
                logits=outputs.logits[i:i+1],
                pred_boxes=outputs.pred_boxes[i:i+1]
            )
            
            det = self._proc.post_process_object_detection(
                single_output, threshold=0.4, target_sizes=[target_size]
            )[0]
            
            regions = self._process_detections(det, page_no, images[i].size)
            all_regions.append(regions)
        
        return all_regions
    
    def _process_detections(self, det, page_no: int, img_size: Tuple[int, int]) -> List[LayoutRegion]:
        """Process detection results into LayoutRegion objects."""
        width, height = img_size
        id2label = getattr(self._model.config, "id2label", {})
        
        def _rtype(lbl: str) -> str:
            l = lbl.lower()
            if "table" in l:
                return "table"
            if "figure" in l or "picture" in l or "image" in l:
                return "figure"
            if "caption" in l:
                return "caption"
            if any(k in l for k in ["text", "title", "paragraph", "list", "header", "footer"]):
                return "text"
            return "other"
        
        regions: List[LayoutRegion] = []
        for box, score, label_id in zip(det["boxes"], det["scores"], det["labels"]):
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            try:
                lbl = id2label[int(label_id)]
            except Exception:
                lbl = str(int(label_id))
            bx = BBox(
                x0=max(0.0, x0 / width),
                y0=max(0.0, y0 / height),
                x1=min(1.0, x1 / width),
                y1=min(1.0, y1 / height),
            )
            regions.append(LayoutRegion(page_no=page_no, rtype=_rtype(lbl), bbox=bx, score=float(score)))
        
        return regions


class BatchedOCRProvider(GOTOCRProvider):
    """Batched version of GOT-OCR provider."""
    
    async def ocr_regions_batch(
        self, 
        *, 
        pdf_path: Path,
        page_nos: List[int],
        regions: List[LayoutRegion]
    ) -> List[str]:
        """OCR multiple regions in a batch."""
        from .utils import render_page_image
        from src.app.settings import get_settings
        import torch
        
        self._ensure_model()
        
        if not regions:
            return []
        
        settings = get_settings()
        dpi = settings.pdf_render_dpi
        
        # Prepare crops
        crops = []
        for page_no, region in zip(page_nos, regions):
            img = render_page_image(pdf_path, page_no, dpi)
            W, H = img.size
            x0 = int(max(0, min(W, region.bbox.x0 * W)))
            y0 = int(max(0, min(H, region.bbox.y0 * H)))
            x1 = int(max(0, min(W, region.bbox.x1 * W)))
            y1 = int(max(0, min(H, region.bbox.y1 * H)))
            
            if x1 > x0 and y1 > y0:
                crop = img.crop((x0, y0, x1, y1))
                crops.append(crop)
            else:
                crops.append(None)
        
        # Process in sub-batches if needed (GOT-OCR can be memory intensive)
        max_batch_size = 4  # Adjust based on GPU memory
        results = []
        
        for i in range(0, len(crops), max_batch_size):
            batch_crops = crops[i:i + max_batch_size]
            batch_results = await self._process_ocr_batch(batch_crops)
            results.extend(batch_results)
        
        return results
    
    async def _process_ocr_batch(self, crops: List[Image.Image | None]) -> List[str]:
        """Process a batch of crops through OCR."""
        import torch
        
        texts = []
        valid_indices = []
        valid_crops = []
        
        # Filter out None crops
        for i, crop in enumerate(crops):
            if crop is not None:
                valid_indices.append(i)
                valid_crops.append(crop)
        
        if not valid_crops:
            return [""] * len(crops)
        
        # Process valid crops
        with torch.inference_mode():
            # Process multiple images at once
            batch_texts = []
            for crop in valid_crops:
                inputs = self._tokenizer(crop, return_tensors="pt")
                inputs = {k: v.to(self._device, dtype=self._dtype if v.dtype.is_floating_point else v.dtype) 
                         for k, v in inputs.items()}
                
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    temperature=0,
                )
                
                text = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                text = self._clean_text(text)
                batch_texts.append(text)
        
        # Map back to original indices
        result_texts = [""] * len(crops)
        for idx, text in zip(valid_indices, batch_texts):
            result_texts[idx] = text
        
        return result_texts


class BatchedTableExtractor(HfTableStructureExtractor):
    """Batched version of table structure extractor."""
    
    async def detect_cells_batch(self, *, images: List[Image.Image]) -> List[List[dict]]:
        """Detect table cells for multiple images in a batch."""
        import torch
        
        self._ensure_model()
        
        if not images:
            return []
        
        # Process all images at once using true batch processing
        target_sizes = [(img.height, img.width) for img in images]
        
        # Let the processor handle all images at once
        batch_inputs = self._proc(images=images, return_tensors="pt")
        
        # Move to device
        batch_inputs = {k: v.to(self._device) for k, v in batch_inputs.items()}
        
        # Run inference
        with torch.inference_mode():
            outputs = self._model(**batch_inputs)
        
        # Post-process each result
        all_cells = []
        
        # Create a simple object to hold the outputs
        class OutputHolder:
            def __init__(self, logits, pred_boxes):
                self.logits = logits
                self.pred_boxes = pred_boxes
        
        for i, target_size in enumerate(target_sizes):
            # Extract outputs for this image
            single_output = OutputHolder(
                logits=outputs.logits[i:i+1],
                pred_boxes=outputs.pred_boxes[i:i+1]
            )
            
            det = self._proc.post_process_object_detection(
                single_output, threshold=0.8, target_sizes=[target_size]
            )[0]
            
            cells = self._process_table_detections(det, images[i].size)
            all_cells.append(cells)
        
        return all_cells
    
    def _process_table_detections(self, det, img_size: Tuple[int, int]) -> List[dict]:
        """Process detection results into cell information."""
        width, height = img_size
        id2label = getattr(self._model.config, "id2label", {})
        
        cells = []
        for box, score, label_id in zip(det["boxes"], det["scores"], det["labels"]):
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            try:
                label = id2label[int(label_id)]
            except Exception:
                label = f"cell_{int(label_id)}"
            
            cell = {
                "bbox": {
                    "x0": max(0.0, x0 / width),
                    "y0": max(0.0, y0 / height),
                    "x1": min(1.0, x1 / width),
                    "y1": min(1.0, y1 / height),
                },
                "label": label,
                "score": float(score),
            }
            cells.append(cell)
        
        return cells


# Use cached instances to avoid reloading models
@lru_cache(maxsize=1)
def get_batched_layout_detector():
    """Get a cached instance of BatchedLayoutDetector."""
    return BatchedLayoutDetector()


@lru_cache(maxsize=1)
def get_batched_ocr_provider():
    """Get a cached instance of BatchedOCRProvider."""
    return BatchedOCRProvider()


@lru_cache(maxsize=1) 
def get_batched_table_extractor():
    """Get a cached instance of BatchedTableExtractor."""
    return BatchedTableExtractor()


def get_batched_providers():
    """Get cached instances of batched providers."""
    return (
        get_batched_layout_detector(),
        get_batched_ocr_provider(),
        get_batched_table_extractor()
    )

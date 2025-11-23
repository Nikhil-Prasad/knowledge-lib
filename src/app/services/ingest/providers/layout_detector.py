from __future__ import annotations

from pathlib import Path
from typing import List

from .base import LayoutDetector, LayoutRegion
from src.app.domain.common import BBox
from src.app.settings import get_settings
from .utils import render_page_image, get_torch_device


class LayoutDetectorDetr(LayoutDetector):
    """Layout detector using DETR (DEtection TRansformer) model for document layout analysis."""
    
    _proc = None
    _model = None
    _device = None

    def _ensure_model(self) -> None:
        if self._proc is not None and self._model is not None:
            return
        from transformers import AutoImageProcessor
        from transformers.models.detr import DetrForSegmentation

        settings = get_settings()
        model_name = getattr(settings, "pdf_layout_model", None) or "cmarkea/detr-layout-detection"
        
        self._proc = AutoImageProcessor.from_pretrained(model_name)
        self._model = DetrForSegmentation.from_pretrained(model_name)
        self._model.eval()
        
        self._device = get_torch_device()
        self._model.to(self._device)

    async def detect(self, *, pdf_path: Path, page_no: int) -> List[LayoutRegion]:
        import torch

        self._ensure_model()
        settings = get_settings()
        dpi = settings.pdf_render_dpi
        img = render_page_image(pdf_path, page_no, dpi)
        width, height = img.size

        with torch.inference_mode():
            inputs = self._proc(images=img, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            det = self._proc.post_process_object_detection(
                outputs, threshold=0.4, target_sizes=[(height, width)]
            )[0]

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

from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image

from .base import LayoutDetector, LayoutRegion
from src.app.domain.common import BBox
from src.app.settings import get_settings


def _render_page_image(pdf_path: Path, page_no: int, dpi: int) -> Image.Image:
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    page = doc[page_no - 1]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


class HfLayoutDetector(LayoutDetector):
    _proc = None
    _model = None
    _device = None

    def _ensure_model(self) -> None:
        if self._proc is not None and self._model is not None:
            return
        from transformers import AutoImageProcessor
        from transformers.models.detr import DetrForSegmentation
        import torch

        settings = get_settings()
        model_name = getattr(settings, "pdf_layout_model", None) or "cmarkea/detr-layout-detection"
        self._proc = AutoImageProcessor.from_pretrained(model_name)
        self._model = DetrForSegmentation.from_pretrained(model_name)
        self._model.eval()
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device)

    async def detect(self, *, pdf_path: Path, page_no: int) -> List[LayoutRegion]:
        import torch

        self._ensure_model()
        settings = get_settings()
        dpi = settings.pdf_render_dpi
        img = _render_page_image(pdf_path, page_no, dpi)
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


class HfOCRProvider:
    async def ocr_region(self, *, pdf_path: Path, page_no: int, region: LayoutRegion) -> str:
        from transformers import pipeline
        import torch

        settings = get_settings()
        model_name = getattr(settings, "pdf_ocr_model", None) or "Salesforce/blip-image-captioning-base"  # placeholder
        dpi = settings.pdf_render_dpi
        img = _render_page_image(pdf_path, page_no, dpi)
        W, H = img.size
        # crop region (normalized bbox)
        x0 = int(region.bbox.x0 * W)
        y0 = int(region.bbox.y0 * H)
        x1 = int(region.bbox.x1 * W)
        y1 = int(region.bbox.y1 * H)
        crop = img.crop((x0, y0, x1, y1))

        # Use generic image-to-text pipeline as a placeholder; real DeepSeek OCR overrides this provider
        pipe = pipeline("image-to-text", model=model_name, device_map="auto")
        out = pipe(crop)
        if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
            return out[0]["generated_text"].strip()
        return ""

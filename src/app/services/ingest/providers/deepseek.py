from __future__ import annotations

from pathlib import Path
import logging

from PIL import Image

from .base import LayoutRegion
from src.app.settings import get_settings

logger = logging.getLogger(__name__)


def _render_page_image(pdf_path: Path, page_no: int, dpi: int) -> Image.Image:
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    page = doc[page_no - 1]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


class DeepseekOCRProvider:
    _pipe = None

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        from transformers import pipeline

        settings = get_settings()
        model_name = getattr(settings, "pdf_ocr_model", None) or "deepseek-ai/DeepSeek-OCR"
        # Use accelerate to select MPS/GPU/CPU automatically
        # DeepSeek-OCR repo uses custom code; allow it explicitly. Fallback to GOT-OCR2_0 if it fails.
        try:
            self._pipe = pipeline(
                "image-to-text",
                model=model_name,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("Loaded DeepSeek-OCR pipeline: %s", model_name)
        except Exception as e:
            logger.warning(
                "DeepSeek-OCR failed to load (%s). Falling back to stepfun-ai/GOT-OCR2_0.", e,
                exc_info=True,
            )
            self._pipe = pipeline(
                "image-to-text",
                model="stepfun-ai/GOT-OCR2_0",
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("Loaded fallback OCR pipeline: stepfun-ai/GOT-OCR2_0")
        return self._pipe

    async def ocr_region(self, *, pdf_path: Path, page_no: int, region: LayoutRegion) -> str:
        settings = get_settings()
        dpi = settings.pdf_render_dpi
        img = _render_page_image(pdf_path, page_no, dpi)
        W, H = img.size
        x0 = int(region.bbox.x0 * W)
        y0 = int(region.bbox.y0 * H)
        x1 = int(region.bbox.x1 * W)
        y1 = int(region.bbox.y1 * H)
        crop = img.crop((x0, y0, x1, y1))

        pipe = self._get_pipeline()
        out = pipe(crop)
        if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
            return out[0]["generated_text"].strip()
        return ""

from __future__ import annotations

from pathlib import Path
import logging
from typing import List

from PIL import Image
import torch

from src.app.settings import get_settings
from .base import LayoutRegion

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


class GotOcrProvider:
    """OCR provider using stepfun-ai/GOT-OCR-2.0-hf via AutoProcessor + AutoModelForImageTextToText."""

    _model = None
    _proc = None
    _device: torch.device | None = None

    def _ensure_model(self) -> None:
        if self._model is not None and self._proc is not None:
            return
        from transformers import AutoProcessor, AutoModelForImageTextToText

        settings = get_settings()
        model_name = getattr(settings, "pdf_ocr_model", None) or "stepfun-ai/GOT-OCR-2.0-hf"

        # Select device
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._proc = AutoProcessor.from_pretrained(model_name, use_fast=True)

        # Choose dtype conservatively per device
        dtype = torch.float32
        if self._device.type == "cuda":
            dtype = torch.bfloat16
        elif self._device.type == "mps":
            dtype = torch.float16

        self._model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map={"": self._device.type},
            torch_dtype=dtype,
        )
        self._model.eval()
        logger.info("Loaded GOT-OCR model: %s on %s", model_name, self._device)

    async def ocr_region(self, *, pdf_path: Path, page_no: int, region: LayoutRegion) -> str:
        settings = get_settings()
        dpi = settings.pdf_render_dpi
        img = _render_page_image(pdf_path, page_no, dpi)
        W, H = img.size
        x0 = int(max(0, min(W, region.bbox.x0 * W)))
        y0 = int(max(0, min(H, region.bbox.y0 * H)))
        x1 = int(max(0, min(W, region.bbox.x1 * W)))
        y1 = int(max(0, min(H, region.bbox.y1 * H)))
        crop = img.crop((x0, y0, x1, y1))

        # Use processor+model per HF docs to ensure image tokens align with features
        self._ensure_model()
        inputs = self._proc(images=crop, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            generate_ids = self._model.generate(
                **inputs,
                do_sample=False,
                tokenizer=getattr(self._proc, "tokenizer", None),
                stop_strings="<|im_end|>",
                max_new_tokens=1024,
            )
        try:
            gen_only = generate_ids[:, inputs["input_ids"].shape[1]:]
        except Exception:
            gen_only = generate_ids
        # Decode
        if hasattr(self._proc, "batch_decode"):
            text = self._proc.batch_decode(gen_only, skip_special_tokens=True)[0]
        else:
            tokenizer = getattr(self._proc, "tokenizer", None)
            text = tokenizer.decode(gen_only[0], skip_special_tokens=True) if tokenizer is not None else ""
        return (text or "").strip()

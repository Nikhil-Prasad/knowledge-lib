from __future__ import annotations

from typing import Tuple

from src.app.settings import get_settings
from .base import PDFPager, LayoutDetector, OCRProvider
from .noop import NoopPager, NoopLayout, NoopOCR


def resolve_providers() -> Tuple[PDFPager, LayoutDetector, OCRProvider]:
    settings = get_settings()

    # Pager: prefer PyMuPDF if available, fallback to Noop
    try:
        from .pymupdf_pager import PymupdfPager  # type: ignore
        pager: PDFPager = PymupdfPager()
    except Exception:
        pager = NoopPager()

    # Layout provider
    if settings.pdf_layout_provider == "noop":
        layout: LayoutDetector = NoopLayout()
    elif settings.pdf_layout_provider == "hf":
        from .hf import HfLayoutDetector  # type: ignore
        layout = HfLayoutDetector()
    else:
        layout = NoopLayout()

    # OCR provider
    if settings.pdf_ocr_provider == "noop":
        ocr: OCRProvider = NoopOCR()
    elif settings.pdf_ocr_provider == "got" or (
        getattr(settings, "pdf_ocr_model", "") or ""
    ).lower().endswith("got-ocr-2.0-hf"):
        from .got_ocr import GotOcrProvider  # type: ignore
        ocr = GotOcrProvider()
    elif settings.pdf_ocr_provider == "deepseek":
        from .deepseek import DeepseekOCRProvider  # type: ignore
        ocr = DeepseekOCRProvider()
    elif settings.pdf_ocr_provider == "hf":
        from .hf import HfOCRProvider  # type: ignore
        ocr = HfOCRProvider()
    else:
        ocr = NoopOCR()

    return pager, layout, ocr

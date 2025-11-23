from __future__ import annotations

from typing import Tuple

from src.app.settings import get_settings
from .base import PDFPager, LayoutDetector, OCRProvider
"""Provider resolver for pager/layout/ocr.

In production, we require explicit, functional providers. No-op providers are not
kept in src; tests should monkeypatch this resolver or pass stubs explicitly.
"""


def resolve_providers() -> Tuple[PDFPager, LayoutDetector, OCRProvider]:
    settings = get_settings()

    # Pager: require PyMuPDF unless explicitly using noop in tests
    try:
        from .cached_providers import get_pdf_pager
        pager: PDFPager = get_pdf_pager()
    except Exception as e:
        raise RuntimeError(
            "PyMuPDF is required for PDF paging. Install 'pymupdf' or set providers to 'noop' for tests."
        ) from e

    # Layout provider
    if settings.pdf_layout_provider == "detr":
        from .cached_providers import get_layout_detector
        layout = get_layout_detector()
    elif settings.pdf_layout_provider == "hf":  # backward compatibility
        from .cached_providers import get_layout_detector
        layout = get_layout_detector()
    else:
        raise ValueError(f"Unknown PDF_LAYOUT_PROVIDER: {settings.pdf_layout_provider}")

    # OCR provider
    if settings.pdf_ocr_provider == "got":
        from .cached_providers import get_got_ocr_provider
        ocr = get_got_ocr_provider()
    else:
        raise ValueError(f"Unknown PDF_OCR_PROVIDER: {settings.pdf_ocr_provider}")

    return pager, layout, ocr

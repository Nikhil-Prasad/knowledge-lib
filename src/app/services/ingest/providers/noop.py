from __future__ import annotations

from pathlib import Path
from typing import List

from .base import PDFPager, PDFPageInfo, LayoutDetector, LayoutRegion, OCRProvider
from src.app.domain.common import BBox


class NoopPager(PDFPager):
    async def pages(self, *, pdf_path: Path, max_pages: int) -> List[PDFPageInfo]:
        # Unknown page count without PDF libs; return a single placeholder page
        return [PDFPageInfo(page_no=1, width_px=None, height_px=None)]


class NoopLayout(LayoutDetector):
    async def detect(self, *, pdf_path: Path, page_no: int) -> List[LayoutRegion]:
        # No layout; return empty regions list
        return []


class NoopOCR(OCRProvider):
    async def ocr_region(self, *, pdf_path: Path, page_no: int, region: LayoutRegion) -> str:
        # No OCR performed
        return ""


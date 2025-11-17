from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol

from src.app.domain.common import BBox


@dataclass
class PDFPageInfo:
    page_no: int
    width_px: int | None
    height_px: int | None


@dataclass
class LayoutRegion:
    page_no: int
    rtype: str  # 'text' | 'table' | 'figure'
    bbox: BBox
    score: float = 1.0


class PDFPager(Protocol):
    async def pages(self, *, pdf_path: Path, max_pages: int) -> List[PDFPageInfo]:
        ...


@dataclass
class TextSpan:
    text: str
    bbox: BBox

class PDFVectorText(Protocol):
    async def text_spans(self, *, pdf_path: Path, page_no: int) -> List[TextSpan]:
        ...


class LayoutDetector(Protocol):
    async def detect(self, *, pdf_path: Path, page_no: int) -> List[LayoutRegion]:
        ...


class OCRProvider(Protocol):
    async def ocr_region(self, *, pdf_path: Path, page_no: int, region: LayoutRegion) -> str:
        ...

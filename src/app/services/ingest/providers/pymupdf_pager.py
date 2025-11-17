from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .base import PDFPager, PDFPageInfo, TextSpan
from src.app.domain.common import BBox
from src.app.settings import get_settings


class PymupdfPager(PDFPager):
    async def pages(self, *, pdf_path: Path, max_pages: int) -> List[PDFPageInfo]:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        infos: List[PDFPageInfo] = []
        dpi = get_settings().pdf_render_dpi
        limit = min(len(doc), max_pages)
        for i in range(limit):
            page = doc[i]
            rect = page.rect  # points (1/72 inch)
            width_px = int(round(rect.width / 72.0 * dpi))
            height_px = int(round(rect.height / 72.0 * dpi))
            infos.append(PDFPageInfo(page_no=i + 1, width_px=width_px, height_px=height_px))
        return infos


async def extract_text_spans(pdf_path: Path, page_no: int) -> List[TextSpan]:
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    page = doc[page_no - 1]
    rect = page.rect
    W = max(1.0, rect.width)
    H = max(1.0, rect.height)
    raw = page.get_text("rawdict")
    spans: List[TextSpan] = []
    # Primary path: spans from rawdict
    try:
        for block in raw.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = (span.get("text") or "").strip()
                    if not t:
                        continue
                    x0, y0, x1, y1 = span.get("bbox", [0, 0, 0, 0])
                    # PyMuPDF span bbox uses top-left origin coordinates; normalize directly to [0,1]
                    # Clip to bounds to avoid negatives from rounding
                    nx0 = max(0.0, min(1.0, x0 / W))
                    ny0 = max(0.0, min(1.0, y0 / H))
                    nx1 = max(0.0, min(1.0, x1 / W))
                    ny1 = max(0.0, min(1.0, y1 / H))
                    # Ensure proper ordering
                    if nx1 < nx0:
                        nx0, nx1 = nx1, nx0
                    if ny1 < ny0:
                        ny0, ny1 = ny1, ny0
                    bx = BBox(x0=nx0, y0=ny0, x1=nx1, y1=ny1)
                    spans.append(TextSpan(text=t, bbox=bx))
    except Exception:
        # fall back below
        spans = []

    # Fallback path: words API if raw spans empty
    if not spans:
        try:
            # words: list of (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            for w in page.get_text("words") or []:
                if not w or len(w) < 5:
                    continue
                x0, y0, x1, y1, t = w[:5]
                t = (t or "").strip()
                if not t:
                    continue
                nx0 = max(0.0, min(1.0, x0 / W))
                ny0 = max(0.0, min(1.0, y0 / H))
                nx1 = max(0.0, min(1.0, x1 / W))
                ny1 = max(0.0, min(1.0, y1 / H))
                if nx1 < nx0:
                    nx0, nx1 = nx1, nx0
                if ny1 < ny0:
                    ny0, ny1 = ny1, ny0
                bx = BBox(x0=nx0, y0=ny0, x1=nx1, y1=ny1)
                spans.append(TextSpan(text=t, bbox=bx))
        except Exception:
            # no spans
            spans = []
    return spans

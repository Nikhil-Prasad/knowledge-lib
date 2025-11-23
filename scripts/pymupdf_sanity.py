#!/usr/bin/env python3
"""Quick PyMuPDF sanity check for PDF text spans.

Usage:
  python scripts/pymupdf_sanity.py pdfs/1810.03163.pdf [page_no]

Prints page size, a text sample, raw span count, and a few span samples.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        print("ERROR: PyMuPDF (fitz) not importable:", e)
        return 2

    if len(sys.argv) < 2:
        print("Usage: python scripts/pymupdf_sanity.py <pdf_path> [page_no]")
        return 1

    pdf_path = Path(sys.argv[1])
    page_no = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if not pdf_path.exists():
        print(f"ERROR: file not found: {pdf_path}")
        return 1

    doc = fitz.open(str(pdf_path))
    if page_no < 1 or page_no > len(doc):
        print(f"ERROR: invalid page {page_no}; document has {len(doc)} pages")
        return 1

    page = doc[page_no - 1]
    rect = page.rect
    print(f"Page {page_no} size: {rect}")

    # Simple text sample
    text_simple = page.get_text("text") or ""
    text_display = text_simple[:500].replace("\n", " ")
    print("\nSIMPLE TEXT SAMPLE:\n", text_display)

    # Raw spans
    raw = page.get_text("rawdict") or {}
    blocks = raw.get("blocks", [])
    spans = []
    for block in blocks:
        # Only text blocks
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                spans.append(span)

    print("\nRAW SPAN COUNT:", len(spans))
    print("FIRST 5 SPANS:")
    for s in spans[:5]:
        bbox = s.get("bbox")
        txt = s.get("text", "")
        print("  bbox=", bbox, " text=", repr(txt)[:80])

    # Normalized bbox sanity (0..1)
    W, H = max(1.0, rect.width), max(1.0, rect.height)
    norm_samples = []
    for s in spans[:5]:
        x0, y0, x1, y1 = s.get("bbox", [0, 0, 0, 0])
        nx0, ny0, nx1, ny1 = x0 / W, y0 / H, x1 / W, y1 / H
        norm_samples.append((round(nx0, 3), round(ny0, 3), round(nx1, 3), round(ny1, 3)))
    print("NORMALIZED BBOX SAMPLES:", norm_samples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


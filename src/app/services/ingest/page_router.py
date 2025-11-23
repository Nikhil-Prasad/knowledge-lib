from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.app.settings import get_settings


@dataclass
class PageSignals:
    text_coverage: float  # 0..1
    image_coverage: float  # 0..1
    sandwich_score: float  # 0..1


Route = Literal["digital", "ocr", "hybrid"]


def route_page(sig: PageSignals) -> Route:
    s = get_settings()
    # Digital if text coverage is high and not a sandwich OCR artifact
    if sig.text_coverage >= s.route_text_coverage_high and sig.sandwich_score < s.route_sandwich_threshold:
        return "digital"
    # OCR if low text and high image coverage
    if sig.text_coverage <= s.route_text_coverage_low and sig.image_coverage >= s.route_image_coverage_high:
        return "ocr"
    return "hybrid"


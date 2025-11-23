"""Cached provider instances using functools.lru_cache to avoid reloading models."""

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .table_transformer import HfTableStructureExtractor
    from .deplot import DePlotProvider
    from .layout_detector import LayoutDetectorDetr
    from .got_ocr import GOTOCRProvider
    from .pdf_pager import PDFPagerImpl


@lru_cache(maxsize=1)
def get_table_structure_extractor() -> "HfTableStructureExtractor":
    """Get a cached instance of HfTableStructureExtractor."""
    from .table_transformer import HfTableStructureExtractor
    return HfTableStructureExtractor()


@lru_cache(maxsize=1)
def get_deplot_provider() -> "DePlotProvider":
    """Get a cached instance of DePlotProvider."""
    from .deplot import DePlotProvider
    return DePlotProvider()


@lru_cache(maxsize=1)
def get_layout_detector() -> "LayoutDetectorDetr":
    """Get a cached instance of LayoutDetectorDetr."""
    from .layout_detector import LayoutDetectorDetr
    return LayoutDetectorDetr()


@lru_cache(maxsize=1)
def get_got_ocr_provider() -> "GOTOCRProvider":
    """Get a cached instance of GOTOCRProvider."""
    from .got_ocr import GOTOCRProvider
    return GOTOCRProvider()


@lru_cache(maxsize=1)
def get_pdf_pager() -> "PDFPagerImpl":
    """Get a cached instance of PDFPagerImpl."""
    from .pdf_pager import PDFPagerImpl
    return PDFPagerImpl()

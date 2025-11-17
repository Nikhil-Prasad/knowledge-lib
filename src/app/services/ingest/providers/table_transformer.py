from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from src.app.domain.common import BBox


class TableDetector:
    async def detect_tables(self, *, image_path: Path) -> List[BBox]:
        """Detect table regions using Microsoft Table Transformer (detection head)."""
        raise NotImplementedError


class TableStructureExtractor:
    async def extract_structure(self, *, image_path: Path, table_bbox: BBox) -> Tuple[int, int, List[List[str]]]:
        """Return (n_rows, n_cols, cell_texts) for a detected table region.

        Cells are returned row-major with already-OCRed text per cell.
        """
        raise NotImplementedError


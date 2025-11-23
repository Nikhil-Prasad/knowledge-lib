from __future__ import annotations

from typing import List, Tuple
from dataclasses import dataclass

from PIL import Image

from src.app.domain.common import BBox
from src.app.settings import get_settings


@dataclass
class TableCell:
    bbox: BBox  # normalized to the TABLE REGION (0..1 in region coords)


class HfTableStructureExtractor:
    """Lightweight wrapper around Microsoft Table-Transformer (structure recognition).

    Uses `microsoft/table-transformer-structure-recognition` to detect table cells
    inside a provided region crop. Returns a list of cell boxes normalized to the
    table region (0..1), leaving grouping/rows/cols to the caller.
    """

    _proc = None
    _model = None
    _device = None

    def _ensure_model(self) -> None:
        if self._proc is not None and self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        from .utils import get_torch_device

        settings = get_settings()
        model_name = (
            getattr(settings, "pdf_table_struct_model", None)
            or "microsoft/table-transformer-structure-recognition"
        )

        self._proc = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModelForObjectDetection.from_pretrained(model_name)
        self._model.eval()
        
        self._device = get_torch_device()
        self._model.to(self._device)

    async def detect_cells(self, *, image: Image.Image) -> List[TableCell]:
        """Detect table cells in a cropped table image; returns cells as BBoxes in crop coords.

        Note: BBoxes are normalized to the crop (0..1).
        """
        import torch

        self._ensure_model()
        W, H = image.size
        with torch.inference_mode():
            inputs = self._proc(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            det = self._proc.post_process_object_detection(
                outputs, threshold=0.4, target_sizes=[(H, W)]
            )[0]

        id2label = getattr(self._model.config, "id2label", {})
        cells: List[TableCell] = []
        for box, score, label_id in zip(det["boxes"], det["scores"], det["labels"]):
            try:
                label = str(id2label[int(label_id)]).lower()
            except Exception:
                label = str(int(label_id))
            if "cell" not in label:
                continue
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            bx = BBox(
                x0=max(0.0, x0 / W),
                y0=max(0.0, y0 / H),
                x1=min(1.0, x1 / W),
                y1=min(1.0, y1 / H),
            )
            cells.append(TableCell(bbox=bx))
        # Sort roughly by reading order
        cells.sort(key=lambda c: (c.bbox.y0, c.bbox.x0))
        return cells

from __future__ import annotations

from typing import List

from PIL import Image

from src.app.settings import get_settings


class DePlotProvider:
    """Wrapper for google/deplot using Pix2Struct processor/model.

    DePlot is a Pix2Struct VQA-style model that requires a header (question/prompt)
    along with the image. We generate a plain-text table and then parse into rows.
    """

    _processor = None
    _model = None
    _device = None
    _dtype = None

    def _ensure_model(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
        from .utils import get_torch_device_and_dtype

        settings = get_settings()
        model_name = getattr(settings, "pdf_deplot_model", None) or "google/deplot"

        self._device, self._dtype = get_torch_device_and_dtype()
        self._processor = Pix2StructProcessor.from_pretrained(model_name)

        self._model = Pix2StructForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._model.eval()

    @staticmethod
    def _parse_to_rows(text: str) -> List[List[str]]:
        # Try markdown table first
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        rows: List[List[str]] = []
        # Markdown style: | a | b |
        if any(ln.startswith("|") and ln.endswith("|") for ln in lines):
            for ln in lines:
                if not (ln.startswith("|") and ln.endswith("|")):
                    continue
                parts = [c.strip() for c in ln.strip("|").split("|")]
                if parts and not set(parts) <= {":---", "---", ":---:", "---:"}:
                    rows.append(parts)
            return rows
        # CSV/TSV heuristic
        sep = ","
        if any("\t" in ln for ln in lines):
            sep = "\t"
        for ln in lines:
            rows.append([c.strip() for c in ln.split(sep)])
        return rows

    async def predict_table(self, *, image: Image.Image) -> List[List[str]]:
        import torch

        self._ensure_model()

        header = "Generate the underlying data table of this chart."
        inputs = self._processor(images=image, text=header, return_tensors="pt")
        # Move tensors to device and enforce dtype consistency for floats
        proc_inputs = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                if getattr(v, "is_floating_point", lambda: False)():
                    proc_inputs[k] = v.to(self._device, dtype=self._dtype)
                else:
                    proc_inputs[k] = v.to(self._device)
            else:
                proc_inputs[k] = v
        with torch.inference_mode():
            output_ids = self._model.generate(
                **proc_inputs,
                max_new_tokens=512,
            )
        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        rows = self._parse_to_rows(text)
        return rows

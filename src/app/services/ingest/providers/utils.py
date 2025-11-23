"""Shared utilities for PDF processing providers."""

from pathlib import Path
from typing import Tuple
import torch
from PIL import Image


def render_page_image(pdf_path: Path, page_no: int, dpi: int) -> Image.Image:
    """Render a PDF page to PIL Image at specified DPI."""
    import fitz  # PyMuPDF
    
    doc = fitz.open(str(pdf_path))
    page = doc[page_no - 1]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def get_torch_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    """Get optimal torch device and dtype for the current system.
    
    Returns:
        (device, dtype) tuple where:
        - device: mps/cuda/cpu based on availability
        - dtype: float32 for MPS/CPU, bfloat16 for CUDA
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32  # MPS requires float32 to avoid mixed-dtype issues
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16  # CUDA can use bfloat16 for efficiency
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    
    return device, dtype


def get_torch_device() -> torch.device:
    """Get optimal torch device (mps/cuda/cpu)."""
    device, _ = get_torch_device_and_dtype()
    return device

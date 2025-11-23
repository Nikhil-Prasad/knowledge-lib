#!/usr/bin/env python3
"""Profile MPS (Metal Performance Shaders) GPU memory usage during PDF processing."""

import asyncio
import time
import torch
import threading
from pathlib import Path
import sys
import os
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app.services.ingest.providers.batched_providers import BatchedLayoutDetector
from src.app.services.ingest.providers.layout_detector import LayoutDetectorDetr
from src.app.services.ingest.providers.pdf_pager import PDFPagerImpl
from src.app.services.ingest.providers.utils import render_page_image
from src.app.settings import get_settings


class MPSMemoryMonitor:
    """Monitor MPS GPU memory usage."""
    
    def __init__(self):
        self.monitoring = False
        self.memory_samples = []
        self.peak_memory = 0
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring GPU memory in background thread."""
        if not torch.backends.mps.is_available():
            print("MPS not available on this system")
            return
            
        self.monitoring = True
        self.memory_samples = []
        self.peak_memory = 0
        
        def monitor_loop():
            while self.monitoring:
                try:
                    current = torch.mps.current_allocated_memory()
                    self.memory_samples.append(current)
                    self.peak_memory = max(self.peak_memory, current)
                    time.sleep(0.1)  # Sample every 100ms
                except Exception:
                    pass
                    
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring and return stats."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        if not self.memory_samples:
            return None
            
        return {
            'peak_gb': self.peak_memory / 1024**3,
            'avg_gb': sum(self.memory_samples) / len(self.memory_samples) / 1024**3,
            'min_gb': min(self.memory_samples) / 1024**3,
            'max_gb': max(self.memory_samples) / 1024**3,
            'samples': len(self.memory_samples)
        }


@contextmanager
def mps_memory_context(name: str):
    """Context manager to track MPS memory for a code block."""
    if torch.backends.mps.is_available():
        start_mem = torch.mps.current_allocated_memory()
        print(f"\n[MPS {name}] Start: {start_mem / 1024**3:.3f} GB")
        
        yield
        
        end_mem = torch.mps.current_allocated_memory()
        print(f"[MPS {name}] End: {end_mem / 1024**3:.3f} GB")
        print(f"[MPS {name}] Delta: {(end_mem - start_mem) / 1024**3:.3f} GB")
    else:
        yield


async def profile_model_sizes():
    """Profile the actual memory usage of each model."""
    print("\n=== MPS GPU Memory Usage Analysis ===")
    
    if not torch.backends.mps.is_available():
        print("MPS not available - using CPU")
        device = "cpu"
    else:
        device = "mps"
        print("Using MPS (Metal Performance Shaders)")
        print(f"Initial MPS memory: {torch.mps.current_allocated_memory() / 1024**3:.3f} GB")
    
    # Test 1: Layout Detection Model
    print("\n1. Layout Detection Model (DETR)")
    with mps_memory_context("DETR Model"):
        layout = LayoutDetectorDetr()
        layout._ensure_model()
        
    # Get model size
    if hasattr(layout._model, 'parameters'):
        param_count = sum(p.numel() for p in layout._model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in layout._model.parameters())
        print(f"   Parameters: {param_count:,}")
        print(f"   Model size: {param_size / 1024**2:.1f} MB")
    
    # Test batch inference
    print("\n   Testing batch sizes:")
    settings = get_settings()
    pdf_path = Path("/Users/nikhilprasad/crown/knowledge-lib/data/pdfs/1810.03163.pdf")
    
    # Render some test images
    test_images = []
    for i in range(1, 9):  # 8 pages
        img = render_page_image(pdf_path, i, settings.pdf_render_dpi)
        test_images.append(img)
    
    for batch_size in [1, 4, 8]:
        batch_images = test_images[:batch_size]
        
        with mps_memory_context(f"Batch size {batch_size}"):
            inputs = layout._proc(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = layout._model(**inputs)
    
    # Test 2: Table Detection Model
    print("\n2. Table Structure Model")
    from src.app.services.ingest.providers.table_transformer import HfTableStructureExtractor
    
    with mps_memory_context("Table Model"):
        table_model = HfTableStructureExtractor()
        table_model._ensure_model()
    
    if hasattr(table_model._model, 'parameters'):
        param_count = sum(p.numel() for p in table_model._model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in table_model._model.parameters())
        print(f"   Parameters: {param_count:,}")
        print(f"   Model size: {param_size / 1024**2:.1f} MB")
    
    # Test 3: GOT-OCR Model
    print("\n3. GOT-OCR Model")
    try:
        from src.app.services.ingest.providers.got_ocr import GOTOCRProvider
        
        with mps_memory_context("GOT-OCR Model"):
            ocr = GOTOCRProvider()
            ocr._ensure_model()
        
        if hasattr(ocr._model, 'parameters'):
            param_count = sum(p.numel() for p in ocr._model.parameters())
            param_size = sum(p.numel() * p.element_size() for p in ocr._model.parameters())
            print(f"   Parameters: {param_count:,}")
            print(f"   Model size: {param_size / 1024**2:.1f} MB")
    except Exception as e:
        print(f"   Could not load GOT-OCR: {e}")
    
    # Final memory state
    if torch.backends.mps.is_available():
        print(f"\nFinal MPS memory: {torch.mps.current_allocated_memory() / 1024**3:.3f} GB")


async def profile_full_pipeline_memory():
    """Profile memory usage during full pipeline execution."""
    print("\n\n=== Full Pipeline MPS Memory Profile ===")
    
    monitor = MPSMemoryMonitor()
    monitor.start()
    
    # Run a simplified pipeline
    pdf_path = Path("/Users/nikhilprasad/crown/knowledge-lib/data/pdfs/1810.03163.pdf")
    pager = PDFPagerImpl()
    layout = BatchedLayoutDetector()
    
    # Pre-load model
    layout._ensure_model()
    
    # Get pages
    page_infos = await pager.pages(pdf_path=pdf_path, max_pages=8)
    
    # Render pages
    settings = get_settings()
    images = []
    for p in page_infos:
        img = render_page_image(pdf_path, p.page_no, settings.pdf_render_dpi)
        images.append(img)
    
    # Process with monitoring
    print("\nProcessing 8 pages with batch detection...")
    start_time = time.time()
    
    with mps_memory_context("Batch Detection"):
        page_nos = [p.page_no for p in page_infos]
        results = await layout.detect_batch(images=images, page_nos=page_nos)
    
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f}s")
    
    # Stop monitoring and get stats
    stats = monitor.stop()
    
    if stats:
        print("\nMPS Memory Statistics:")
        print(f"  Peak: {stats['peak_gb']:.3f} GB")
        print(f"  Average: {stats['avg_gb']:.3f} GB")
        print(f"  Min: {stats['min_gb']:.3f} GB")
        print(f"  Max: {stats['max_gb']:.3f} GB")
        print(f"  Samples: {stats['samples']}")


def main():
    print("=" * 60)
    print("MPS GPU Memory Profiling for PDF Processing")
    print("=" * 60)
    
    # Run both profiles
    asyncio.run(profile_model_sizes())
    asyncio.run(profile_full_pipeline_memory())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if torch.backends.mps.is_available():
        current = torch.mps.current_allocated_memory()
        print(f"Current MPS memory allocated: {current / 1024**3:.3f} GB")
        print("\nNOTE: MPS on Apple Silicon uses unified memory architecture.")
        print("GPU memory is shared with system RAM, so values shown are")
        print("allocations within the unified memory pool.")
    else:
        print("MPS not available on this system.")


if __name__ == "__main__":
    main()

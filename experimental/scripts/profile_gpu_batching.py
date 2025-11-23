#!/usr/bin/env python3
"""Profile GPU batching to understand performance bottlenecks."""

import asyncio
import time
import torch
from pathlib import Path
import sys
import os
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app.services.ingest.providers.batched_providers import BatchedLayoutDetector
from src.app.services.ingest.providers.pdf_pager import PDFPagerImpl
from src.app.services.ingest.providers.utils import render_page_image
from src.app.settings import get_settings


class GPUProfiler:
    """Profile GPU operations with detailed timing."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        
    def time_section(self, name: str):
        """Context manager for timing sections."""
        class Timer:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                self.start = None
                
            def __enter__(self):
                self.start = time.perf_counter()
                return self
                
            def __exit__(self, *args):
                elapsed = time.perf_counter() - self.start
                if self.name not in self.profiler.timings:
                    self.profiler.timings[self.name] = []
                self.profiler.timings[self.name].append(elapsed)
                
        return Timer(self, name)
    
    def report(self):
        """Print timing report."""
        print("\n=== GPU Batching Profile Report ===")
        for name, times in self.timings.items():
            total = sum(times)
            avg = total / len(times) if times else 0
            print(f"\n{name}:")
            print(f"  Total: {total:.3f}s")
            print(f"  Count: {len(times)}")
            print(f"  Average: {avg:.3f}s")
            if len(times) > 1:
                print(f"  Min: {min(times):.3f}s")
                print(f"  Max: {max(times):.3f}s")


async def profile_layout_batching(pdf_path: Path, max_pages: int = 25):
    """Profile the layout detection batching in detail."""
    profiler = GPUProfiler()
    settings = get_settings()
    
    # Initialize components
    with profiler.time_section("Model initialization"):
        layout = BatchedLayoutDetector()
        layout._ensure_model()  # Pre-load model
        
    pager = PDFPagerImpl()
    
    # Get page info
    with profiler.time_section("Page enumeration"):
        page_infos = await pager.pages(pdf_path=pdf_path, max_pages=max_pages)
    
    print(f"\nProcessing {len(page_infos)} pages with batch size {layout.layout_batch_size}")
    
    # Render all pages first
    page_images = {}
    print("\nRendering pages...")
    for i, p in enumerate(page_infos):
        with profiler.time_section("Page rendering"):
            img = render_page_image(pdf_path, p.page_no, settings.pdf_render_dpi)
            page_images[p.page_no] = {"image": img}
        if (i + 1) % 5 == 0:
            print(f"  Rendered {i + 1}/{len(page_infos)} pages")
    
    # Test different batch sizes
    for batch_size in [1, 4, 8, 16]:
        if batch_size > len(page_infos):
            continue
            
        print(f"\n\nTesting batch size: {batch_size}")
        layout.layout_batch_size = batch_size
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            # MPS doesn't have empty_cache, but we can try to force collection
            import gc
            gc.collect()
        
        # Process in batches
        batch_results = []
        for i in range(0, len(page_infos), batch_size):
            batch = page_infos[i:i+batch_size]
            batch_images = [page_images[p.page_no]["image"] for p in batch]
            batch_page_nos = [p.page_no for p in batch]
            
            # Time the actual batch detection
            with profiler.time_section(f"Batch detection (size={batch_size})"):
                # Manually call the batch method
                with profiler.time_section(f"  Image preprocessing (size={batch_size})"):
                    processed_images = []
                    target_sizes = []
                    for img in batch_images:
                        width, height = img.size
                        inputs = layout._proc(images=img, return_tensors="pt")
                        processed_images.append(inputs)
                        target_sizes.append((height, width))
                
                with profiler.time_section(f"  Tensor batching (size={batch_size})"):
                    batch_inputs = {}
                    for key in processed_images[0].keys():
                        batch_inputs[key] = torch.cat([inp[key] for inp in processed_images], dim=0)
                
                with profiler.time_section(f"  GPU transfer (size={batch_size})"):
                    batch_inputs = {k: v.to(layout._device) for k, v in batch_inputs.items()}
                
                with profiler.time_section(f"  Model inference (size={batch_size})"):
                    with torch.inference_mode():
                        outputs = layout._model(**batch_inputs)
                
                with profiler.time_section(f"  Post-processing (size={batch_size})"):
                    # Post-process results
                    for j in range(len(batch)):
                        # Simplified post-processing
                        pass
    
    # Report results
    profiler.report()
    
    # GPU memory stats
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    elif torch.backends.mps.is_available():
        print(f"\nUsing MPS (Metal Performance Shaders)")
        print(f"  Current allocated: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")


async def compare_sequential_vs_batch():
    """Compare sequential vs batch processing."""
    pdf_path = Path("/Users/nikhilprasad/crown/knowledge-lib/data/pdfs/1810.03163.pdf")
    settings = get_settings()
    
    # Initialize
    layout = BatchedLayoutDetector()
    layout._ensure_model()
    pager = PDFPagerImpl()
    
    # Get pages
    page_infos = await pager.pages(pdf_path=pdf_path, max_pages=8)
    
    # Render pages
    page_images = []
    for p in page_infos:
        img = render_page_image(pdf_path, p.page_no, settings.pdf_render_dpi)
        page_images.append(img)
    
    print(f"Testing with {len(page_images)} pages")
    
    # Test 1: Sequential processing
    print("\n1. Sequential Processing (one at a time):")
    start = time.perf_counter()
    for img in page_images:
        inputs = layout._proc(images=img, return_tensors="pt")
        inputs = {k: v.to(layout._device) for k, v in inputs.items()}
        with torch.inference_mode():
            _ = layout._model(**inputs)
    seq_time = time.perf_counter() - start
    print(f"   Time: {seq_time:.3f}s ({seq_time/len(page_images):.3f}s per page)")
    
    # Test 2: True batch processing
    print("\n2. True Batch Processing (all at once):")
    start = time.perf_counter()
    
    # Process all images at once
    all_inputs = layout._proc(images=page_images, return_tensors="pt")
    all_inputs = {k: v.to(layout._device) for k, v in all_inputs.items()}
    
    with torch.inference_mode():
        _ = layout._model(**all_inputs)
    
    batch_time = time.perf_counter() - start
    print(f"   Time: {batch_time:.3f}s ({batch_time/len(page_images):.3f}s per page)")
    print(f"   Speedup: {seq_time/batch_time:.2f}x")
    
    # Test 3: Our current batching approach
    print("\n3. Current Batching Approach (manual concat):")
    start = time.perf_counter()
    
    processed_images = []
    for img in page_images:
        inputs = layout._proc(images=img, return_tensors="pt")
        processed_images.append(inputs)
    
    batch_inputs = {}
    for key in processed_images[0].keys():
        batch_inputs[key] = torch.cat([inp[key] for inp in processed_images], dim=0)
    
    batch_inputs = {k: v.to(layout._device) for k, v in batch_inputs.items()}
    
    with torch.inference_mode():
        _ = layout._model(**batch_inputs)
    
    our_time = time.perf_counter() - start
    print(f"   Time: {our_time:.3f}s ({our_time/len(page_images):.3f}s per page)")
    print(f"   Overhead vs true batch: {our_time - batch_time:.3f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Profile GPU batching performance")
    parser.add_argument("--pdf", default="/Users/nikhilprasad/crown/knowledge-lib/data/pdfs/1810.03163.pdf")
    parser.add_argument("--test", choices=["full", "compare"], default="compare")
    args = parser.parse_args()
    
    if args.test == "full":
        asyncio.run(profile_layout_batching(Path(args.pdf)))
    else:
        asyncio.run(compare_sequential_vs_batch())


if __name__ == "__main__":
    main()

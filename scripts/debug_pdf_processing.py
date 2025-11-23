#!/usr/bin/env python3
"""Debug script to identify where PDF processing is stalling."""

import asyncio
import logging
import sys
import time
from pathlib import Path
from uuid import uuid4

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set specific loggers to INFO to reduce noise
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("fitz").setLevel(logging.INFO)
logging.getLogger("torch").setLevel(logging.INFO)
logging.getLogger("transformers").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

async def debug_providers():
    """Test loading each provider individually."""
    from src.app.services.ingest.providers import resolve_providers
    
    print("\n=== Testing Provider Loading ===")
    
    # Test PDF pager
    print("\n1. Loading PDF Pager...")
    start = time.time()
    try:
        from src.app.services.ingest.providers.cached_providers import get_pdf_pager
        pager = get_pdf_pager()
        print(f"   ✓ PDF Pager loaded in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"   ✗ PDF Pager failed: {e}")
        return False
    
    # Test Layout Detector
    print("\n2. Loading Layout Detector (DETR)...")
    start = time.time()
    try:
        from src.app.services.ingest.providers.cached_providers import get_layout_detector
        layout = get_layout_detector()
        print(f"   ✓ Layout Detector loaded in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"   ✗ Layout Detector failed: {e}")
        return False
    
    # Test GOT-OCR
    print("\n3. Loading GOT-OCR Provider...")
    print("   (This may take a while if downloading the model for the first time)")
    start = time.time()
    try:
        from src.app.services.ingest.providers.cached_providers import get_got_ocr_provider
        ocr = get_got_ocr_provider()
        # Force model loading
        ocr._ensure_model()
        print(f"   ✓ GOT-OCR loaded in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"   ✗ GOT-OCR failed: {e}")
        return False
    
    return True

async def test_single_page(pdf_path: Path):
    """Test processing a single page."""
    from src.app.services.ingest.providers import resolve_providers
    
    print("\n=== Testing Single Page Processing ===")
    pager, layout, ocr = resolve_providers()
    
    # Get first page
    start = time.time()
    pages = await pager.pages(pdf_path=pdf_path, max_pages=1)
    if not pages:
        print("No pages found!")
        return
    
    page = pages[0]
    print(f"Page info: {page.page_no}, {page.width_px}x{page.height_px}")
    
    # Test layout detection
    print("\nTesting layout detection...")
    start = time.time()
    regions = await layout.detect(pdf_path=pdf_path, page_no=page.page_no)
    print(f"Found {len(regions)} regions in {time.time() - start:.2f}s")
    for i, r in enumerate(regions[:3]):  # Show first 3
        print(f"  Region {i}: {r.rtype} at ({r.bbox.x0:.2f}, {r.bbox.y0:.2f})")
    
    # Test OCR on first text region
    text_regions = [r for r in regions if r.rtype == "text"]
    if text_regions:
        print(f"\nTesting OCR on first text region...")
        start = time.time()
        text = await ocr.ocr_region(pdf_path=pdf_path, page_no=page.page_no, region=text_regions[0])
        print(f"OCR completed in {time.time() - start:.2f}s")
        print(f"Text preview: {text[:100]}...")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_pdf_processing.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Testing PDF: {pdf_path}")
    
    # Test providers
    if not await debug_providers():
        print("\nProvider loading failed. Check if models need to be downloaded.")
        return
    
    # Test single page
    await test_single_page(pdf_path)
    
    # Test full pipeline with minimal pages
    print("\n=== Testing Full Pipeline (3 pages) ===")
    from src.app.services.ingest.container_pipeline import process_pdf_container_async
    
    container_id = uuid4()
    start = time.time()
    
    try:
        # Temporarily limit to 3 pages for testing
        from src.app.settings import get_settings
        settings = get_settings()
        original_max = settings.pdf_max_pages
        settings.pdf_max_pages = 3
        
        await process_pdf_container_async(
            container_id=container_id,
            pdf_path=pdf_path
        )
        
        settings.pdf_max_pages = original_max
        print(f"\n✓ Pipeline completed in {time.time() - start:.2f}s")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

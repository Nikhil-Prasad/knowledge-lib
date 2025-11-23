#!/usr/bin/env python3
"""Test script for batched PDF processing with memory profiling."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import logging
import sys
import os
from typing import Optional
from uuid import UUID

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.add_memory_profiling import MemoryMonitor, log_system_info, get_memory_info
from src.app.db.session.session_async import AsyncSessionLocal
from src.app.services.ingest.container_pipeline import ingest_pdf_container_pipeline
from src.app.services.ingest.container_pipeline_batched import process_pdf_container_batched
from src.app.api.v1.ingest.schemas import IngestOptions


async def create_container(pdf_path: Path, collection_id: Optional[UUID]) -> UUID:
    """Create the container record."""
    async with AsyncSessionLocal() as session:
        resp = await ingest_pdf_container_pipeline(
            session,
            pdf_path=pdf_path,
            options=IngestOptions(dedupe=False),
            source_uri=f"file://{pdf_path}",
            title_hint=pdf_path.stem,
            collection_id=collection_id,
        )
        return resp.container_id


async def _run(pdf_path: Path, collection_id: Optional[UUID], use_batching: bool) -> None:
    """Run the pipeline with optional batching."""
    # Log initial system info
    log_system_info()
    
    # Log memory before starting
    print("\n[MEMORY] Initial state:")
    mem_info = get_memory_info()
    print(f"  System: {mem_info['system']['used_gb']:.2f}/{mem_info['system']['total_gb']:.2f} GB ({mem_info['system']['percent']:.1f}%)")
    print(f"  Process: {mem_info['process']['rss_gb']:.2f} GB")
    
    # Create container record
    with MemoryMonitor("Container Creation"):
        container_id = await create_container(pdf_path, collection_id)
    print(f"\nCreated container: {container_id}")
    
    # Process PDF with batching
    mode = "BATCHED" if use_batching else "SEQUENTIAL"
    print(f"\nProcessing PDF pages in {mode} mode...")
    
    with MemoryMonitor(f"{mode} PDF Processing"):
        await process_pdf_container_batched(
            container_id=container_id,
            pdf_path=pdf_path,
            use_batching=use_batching
        )
    
    # Log final memory state
    print("\n[MEMORY] Final state:")
    mem_info = get_memory_info()
    print(f"  System: {mem_info['system']['used_gb']:.2f}/{mem_info['system']['total_gb']:.2f} GB ({mem_info['system']['percent']:.1f}%)")
    print(f"  Process: {mem_info['process']['rss_gb']:.2f} GB")
    
    print("\nDone.")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Test batched PDF processing with memory profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables for batch sizes:
  LAYOUT_BATCH_SIZE     Layout detection batch size (default: 8)
  OCR_BATCH_SIZE        OCR batch size (default: 16)  
  TABLE_BATCH_SIZE      Table extraction batch size (default: 8)
  
Examples:
  # Sequential processing (baseline)
  python scripts/process_pdf_batched.py --no-batch data/pdfs/example.pdf
  
  # Batched processing (optimized)
  python scripts/process_pdf_batched.py data/pdfs/example.pdf
  
  # Custom batch sizes
  LAYOUT_BATCH_SIZE=4 OCR_BATCH_SIZE=8 python scripts/process_pdf_batched.py data/pdfs/example.pdf
"""
    )
    p.add_argument("path", help="Path to PDF file")
    p.add_argument(
        "--no-batch", 
        action="store_true", 
        help="Disable batching (use sequential processing)"
    )
    p.add_argument(
        "--collection-id", 
        default=None, 
        help="Optional collection UUID to associate"
    )
    p.add_argument(
        "--log-level", 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        help="Logging level (default: %(default)s)"
    )
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    pdf = Path(args.path).expanduser().resolve()
    if not pdf.is_file():
        print(f"Not a file: {pdf}")
        return 2

    coll: Optional[UUID] = None
    if args.collection_id:
        try:
            coll = UUID(args.collection_id)
        except Exception:
            print("Invalid collection_id; ignoring.")

    use_batching = not args.no_batch
    asyncio.run(_run(pdf, coll, use_batching))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

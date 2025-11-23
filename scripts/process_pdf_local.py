#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import logging
import time
from typing import Optional
from uuid import UUID
from functools import wraps
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.add_memory_profiling import MemoryMonitor, log_system_info, get_memory_info
from src.app.db.session.session_async import AsyncSessionLocal
from src.app.services.ingest.container_pipeline import (
    ingest_pdf_container_pipeline,
    process_pdf_container_async,
)
from src.app.api.v1.ingest.schemas import IngestOptions


# Simple profiling decorator
def profile_stage(stage_name: str):
    """Decorator to profile execution time of async functions"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"\n[PROFILE] {stage_name}: {elapsed:.3f}s")
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"\n[PROFILE] {stage_name}: {elapsed:.3f}s")
            return result
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@profile_stage("Container Creation")
async def create_container(pdf_path: Path, collection_id: Optional[UUID]) -> UUID:
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


@profile_stage("PDF Processing Pipeline") 
async def process_pdf(container_id: UUID, pdf_path: Path) -> None:
    await process_pdf_container_async(container_id=container_id, pdf_path=pdf_path)


@profile_stage("Total Execution Time")
async def _run(pdf_path: Path, collection_id: Optional[UUID]) -> None:
    # Log initial system info
    log_system_info()
    
    # Log memory before starting
    print("\n[MEMORY] Initial state:")
    mem_info = get_memory_info()
    print(f"  System: {mem_info['system']['used_gb']:.2f}/{mem_info['system']['total_gb']:.2f} GB ({mem_info['system']['percent']:.1f}%)")
    print(f"  Process: {mem_info['process']['rss_gb']:.2f} GB")
    
    # Create container record and then process in the same run (no API server needed)
    with MemoryMonitor("Container Creation"):
        container_id = await create_container(pdf_path, collection_id)
    print(f"Created container: {container_id}")
    
    print("Processing PDF pages (layout + text/ocr + figures/tables)...")
    with MemoryMonitor("PDF Processing Pipeline"):
        await process_pdf(container_id, pdf_path)
    
    # Log final memory state
    print("\n[MEMORY] Final state:")
    mem_info = get_memory_info()
    print(f"  System: {mem_info['system']['used_gb']:.2f}/{mem_info['system']['total_gb']:.2f} GB ({mem_info['system']['percent']:.1f}%)")
    print(f"  Process: {mem_info['process']['rss_gb']:.2f} GB")
    
    print("\nDone.")


def main() -> int:
    p = argparse.ArgumentParser(description="Process a local PDF through the container pipeline without starting the API")
    p.add_argument("path", help="Path to PDF file")
    p.add_argument("--collection-id", default=None, help="Optional collection UUID to associate")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Logging level (default: %(default)s)")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

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

    asyncio.run(_run(pdf, coll))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

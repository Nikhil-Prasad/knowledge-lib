#!/usr/bin/env python3
"""Script to test microbatched PDF processing with GPU lanes."""

import asyncio
import sys
import logging
from pathlib import Path
from uuid import uuid4
import time
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.add_memory_profiling import MemoryMonitor, get_memory_info, log_system_info
from src.app.services.ingest.container_pipeline_microbatched import process_pdf_container_microbatched

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)-30s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Simple memory profiler for tracking memory usage."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
        
    def start(self):
        self.start_time = time.time()
        self.start_memory = get_memory_info()
        self.peak_memory = self.start_memory['system']['used_gb']
        
    def stop(self):
        end_memory = get_memory_info()
        duration = time.time() - self.start_time
        
        return {
            'duration': duration,
            'peak_system': max(self.peak_memory, end_memory['system']['used_gb']),
            'system_change': end_memory['system']['used_gb'] - self.start_memory['system']['used_gb'],
            'process_change': end_memory['process']['rss_gb'] - self.start_memory['process']['rss_gb'],
        }
    
    def update_peak(self):
        current = get_memory_info()
        self.peak_memory = max(self.peak_memory, current['system']['used_gb'])


async def main():
    if len(sys.argv) != 2:
        print("Usage: python process_pdf_microbatched.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Log system info
    log_system_info()
    
    # Initialize memory monitoring
    profiler = MemoryProfiler()
    profiler.start()
    
    # Log initial state
    mem_info = get_memory_info()
    logger.info("\n[MEMORY] Initial state:")
    logger.info(f"  System: {mem_info['system']['used_gb']:.2f}/{mem_info['system']['total_gb']:.2f} GB")
    logger.info(f"  Process: {mem_info['process']['rss_gb']:.2f} GB")
    
    # Create container
    container_id = uuid4()
    print(f"\nCreated container: {container_id}")
    
    print("\nProcessing PDF pages in MICROBATCHED mode with GPU lanes...")
    
    # Process with microbatching
    try:
        with MemoryMonitor("PDF Processing"):
            await process_pdf_container_microbatched(
                container_id=container_id,
                pdf_path=pdf_path,
                use_microbatching=True
            )
    except Exception as e:
        print(f"\nError processing PDF: {e}")
        import traceback
        traceback.print_exc()
    
    # Log final state
    mem_info = get_memory_info()
    logger.info("\n[MEMORY] Final state:")
    logger.info(f"  System: {mem_info['system']['used_gb']:.2f}/{mem_info['system']['total_gb']:.2f} GB")
    logger.info(f"  Process: {mem_info['process']['rss_gb']:.2f} GB")
    
    # Stop monitoring and get stats
    stats = profiler.stop()
    print(f"\n[MEMORY] Summary:")
    print(f"  Duration: {stats['duration']:.2f}s")
    print(f"  Peak system memory: {stats['peak_system']:.2f} GB")
    print(f"  System memory change: {stats['system_change']:+.2f} GB")
    print(f"  Process memory change: {stats['process_change']:+.2f} GB")
    
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

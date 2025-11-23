#!/usr/bin/env python3
"""Add memory and GPU profiling utilities."""

import psutil
import torch
import functools
import time
import logging
from typing import Any, Callable, TypeVar, cast

logger = logging.getLogger(__name__)

def get_memory_info():
    """Get current memory usage info."""
    # System memory
    vm = psutil.virtual_memory()
    
    # GPU memory if available
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'reserved': torch.cuda.memory_reserved() / 1024**3,
            'device': torch.cuda.get_device_name(),
        }
    elif torch.backends.mps.is_available():
        # MPS doesn't have direct memory queries, estimate from system
        gpu_info = {
            'device': 'Apple Silicon (MPS)',
            'note': 'Shared memory with system RAM'
        }
    
    return {
        'system': {
            'total_gb': vm.total / 1024**3,
            'available_gb': vm.available / 1024**3,
            'used_gb': vm.used / 1024**3,
            'percent': vm.percent
        },
        'gpu': gpu_info,
        'process': {
            'rss_gb': psutil.Process().memory_info().rss / 1024**3,
            'vms_gb': psutil.Process().memory_info().vms / 1024**3,
        }
    }

F = TypeVar('F', bound=Callable[..., Any])

def profile_memory(func: F) -> F:
    """Decorator to profile memory usage of a function."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Before
        mem_before = get_memory_info()
        logger.info(f"\n[MEMORY] Before {func.__name__}:")
        logger.info(f"  System: {mem_before['system']['used_gb']:.2f}/{mem_before['system']['total_gb']:.2f} GB ({mem_before['system']['percent']:.1f}%)")
        logger.info(f"  Process RSS: {mem_before['process']['rss_gb']:.2f} GB")
        if 'allocated' in mem_before['gpu']:
            logger.info(f"  GPU: {mem_before['gpu']['allocated']:.2f} GB allocated")
        
        start_time = time.perf_counter()
        
        # Run function
        result = func(*args, **kwargs)
        
        # After
        mem_after = get_memory_info()
        duration = time.perf_counter() - start_time
        
        # Calculate deltas
        sys_delta = mem_after['system']['used_gb'] - mem_before['system']['used_gb']
        proc_delta = mem_after['process']['rss_gb'] - mem_before['process']['rss_gb']
        
        logger.info(f"\n[MEMORY] After {func.__name__} ({duration:.2f}s):")
        logger.info(f"  System: {mem_after['system']['used_gb']:.2f}/{mem_after['system']['total_gb']:.2f} GB ({mem_after['system']['percent']:.1f}%)")
        logger.info(f"  Process RSS: {mem_after['process']['rss_gb']:.2f} GB")
        if 'allocated' in mem_after['gpu']:
            logger.info(f"  GPU: {mem_after['gpu']['allocated']:.2f} GB allocated")
            gpu_delta = mem_after['gpu'].get('allocated', 0) - mem_before['gpu'].get('allocated', 0)
            logger.info(f"  GPU Delta: {gpu_delta:+.2f} GB")
        logger.info(f"  System Delta: {sys_delta:+.2f} GB")
        logger.info(f"  Process Delta: {proc_delta:+.2f} GB")
        
        return result
    
    return cast(F, wrapper)

class MemoryMonitor:
    """Context manager for monitoring memory usage."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_mem = None
        self.start_time = None
        
    def __enter__(self):
        self.start_mem = get_memory_info()
        self.start_time = time.perf_counter()
        logger.info(f"\n[MEMORY MONITOR] Starting {self.name}")
        logger.info(f"  System: {self.start_mem['system']['used_gb']:.2f}/{self.start_mem['system']['total_gb']:.2f} GB")
        logger.info(f"  Process: {self.start_mem['process']['rss_gb']:.2f} GB")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_mem = get_memory_info()
        duration = time.perf_counter() - self.start_time
        
        sys_delta = end_mem['system']['used_gb'] - self.start_mem['system']['used_gb']
        proc_delta = end_mem['process']['rss_gb'] - self.start_mem['process']['rss_gb']
        
        logger.info(f"\n[MEMORY MONITOR] Finished {self.name} in {duration:.2f}s")
        logger.info(f"  Peak System: {end_mem['system']['used_gb']:.2f} GB")
        logger.info(f"  Peak Process: {end_mem['process']['rss_gb']:.2f} GB")
        logger.info(f"  System Change: {sys_delta:+.2f} GB")
        logger.info(f"  Process Change: {proc_delta:+.2f} GB")
        
        # Log GPU info if available
        if 'allocated' in end_mem['gpu']:
            gpu_delta = end_mem['gpu'].get('allocated', 0) - self.start_mem['gpu'].get('allocated', 0)
            logger.info(f"  GPU Memory: {end_mem['gpu']['allocated']:.2f} GB (change: {gpu_delta:+.2f} GB)")

def log_system_info():
    """Log system information."""
    logger.info("\n[SYSTEM INFO]")
    logger.info(f"  CPU Count: {psutil.cpu_count()}")
    
    mem = psutil.virtual_memory()
    logger.info(f"  Total RAM: {mem.total / 1024**3:.1f} GB")
    logger.info(f"  Available RAM: {mem.available / 1024**3:.1f} GB")
    
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name()}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif torch.backends.mps.is_available():
        logger.info(f"  GPU: Apple Silicon (MPS) - Unified Memory Architecture")
    else:
        logger.info(f"  GPU: Not available")

if __name__ == "__main__":
    # Test the utilities
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    log_system_info()
    
    with MemoryMonitor("Test allocation"):
        # Allocate some memory
        data = [0] * (100 * 1024 * 1024)  # ~800MB
        time.sleep(1)
    
    print("\nMemory info:")
    print(get_memory_info())

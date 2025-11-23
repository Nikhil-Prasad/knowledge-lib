"""MicroBatcher for GPU-optimized streaming pipeline."""

from __future__ import annotations

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, TypeVar, Generic, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MicroBatchItem(Generic[T]):
    """Individual item in a microbatch."""
    data: T
    size_bytes: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MicroBatcher(Generic[T]):
    """
    Collects items into microbatches based on three flush triggers:
    1. max_items: Maximum number of items in a batch
    2. max_bytes: Maximum total size in bytes
    3. max_latency_ms: Maximum time since first item arrived
    """
    
    def __init__(
        self,
        max_items: int = 8,
        max_bytes: int = 256 * 1024 * 1024,  # 256MB
        max_latency_ms: int = 50,
        size_calculator: Optional[Callable[[T], int]] = None
    ):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.max_latency_ms = max_latency_ms / 1000.0  # Convert to seconds
        self.size_calculator = size_calculator or self._default_size_calculator
        
        self.buffer: List[MicroBatchItem[T]] = []
        self.total_bytes = 0
        self.first_item_time: Optional[float] = None
        
    def _default_size_calculator(self, item: T) -> int:
        """Default size calculator - estimates based on image dimensions if available."""
        if hasattr(item, 'size'):  # PIL Image
            w, h = item.size
            return w * h * 3  # RGB bytes
        elif hasattr(item, 'shape'):  # NumPy/Torch tensor
            import numpy as np
            return np.prod(item.shape) * item.itemsize
        else:
            # Rough estimate
            return len(str(item))
    
    def add(self, item: T, metadata: Optional[Dict[str, Any]] = None) -> Optional[List[MicroBatchItem[T]]]:
        """
        Add an item to the batch.
        Returns a list of items to process if any flush trigger is met, None otherwise.
        """
        size_bytes = self.size_calculator(item)
        timestamp = time.time()
        
        batch_item = MicroBatchItem(
            data=item,
            size_bytes=size_bytes,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        # If this is the first item, record the time
        if not self.buffer:
            self.first_item_time = timestamp
            
        self.buffer.append(batch_item)
        self.total_bytes += size_bytes
        
        # Check flush triggers
        if self._should_flush():
            return self._flush()
        
        return None
    
    def _should_flush(self) -> bool:
        """Check if any flush trigger is met."""
        if not self.buffer:
            return False
            
        # Trigger 1: Max items reached
        if len(self.buffer) >= self.max_items:
            logger.debug(f"Flush trigger: max_items ({len(self.buffer)}/{self.max_items})")
            return True
            
        # Trigger 2: Max bytes reached
        if self.total_bytes >= self.max_bytes:
            logger.debug(f"Flush trigger: max_bytes ({self.total_bytes}/{self.max_bytes})")
            return True
            
        # Trigger 3: Max latency reached
        if self.first_item_time is not None:
            elapsed = time.time() - self.first_item_time
            if elapsed >= self.max_latency_ms:
                logger.debug(f"Flush trigger: max_latency ({elapsed:.3f}s/{self.max_latency_ms:.3f}s)")
                return True
                
        return False
    
    def maybe_flush_due_to_time(self) -> Optional[List[MicroBatchItem[T]]]:
        """Check if we should flush due to timeout. Called periodically."""
        if self._should_flush():
            return self._flush()
        return None
    
    def force_flush(self) -> Optional[List[MicroBatchItem[T]]]:
        """Force flush any pending items."""
        if self.buffer:
            return self._flush()
        return None
    
    def _flush(self) -> List[MicroBatchItem[T]]:
        """Flush the current buffer and reset state."""
        items = self.buffer[:]
        
        # Log batch info
        logger.info(
            f"Flushing microbatch: items={len(items)}, "
            f"bytes={self.total_bytes:,}, "
            f"age={time.time() - self.first_item_time:.3f}s"
        )
        
        # Reset state
        self.buffer = []
        self.total_bytes = 0
        self.first_item_time = None
        
        return items
    
    @property
    def pending_count(self) -> int:
        """Number of items currently in the buffer."""
        return len(self.buffer)
    
    @property
    def pending_bytes(self) -> int:
        """Total bytes of items currently in the buffer."""
        return self.total_bytes
    
    @property
    def age_seconds(self) -> float:
        """Age of the oldest item in seconds."""
        if self.first_item_time is None:
            return 0.0
        return time.time() - self.first_item_time


class AsyncMicroBatcher(MicroBatcher[T]):
    """Async version of MicroBatcher with automatic time-based flushing."""
    
    def __init__(
        self,
        *args,
        flush_callback: Optional[Callable[[List[MicroBatchItem[T]]], asyncio.Task]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.flush_callback = flush_callback
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
    async def add_async(
        self, 
        item: T, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[List[MicroBatchItem[T]]]:
        """Add an item asynchronously with automatic flush scheduling."""
        async with self._lock:
            result = self.add(item, metadata)
            
            # Schedule a time-based flush if this is the first item
            if len(self.buffer) == 1 and self._flush_task is None:
                self._flush_task = asyncio.create_task(self._schedule_flush())
                
            return result
    
    async def _schedule_flush(self):
        """Schedule a flush after max_latency_ms."""
        try:
            await asyncio.sleep(self.max_latency_ms)
            async with self._lock:
                items = self.maybe_flush_due_to_time()
                if items and self.flush_callback:
                    await self.flush_callback(items)
        finally:
            self._flush_task = None
    
    async def close(self):
        """Close the batcher and flush any pending items."""
        if self._flush_task:
            self._flush_task.cancel()
            
        async with self._lock:
            items = self.force_flush()
            if items and self.flush_callback:
                await self.flush_callback(items)

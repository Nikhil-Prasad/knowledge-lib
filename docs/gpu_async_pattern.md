# GPU Operations in Async Event Loops

## The Problem: PyTorch GPU Operations Block

```python
# This BLOCKS the event loop!
async def bad_gpu_inference():
    result = model(batch)  # Blocks until GPU completes
    return result
```

Even though the function is `async`, the PyTorch call is synchronous and blocks.

## Solution: Offload to Thread + Batch

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class LayoutDetector:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)  # Single thread for GPU
        self.model = load_model()
    
    async def detect_batch(self, images):
        """Async wrapper that doesn't block event loop"""
        # Offload the blocking GPU operation to a thread
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._sync_detect_batch,  # Synchronous function
            images
        )
    
    def _sync_detect_batch(self, images):
        """Synchronous batch processing on GPU"""
        # This runs in a separate thread
        batch = torch.stack(images)
        with torch.no_grad():
            results = self.model(batch)  # GPU processes entire batch
        return results
```

## Why We Need the Thread

1. **Event Loop Must Stay Responsive**
   - Other async operations need to run (database, network, etc.)
   - GPU operations can take 100ms+ and would freeze everything

2. **GPU Operations are Inherently Synchronous**
   - CUDA/MPS operations block until complete
   - No async GPU operations in PyTorch

3. **But Only One Thread for GPU**
   - Multiple threads can't speed up GPU access
   - They'd just queue up waiting for GPU
   - One thread is sufficient

## Complete Pattern for Batch Processing

```python
class OptimizedLayoutDetector:
    def __init__(self, batch_size=4):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.model = load_model()
        self.batch_size = batch_size
        self.pending_tasks = []
        
    async def detect(self, pdf_path: Path, page_no: int):
        """Single page interface that internally batches"""
        # Add to pending batch
        future = asyncio.Future()
        self.pending_tasks.append((pdf_path, page_no, future))
        
        # Process batch if full
        if len(self.pending_tasks) >= self.batch_size:
            await self._process_batch()
        
        return await future
    
    async def _process_batch(self):
        """Process accumulated pages as batch"""
        if not self.pending_tasks:
            return
            
        # Extract batch
        batch = self.pending_tasks[:self.batch_size]
        self.pending_tasks = self.pending_tasks[self.batch_size:]
        
        # Prepare images
        images = [load_image(pdf, page) for pdf, page, _ in batch]
        
        # Run GPU inference in thread
        results = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._sync_detect_batch,
            images
        )
        
        # Resolve futures
        for (_, _, future), result in zip(batch, results):
            future.set_result(result)
    
    def _sync_detect_batch(self, images):
        """Runs in thread pool"""
        batch = torch.stack(images)
        with torch.no_grad():
            return self.model(batch)
```

## Key Points

1. **Yes, we need a worker thread** - to prevent blocking the event loop
2. **But only one thread** - multiple threads don't help GPU operations
3. **Inside the thread, we batch** - to maximize GPU utilization
4. **The thread is just a bridge** - from async world to sync GPU operations

## In the Current Codebase

The current implementation doesn't use batching yet:
```python
# Current (sequential, but async-safe)
for p in page_infos:
    regions = await layout.detect(pdf_path=pdf_path, page_no=p.page_no)
```

The providers likely use `run_in_executor` internally to stay async-safe, but process one page at a time. The optimization would be to batch these calls.

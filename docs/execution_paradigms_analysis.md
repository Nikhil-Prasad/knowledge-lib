# Execution Paradigms for ML Pipeline

## Current Situation

Our pipeline has **mixed workloads**:
- **I/O-bound**: Database queries, file reading/writing, API calls
- **Compute-bound**: Model inference (DETR, GOT-OCR, Table Transformer, DePlot)

## The Problem with Pure Async

```python
# Current approach - misleading!
async def detect(self, *, pdf_path: Path, page_no: int) -> List[LayoutRegion]:
    # This is NOT actually async - it blocks!
    with torch.inference_mode():
        outputs = self._model(**inputs)  # Blocks the event loop
```

The `async` wrapper doesn't make PyTorch inference asynchronous. It still blocks the event loop, preventing other coroutines from running.

## Better Paradigms

### 1. **Thread Pool Executor** (Recommended for current setup)
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class LayoutDetectorDetr:
    _executor = ThreadPoolExecutor(max_workers=2)  # Shared executor
    
    async def detect(self, *, pdf_path: Path, page_no: int) -> List[LayoutRegion]:
        loop = asyncio.get_event_loop()
        # Run blocking inference in thread pool
        return await loop.run_in_executor(
            self._executor,
            self._detect_sync,
            pdf_path,
            page_no
        )
    
    def _detect_sync(self, pdf_path: Path, page_no: int) -> List[LayoutRegion]:
        # Actual synchronous inference
        with torch.inference_mode():
            # ... model inference ...
```

**Pros:**
- Doesn't block the event loop
- Other I/O operations can proceed
- Works well with MPS/GPU (PyTorch releases GIL during inference)
- Minimal code changes

**Cons:**
- Still subject to Python overhead
- Thread pool overhead for small inferences

### 2. **Process Pool** (For CPU-heavy workloads)
```python
from multiprocessing import Pool

class BatchProcessor:
    def __init__(self):
        self.pool = Pool(processes=4)
    
    async def process_pages(self, pdf_path: Path, pages: List[int]):
        loop = asyncio.get_event_loop()
        # Process pages in parallel across processes
        results = await loop.run_in_executor(
            None,
            self.pool.map,
            partial(process_single_page, pdf_path),
            pages
        )
```

**Pros:**
- True parallelism for CPU-bound work
- Bypasses GIL completely

**Cons:**
- High memory overhead (model loaded per process)
- IPC overhead
- Doesn't play well with MPS/CUDA (GPU context issues)

### 3. **Hybrid Approach** (Most realistic)
```python
class PipelineOrchestrator:
    def __init__(self):
        # I/O operations stay async
        self.db_session = AsyncSession()
        
        # Compute operations use thread pool
        self.inference_executor = ThreadPoolExecutor(max_workers=2)
        
        # Batch processing for efficiency
        self.batch_size = 4
    
    async def process_container(self, container_id: UUID, pdf_path: Path):
        # I/O: Async page discovery
        pages = await self._get_pages_async()
        
        # Compute: Batch pages for efficient GPU usage
        for batch in chunks(pages, self.batch_size):
            # Run batch inference in thread pool
            results = await self._run_batch_inference(batch)
            
            # I/O: Async database writes
            await self._persist_results_async(results)
```

### 4. **Pure Synchronous with Queue Workers** (Production-ready)
```python
# Worker process/container
def worker_main():
    # Load models once
    detector = LayoutDetectorDetr()
    ocr = GOTOCRProvider()
    
    while True:
        # Pull job from queue (Redis/RabbitMQ/Celery)
        job = queue.get_job()
        
        # Process synchronously
        results = process_pdf(job.pdf_path, detector, ocr)
        
        # Push results
        queue.put_results(job.id, results)

# API remains async for I/O
async def ingest_pdf_api(pdf_path: Path):
    # Quick async operations
    container = await create_container_async()
    
    # Queue for processing
    await queue_job(container.id, pdf_path)
    
    return {"status": "queued", "container_id": container.id}
```

## Recommendations

### For Development (Current Stage):
Use **Thread Pool Executor** pattern:
- Minimal refactoring
- Keeps async API for consistency
- Good enough performance
- Works well with MPS

### For Production:
Use **Queue + Workers** pattern:
- Separate API from inference
- Scale workers independently
- Better resource utilization
- Fault tolerance

### Implementation Example:

```python
# providers/base.py
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

class AsyncModelProvider(ABC):
    """Base class for providers with compute-bound operations."""
    
    _executor = ThreadPoolExecutor(max_workers=2)  # Shared across providers
    
    @abstractmethod
    def _process_sync(self, *args, **kwargs):
        """Synchronous processing method."""
        pass
    
    async def process(self, *args, **kwargs):
        """Async wrapper using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._process_sync,
            *args,
            **kwargs
        )
```

## Key Insights

1. **Don't fake async**: If it's compute-bound, acknowledge it
2. **Batch when possible**: GPU inference is more efficient in batches
3. **Separate concerns**: I/O async, compute sync/threaded
4. **Profile first**: MPS behavior differs from CUDA
5. **Consider memory**: Models are large; loading strategy matters

## MPS-Specific Considerations

```python
# MPS (Apple Silicon) handles threading well
torch.set_num_threads(1)  # Prevent oversubscription
os.environ["OMP_NUM_THREADS"] = "1"

# MPS operations release GIL effectively
# Thread pool works well for MPS inference

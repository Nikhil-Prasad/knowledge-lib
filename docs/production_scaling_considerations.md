# Production Scaling: Batching vs Streaming

## Memory Footprint Comparison

### Current Sequential Pipeline (Stream Processing)
```
1 request × 1 page in memory = ~50MB
30 concurrent requests × 1 page = ~1.5GB peak memory
```

### Batched Pipeline
```
1 request × 25 pages in memory = ~1.25GB
30 concurrent requests × 25 pages = ~37.5GB peak memory (!!)
```

## The Multi-Tenant Dilemma

### Single Request Optimization
- Batching wins: 18s → 3s
- But requires holding all pages in memory

### Multi-Request Environment
- Sequential wins: Lower memory footprint
- But each request takes longer

## Production Architecture Patterns

### 1. Request Queuing with Batch Aggregation
```python
class GPUBatchAggregator:
    """Collect pages from multiple requests, batch together"""
    
    def __init__(self, batch_timeout=100ms):
        self.pending_pages = []  # [(request_id, page)]
        self.batch_size = 8
        
    async def add_page(self, request_id, page):
        self.pending_pages.append((request_id, page))
        
        # Process if we have enough pages OR timeout
        if len(self.pending_pages) >= self.batch_size:
            await self.process_batch()
    
    async def process_batch(self):
        # Mix pages from different requests!
        batch = self.pending_pages[:self.batch_size]
        results = gpu_model(batch)
        
        # Route results back to correct requests
        for (req_id, page), result in zip(batch, results):
            await send_to_request(req_id, result)
```

### 2. Hybrid Approach
```python
# Small PDFs: Process immediately (sequential)
if num_pages < 10:
    process_sequential()  # Low memory

# Large PDFs: Queue for batch processing
else:
    queue_for_batch_worker()  # Dedicated resources
```

### 3. Dynamic Memory Management
```python
class AdaptiveProcessor:
    def __init__(self):
        self.memory_limit = 4GB
        self.current_usage = 0
        
    async def process_request(self, pdf):
        estimated_memory = pdf.num_pages * 50MB
        
        if self.current_usage + estimated_memory < self.memory_limit:
            # We have room for batching
            return await process_batched(pdf)
        else:
            # Fall back to streaming
            return await process_sequential(pdf)
```

## GPU Sharing Strategies

### 1. Time-Sliced GPU Sharing
```python
# Each request gets GPU time slices
async def gpu_scheduler():
    while True:
        for request in active_requests:
            # Process one batch for this request
            await process_one_batch(request)
            # Yield to next request
            await asyncio.sleep(0)
```

### 2. Model Instance Pooling
```python
class ModelPool:
    def __init__(self, num_instances=3):
        # Load model multiple times
        self.models = [load_model() for _ in range(num_instances)]
        self.available = asyncio.Queue()
        
        for model in self.models:
            self.available.put_nowait(model)
    
    async def process(self, data):
        model = await self.available.get()
        try:
            return model(data)
        finally:
            await self.available.put(model)
```

## Memory Calculations

### Page Image Sizes (200 DPI)
- Letter size: 1700×2200 pixels
- RGB: 1700×2200×3 = 11.2MB raw
- WebP compressed: ~2MB on disk
- In memory (tensor): ~15MB per page

### Model Memory
- DETR Layout: ~500MB
- GOT-OCR: ~1GB
- Table Transformer: ~300MB
- DePlot: ~1GB
- Total models: ~2.8GB constant

### Request Memory
- Sequential: Models + 1 page = ~2.85GB
- Batched: Models + 25 pages = ~3.2GB
- 30 concurrent sequential: 30×15MB = 450MB variable
- 30 concurrent batched: 30×375MB = 11.25GB variable!

## Recommendations

### For APIs/Multi-Tenant
1. **Keep sequential processing** as default
2. **Add request queuing** to mix batches across requests
3. **Monitor memory** and fall back to sequential
4. **Use GPU time-slicing** for fairness

### For Batch Jobs/Single-Tenant
1. **Use full batching** for maximum speed
2. **Scale horizontally** with multiple GPU workers
3. **Process largest PDFs first** (better GPU utilization)

### Hybrid Best Practice
```python
class ProductionPipeline:
    def __init__(self):
        self.queue = BatchQueue()
        self.memory_monitor = MemoryMonitor()
        
    async def process(self, pdf):
        if pdf.num_pages < 10:
            # Small PDF: process immediately
            return await self.process_sequential(pdf)
        elif self.memory_monitor.has_capacity(pdf):
            # Medium PDF with available memory: batch
            return await self.process_batched(pdf)
        else:
            # Large PDF or high load: queue it
            return await self.queue.add(pdf)
```

## Key Insight

The optimal strategy depends on workload:
- **High volume, small PDFs**: Sequential + cross-request batching
- **Low volume, large PDFs**: Full batching per request
- **Mixed workload**: Adaptive hybrid approach

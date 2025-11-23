# Concurrency Patterns by Bottleneck Type

## 1. I/O-Bound → AsyncIO / Event Loop

**When to use**: Network requests, database queries, file system operations (when not CPU-intensive)

```python
# Example: Multiple API calls
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(session.get(url))
        results = await asyncio.gather(*tasks)
        
# Example: Database operations
async def process_records():
    async with asyncpg.create_pool() as pool:
        async with pool.acquire() as conn:
            await conn.fetch("SELECT * FROM table")
```

**Why it works**:
- Single thread yields control while waiting for I/O
- No actual computation happening during wait
- Event loop manages thousands of concurrent operations
- Zero CPU overhead for waiting

## 2. CPU-Bound → ThreadPoolExecutor (with GIL release)

**When to use**: C/C++ extensions, NumPy operations, image processing, cryptography

```python
# Example: PyMuPDF rendering (C++ library)
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for page_no in range(num_pages):
        future = executor.submit(render_page, page_no)  # C++ releases GIL
        futures.append(future)
    
    results = [f.result() for f in futures]

# Example: NumPy operations
def compute_heavy(data):
    return np.fft.fft(data)  # NumPy releases GIL

with ThreadPoolExecutor() as executor:
    results = executor.map(compute_heavy, datasets)
```

**Why it works**:
- C/C++ libraries release Python's GIL
- Each thread can use a different CPU core
- True parallel execution on multi-core CPUs
- Limited by number of CPU cores

## 3. GPU-Bound → Batching

**When to use**: Deep learning models, GPU-accelerated computation

```python
# Bad: Sequential processing
for image in images:
    result = model(image)  # GPU processes 1 at a time

# Good: Batch processing
batch = torch.stack(images)
results = model(batch)  # GPU processes all at once

# Better: Dynamic batching with memory management
def process_with_dynamic_batching(items, max_batch_size=32):
    for i in range(0, len(items), max_batch_size):
        batch = items[i:i+max_batch_size]
        yield model(batch)
```

**Why it works**:
- GPU has thousands of cores that work in parallel
- Single operation overhead is high
- Batch operations utilize GPU parallelism
- Memory transfer overhead amortized

## Special Cases

### Mixed I/O and CPU-Bound
```python
# ProcessPoolExecutor for pure Python CPU-bound
with ProcessPoolExecutor() as executor:
    results = executor.map(pure_python_compute, data)
```

### CPU-Bound Python (can't release GIL)
```python
# Use multiprocessing instead of threading
from multiprocessing import Pool
with Pool() as pool:
    results = pool.map(python_heavy_compute, data)
```

## Quick Decision Tree

1. **Waiting for something?** → AsyncIO
   - Network, disk I/O, database
   - `async/await`, `asyncio.gather()`, `TaskGroup`

2. **Crunching numbers on CPU?**
   - C/C++ library → ThreadPoolExecutor
   - Pure Python → ProcessPoolExecutor
   - NumPy/SciPy → ThreadPoolExecutor

3. **Using GPU?** → Batching
   - Collect inputs into batches
   - Process entire batch at once
   - Consider memory limits

## In This Project

- **Page Rendering** (PyMuPDF/C++): ThreadPoolExecutor ✅
- **Network Embedding Calls**: AsyncIO (already async) ✅  
- **Layout Detection** (DETR/GPU): Should use batching ⏳
- **OCR** (GOT-OCR/GPU): Should use batching ⏳
- **Database Operations**: Already using AsyncIO ✅

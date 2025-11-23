# GPU-Bound vs I/O-Bound Concurrency

## Current Models in the Pipeline

1. **Layout Detection**: DETR model (cmarkea/detr-layout-detection)
2. **OCR**: GOT-OCR-2.0 model (stepfun-ai/GOT-OCR-2.0-hf) 
3. **Table Structure**: Table Transformer models
4. **Figure Processing**: DePlot model (disabled in current version)

## Why ThreadPoolExecutor Works for Page Rendering but Not Models

### Page Rendering (CPU-Bound) ✅
- PyMuPDF uses C++ for rasterization (CPU-intensive computation)
- Multiple CPU cores can render different pages in parallel
- ThreadPoolExecutor distributes work across CPU cores
- The disk I/O (saving images) is minimal compared to rendering computation
- Each thread gets its own CPU core to do the math

### Model Inference (GPU-Bound) ⚠️
- Models compete for GPU memory and compute
- GPU has limited parallelism (unlike CPU threads)
- Model switching has overhead (loading/unloading from GPU)
- Memory constraints - can't load multiple large models simultaneously

## Why ThreadPoolExecutor Doesn't Help GPU Operations

The key difference is resource architecture:

### Multiple CPU Cores
- 4+ independent processing units
- Each thread can use a different core
- True parallel execution
- ThreadPoolExecutor maps threads to cores

### Single GPU
- One unified processing unit (with many small cores)
- All operations funnel through same GPU
- GPU scheduler handles parallelism internally
- Multiple Python threads just create a queue waiting for GPU

## Better Approaches for GPU Concurrency

### 1. Model Batching (Recommended)
```python
# Instead of processing pages one by one
for page in pages:
    result = model(page)  # GPU processes 1 page

# Process multiple pages in one batch
batch = [page1, page2, page3, page4]
results = model(batch)  # GPU processes 4 pages at once
```

### 2. Pipeline Parallelism
```python
# Run different models on different pages simultaneously
# Page 1: Layout Detection
# Page 2: OCR (using results from previous layout detection)
# Page 3: Table extraction
# Like an assembly line!
```

### 3. Model-Specific Optimizations
- **Shared backbone**: DETR for both layout and table detection
- **Model caching**: Keep frequently used models in GPU memory
- **Dynamic batching**: Batch similar operations together

## Memory Constraints on MPS (Apple Silicon)

- Shared memory between CPU and GPU
- Loading multiple models can cause memory pressure
- Better to optimize single model usage than parallelize

## Recommended Approach

For this pipeline, the best optimization would be:

1. **Batch Processing**: Collect multiple pages and process them together
2. **Model Warmup**: Keep models loaded between pages
3. **Async I/O**: Overlap disk I/O with GPU processing
4. **Smart Scheduling**: Process similar operations together

Example implementation:
```python
# Batch layout detection for multiple pages
async def process_pages_batched(pages, batch_size=4):
    for i in range(0, len(pages), batch_size):
        batch = pages[i:i+batch_size]
        
        # Layout detection on batch
        layouts = await layout_model.detect_batch(batch)
        
        # OCR on regions that need it
        ocr_tasks = []
        for page_layout in layouts:
            if needs_ocr(page_layout):
                ocr_tasks.append(page_layout)
        
        if ocr_tasks:
            ocr_results = await ocr_model.process_batch(ocr_tasks)
```

This would be more efficient than ThreadPoolExecutor for GPU operations!

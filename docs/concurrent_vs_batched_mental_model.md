# Mental Model: Concurrent vs Batched Processing

## The Key Insight: Pipeline Efficiency

### Optimized Concurrent Approach (18.7s)
```
Time →
Page 1: [Render] → [Layout] → [OCR] → Done
Page 2:    [Render] → [Layout] → [OCR] → Done  
Page 3:       [Render] → [Layout] → [OCR] → Done
Page 4:          [Render] → [Layout] → [OCR] → Done
...
```
- **Rendering**: Parallel (ThreadPoolExecutor with 4 workers)
- **Layout Detection**: Sequential, but starts as soon as each page is ready
- **Total time**: Dominated by the longest pipeline

### Batched Approach (28.4s)
```
Time →
Step 1: [Render ALL 25 pages] ........................→
Step 2:                                                  [Collect into batches] →
Step 3:                                                                           [GPU Batch 1] → [GPU Batch 2] → [GPU Batch 3] → [GPU Batch 4]
Step 4:                                                                                                                                      [OCR/Tables]
```
- **Must wait** for ALL pages to render before ANY layout detection
- **Batching overhead**: Collecting, concatenating tensors, moving to GPU
- **No pipeline overlap**: Each stage blocks the next

## Why Concurrent Wins

### 1. Pipeline Parallelism
In the concurrent approach, while page 5 is rendering, page 1 might already be through layout detection and into OCR. This creates a smooth pipeline where work is always happening.

### 2. Immediate Processing
```python
# Concurrent: Process immediately
for page in pages:
    image = render(page)  # As soon as this finishes...
    layout = detect(image)  # ...this starts immediately

# Batched: Must wait for all
images = [render(page) for page in pages]  # Wait for ALL
batches = create_batches(images)  # Additional overhead
for batch in batches:
    layouts = detect_batch(batch)  # Finally process
```

### 3. Memory Access Pattern
- **Concurrent**: Image is in CPU cache, immediately processed
- **Batched**: Images go to memory, later retrieved, batched, then processed

## The Numbers Breakdown

From our profiling:
- **Page rendering**: ~0.066s per page (1.66s for 25 pages with 4 workers)
- **Layout detection**: ~0.3s per page on GPU

### Concurrent Timeline:
```
0s: Start rendering pages 1-4
0.066s: Page 1 ready → Start layout detection
0.132s: Page 2 ready → Queue for layout
...
1.66s: All pages rendered
1.66s + (25 * 0.3s) = ~9s for all layout detection
(But OCR overlaps, so total ~18.7s)
```

### Batched Timeline:
```
0s: Start rendering pages 1-4
1.66s: ALL pages must be rendered
1.66s-2s: Collect and batch images (overhead)
2s-25s: Batch processing (includes overhead)
25s-28s: Final steps
Total: ~28.4s
```

## The GPU Batching Paradox

You'd expect GPU batching to be faster because:
- GPUs are good at parallel processing
- Batch of 8 should be ~8x faster than sequential

But in reality:
1. **Small models** (163MB) don't saturate the GPU
2. **Memory transfer overhead** (CPU→GPU) for batches
3. **Framework overhead** in PyTorch for batch assembly
4. **Lost pipeline efficiency** from waiting

## When Batching Would Win

Batching would be faster if:
1. **Larger models** that fully utilize GPU (e.g., 10GB+ models)
2. **More complex processing** where GPU computation >> transfer overhead
3. **True parallel needs** like training where gradients must be synchronized

## Optimal Strategy for Your Use Case

Your concurrent approach is optimal because:
1. **Models are small** - GPU isn't the bottleneck
2. **Pipeline efficiency** - Always doing useful work
3. **Lower memory** - Process and discard vs accumulate
4. **Better latency** - First results come faster

The key insight: **Don't optimize for GPU utilization, optimize for end-to-end latency.**

# Batch Timing Analysis: The Real Problem

## The Numbers Don't Add Up!

### Sequential/Concurrent Approach:
- **Total page processing** (layout + OCR + extraction): 12.7s for 25 pages
- That's ~0.51s per page for EVERYTHING

### Batched Approach:
- **Layout detection alone**: 23s for 25 pages
- That's ~0.92s per page for JUST layout detection!

## The Root Cause: Model Loading Overhead

Looking at our profiling results:
```
Sequential: 0.320s per page
True batch: 0.293s per page (only 1.09x speedup!)
Our batching: 0.765s per page (2.4x SLOWER!)
```

## What's Really Happening

### In the Optimized Concurrent Pipeline:
1. Model loads ONCE at startup
2. Model stays warm in GPU memory
3. Each page processes through quickly: ~0.32s

### In the Batched Pipeline:
Looking at the logs, the issue is likely:
1. Model initialization happens INSIDE the batch processing
2. The processor initialization happens multiple times
3. Memory allocation/deallocation overhead

## The Evidence

From `results_optimized_batch.txt`:
```
2025-11-23 15:47:54,602 INFO: Starting batched layout detection for 25 pages (batch_size=8)
Using a slow image processor as `use_fast` is unset...  # This shouldn't happen every time!
2025-11-23 15:47:57,125 INFO: Loading pretrained weights...  # Model loading DURING processing!
...
2025-11-23 16:01:20,784 INFO: [PROFILE] Batched layout detection: 23.023s
```

The model is being loaded/initialized DURING the batch processing, not before!

## The Fix

The issue is in our batched implementation. The model should be:
1. Loaded ONCE before processing starts
2. Kept warm in memory
3. Reused for all batches

Instead, it seems we're:
1. Loading the model during processing
2. Possibly reinitializing for each batch
3. Adding massive overhead

## Why Sequential is Faster

In the sequential approach:
- Model loads once: ~2-3s
- Each page: ~0.32s × 25 = 8s
- Total: ~11s

In the batched approach:
- Model loads during processing: ~2-3s
- Overhead per batch: Unknown but significant
- Processing: Slower due to overhead
- Total: 23s (just for layout!)

## The Lesson

Batching isn't inherently slower - our implementation has a bug where the model initialization happens at the wrong time, adding 10+ seconds of overhead. The actual GPU processing would be nearly identical speed, but we're paying a huge initialization cost.

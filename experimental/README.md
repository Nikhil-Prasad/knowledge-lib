# Experimental Batching Implementations

This folder contains experimental implementations for batch processing PDF pages using GPU batching strategies. These were created to investigate potential performance improvements but ultimately showed that the current concurrent processing approach is more efficient.

## Contents

### `/ingest/`
- `container_pipeline_batched.py` - Batched version of the container pipeline
- `container_pipeline_microbatched.py` - Microbatched version with GPU lane management
- `microbatcher.py` - Dynamic microbatching utility
- `gpu_lanes.py` - GPU lane management for concurrent processing

### `/providers/`
- `batched_providers.py` - Batched versions of the model providers (HF, DETR, GOT-OCR)

### `/scripts/`
- `process_pdf_batched.py` - Test script for batched processing
- `process_pdf_microbatched.py` - Test script for microbatched processing
- `profile_gpu_batching.py` - GPU batching profiler

## Key Findings

The experiments revealed that:

1. **Current approach is optimal**: The existing concurrent processing (4 workers) achieves 18-20s for a 25-page PDF
2. **Batching added overhead**: Manual batching increased processing time to 41-48s due to:
   - Sequential batch processing blocking
   - CPU-GPU synchronization overhead
   - Memory management complexity

3. **GPU utilization patterns**: Apple Silicon's unified memory architecture handles concurrent requests efficiently without explicit batching

## Why These Approaches Were Abandoned

1. **No performance gain**: Batching actually decreased performance by 2-2.5x
2. **Added complexity**: Required significant code changes without benefits
3. **Hardware mismatch**: Apple Silicon GPUs handle concurrent operations differently than traditional GPUs

## Recommendations

The current concurrent processing approach should be maintained as it provides:
- Better performance (18-20s vs 41-48s)
- Simpler code structure
- Better memory efficiency
- More predictable behavior

These experimental implementations are preserved for reference and potential future investigation on different hardware architectures.

# Batch Pipeline Architecture for GPU Optimization

## Current Sequential Pipeline (Page-by-Page)

```python
for page in pages:
    # Each page fully processed before moving to next
    layout = detect_layout(page)          # GPU call #1
    for region in layout:
        if region.type == "text":
            text = ocr(region)            # GPU call #2
        elif region.type == "figure":
            caption = deplot(region)      # GPU call #3
        elif region.type == "table":
            structure = table_trans(region) # GPU call #4
    save_to_db(page_results)
```

**Problem**: GPU processes one small item at a time = inefficient

## Optimized Batch Pipeline (Stage-by-Stage)

```python
# Stage 1: Layout Detection (all pages)
all_layouts = []
for batch in chunk_pages(pages, batch_size=8):
    layouts = layout_model(batch)  # GPU processes 8 pages at once
    all_layouts.extend(layouts)

# Stage 2: Categorize regions
text_regions = []
figure_regions = []
table_regions = []
for page_layout in all_layouts:
    for region in page_layout.regions:
        if region.type == "text":
            text_regions.append((page_no, region))
        elif region.type == "figure":
            figure_regions.append((page_no, region))
        elif region.type == "table":
            table_regions.append((page_no, region))

# Stage 3: OCR all text regions
text_results = []
for batch in chunk_items(text_regions, batch_size=16):
    results = ocr_model(batch)  # GPU processes 16 text regions
    text_results.extend(results)

# Stage 4: Process all figures
figure_results = []
for batch in chunk_items(figure_regions, batch_size=4):
    results = deplot_model(batch)  # GPU processes 4 figures
    figure_results.extend(results)

# Stage 5: Process all tables
table_results = []
for batch in chunk_items(table_regions, batch_size=8):
    results = table_transformer(batch)  # GPU processes 8 tables
    table_results.extend(results)

# Stage 6: Batch database write
batch_save_to_db(text_results + figure_results + table_results)
```

## Memory Management Strategy

```python
class BatchPipelineProcessor:
    def __init__(self):
        self.layout_batch_size = 8     # Pages
        self.ocr_batch_size = 16       # Text regions
        self.figure_batch_size = 4     # Figures (DePlot is memory heavy)
        self.table_batch_size = 8      # Tables
        
        # Keep everything in memory during processing
        self.page_images = {}          # {page_no: image_tensor}
        self.layouts = {}              # {page_no: layout_result}
        self.text_regions = []         # [(page_no, region, image_crop)]
        self.figure_regions = []       # [(page_no, region, image_crop)]
        self.table_regions = []        # [(page_no, region, image_crop)]
```

## Advantages of Stage-by-Stage

1. **GPU Efficiency**
   - Layout: 8 pages/batch vs 1 page → ~6x faster
   - OCR: 16 regions/batch vs 1 region → ~12x faster
   - Tables: 8 tables/batch vs 1 table → ~6x faster

2. **Model Loading**
   - Load each model once, process all relevant items
   - No constant switching between models

3. **Memory Patterns**
   - Predictable memory usage per stage
   - Can tune batch sizes per model

## Trade-offs

### Pros:
- Maximum GPU utilization
- Minimal model switching overhead
- Efficient memory access patterns
- Easy to parallelize stages (e.g., OCR and DePlot simultaneously)

### Cons:
- Higher peak memory usage (storing intermediate results)
- More complex error handling (partial batch failures)
- Less intuitive code flow
- Delayed results (can't stream page-by-page)

## Implementation Considerations

1. **Memory Budget**
   ```python
   # Estimate memory needs
   layout_memory = num_pages * page_size * layout_batch_size
   ocr_memory = avg_regions_per_page * region_size * ocr_batch_size
   # Ensure total < available_gpu_memory
   ```

2. **Error Recovery**
   ```python
   # Process in transactions
   try:
       layout_results = process_layout_batch(pages)
   except GPUMemoryError:
       # Reduce batch size and retry
       layout_results = process_layout_batch(pages, batch_size//2)
   ```

3. **Progress Tracking**
   ```python
   # Report progress per stage, not per page
   async def process_with_progress():
       yield "Layout detection", 0.2
       yield "Text extraction", 0.4  
       yield "Figure processing", 0.6
       yield "Table processing", 0.8
       yield "Database write", 1.0
   ```

## Pseudo-code for Full Pipeline

```python
async def process_pdf_batched(pdf_path, container_id):
    # Stage 0: Render all pages (CPU parallel)
    page_images = await render_pages_parallel(pdf_path)
    
    # Stage 1: Layout detection (GPU batch)
    layouts = await detect_layouts_batched(page_images)
    
    # Stage 2: Categorize and prepare regions
    text_regions, figure_regions, table_regions = categorize_regions(layouts)
    
    # Stage 3-5: Process each type (GPU batch)
    # Can potentially run in parallel with careful memory management
    text_results = await ocr_text_batched(text_regions)
    figure_results = await deplot_figures_batched(figure_regions)
    table_results = await extract_tables_batched(table_regions)
    
    # Stage 6: Batch database operations
    await batch_insert_results(
        container_id,
        text_results,
        figure_results, 
        table_results
    )
```

This transforms the pipeline from N * 4 GPU calls (N pages × 4 models) to just 4 batched GPU stages!

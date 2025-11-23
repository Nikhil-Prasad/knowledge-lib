# Knowledge‑Lib Architectural Memory

This file captures the key decisions, nomenclature, schema shape, and service layout agreed during the design/refactor of the ingestion and retrieval stack. Treat it as a living memory to orient future work.

## High‑Level Principles
- Many pipelines → one uniform retrieval layer (SQL‑first).
- Containers represent top‑level sources (PDF/DOCX/HTML/.txt). Segments are the retrievable units.
- Links + Anchors form a cross‑modal graph over segments for expansion and grounding.
- API DTOs are thin; services do fetch/identify/routing; persistence uses ORM; retrieval uses SQL.

## Core Nomenclature
- Containers: non‑retrievable structure/provenance (Document/Page concepts). Uniform key is `container_id` (UUID).
- Segments: retrievable units per modality. Unified shape across modalities with `segment_id` (UUID) + `modality`.
- Modalities: `text | table | citation | image | video | audio | container`. Note: `container` is a routing/structure hint, not a segment.
- Text segment subtype: `TextSegmentType = title | heading | paragraph | caption | footnote | sentence_window | blob`.
- Links: edges between segments with `relation` (supports, contradicts, refers_to, caption_of, mentions_entity, same_as, similar_to) and `scope` (document, collection, global).
- Anchors: precise locators attached to links (TextAnchor span, BBoxAnchor page bbox, TableAnchor cell, AVAnchor time window, CitationRef references a CitationSegment).

## Domain Model (src/app/domain)
- Split into focused modules:
  - `common.py`: Modality, TextSegmentType, BBox, SegmentBase (no timestamps, no asset_id).
  - `containers.py`: Document(container_id), Page, TableSchemaCol, TableSet.
  - `segments.py`: TextSegment, TableRow, Figure, AudioSegment, VideoSegment, CitationSegment, BibliographyEntry; Segment union; SegmentRef(container_id).
  - `anchors.py`: TextAnchor, BBoxAnchor, TableAnchor, AVAnchor, CitationRef; Anchor union.
  - `links.py`: Link, LinkCandidate, Relation, LinkScope.
  - `__init__.py`: re‑exports the public surface.
- Removed Asset and asset_id – provenance lives on containers (source_uri, mime_type, sha256).

## DB Schema (ORM: src/app/db/models/models.py)
- Renames/Uniformity:
  - `documents` → `containers` (PK: container_id).
  - `chunks` → `text_segments` (PK: segment_id).
  - All FKs renamed to `container_id` (pages + all segment tables).
  - `figures.caption_chunk_id` → `caption_segment_id` (FK → text_segments.segment_id).
  - `audio_segments.transcript_chunk_id` → `transcript_segment_id` (FK → text_segments.segment_id).
- Tables of interest:
  - containers(container_id, source_uri, mime_type, sha256, title, …)
  - pages(container_id, page_no, text, image_uri, width_px, height_px)
  - text_segments(segment_id, container_id, page_no, object_type enum, section_path, bbox, text, text_source enum, text_fts, emb_v1, emb_model, emb_version, chunk_version)
  - table_sets(table_id, container_id, name, n_rows, n_cols, schema JSON, page_no, bbox)
  - table_rows(row_id, table_id, row_index, row_json, row_text, row_text_fts, emb_v1, emb_model, emb_version)
  - figures(figure_id, container_id, page_no, bbox, caption_segment_id, image_uri, emb_v1, emb_siglip, …)
  - audio_segments, video_segments (with emb_v1)
  - bibliography_entries(bib_id, container_id,…)
  - citation_anchors(anchor_id, container_id, page_no, char_offset, marker, target_bib)
  - links(link_id, src_segment_id, src_modality, dst_segment_id, dst_modality, relation, scope,…)
  - link_anchors(link_id, atype, anchor JSON)
  - page_analysis(container_id, page_no, route, text_coverage, image_coverage, sandwich_score, quality_score?, timings JSON?, version)
- Indexes/FTS/Vector:
  - text_segments: GIN on `text_fts`; HNSW on `emb_v1` (cosine).
  - table_rows: GIN on `row_text_fts`; HNSW on `emb_v1`.
  - figures/audio/video: HNSW on `emb_v1`; figures also HNSW on `emb_siglip` (cosine).
  - B‑tree helpers on `(container_id, page_no)`, etc.; dedupe index on `containers.sha256`.
- Views:
  - `segments_text(segment_id, modality, container_id, page_no, text, emb_v1)`; `segments_all` currently equal to `segments_text`.
- Constraints added:
  - CHECK `links.src_modality/dst_modality ∈ {text, table, citation, image, audio, video}`.
  - CHECK `pages.page_no >= 1`.
  - UNIQUE `(table_id, row_index)` on `table_rows`.

## API Surface
- Ingest (unified): `POST /v1/ingest { source, options } → { container_id, pages_created, segments_created }`.
  - Sources: raw_text | remote_uri | data_uri | upload_ref.
  - Options (thinned): `{ dedupe: true, modality_hint?: Modality }`.
- Search: returns unified hits `{ modality, segment_id, container_id, page_no?, score, snippet? }`.

## Services Organization (src/app/services)
- Ingest package:
  - `ingest/__init__.py`: stable `ingest(session, req)` entrypoint.
  - `ingest/orchestrator.py`: orchestrates resolve → identify → choose pipeline → call pipeline.
    - Current scope: text + PDF; supports `RawTextSource` and `UploadRefSource` (local `file://` or absolute paths). Remote URIs intentionally not implemented.
  - `ingest/text_pipeline.py`: text pipeline implementation (`ingest_raw_text_pipeline`) + helpers.
  - `ingest/container_pipeline.py`: PDF container pipeline v1 (render → layout → vector/OCR fusion → segments; figure crops; citations parsing).
  - `ingest/types.py`: (not required right now; we route directly from DTOs).
- Embeddings:
  - Store raw vectors in per‑table `emb_v1` columns; cosine HNSW index.
  - Provider under `services/embeddings/oai_embeddings.py` (OpenAI, async client).
  - Background task pattern: API route schedules `embed_container_segments(container_id)` via FastAPI BackgroundTasks.
    - Runner: `services/embeddings/embed_runner.py` (sync function today) batches (256) segments per request,
      selects `text_segments` with `emb_v1 IS NULL` for the given `container_id`, calls `embed_many`, and
      updates `emb_v1/emb_model/emb_version` with per‑batch commits. Idempotent and safe to re‑run.
    - Reasoning: keep request latency low and DB transaction short; embeddings complete shortly after commit.
  - Vision: `figures.emb_siglip` column added for SigLIP embeddings (HNSW). Computation to be added in a future runner.
- Scripts:
  - `scripts/ingest_text.py` (CLI: raw text or file) and `scripts/search_text.py` (FTS sanity checks).
  - `scripts/process_pdf_local.py` (CLI: process a local PDF end‑to‑end without API; dedupe off; logs routing/regions/OCR).

## Ingestion (Text‑Only) – implemented
1) Normalize text (Unicode NFKC; CRLF→LF; strip control chars except tabs/newlines; collapse blank lines; trim). Abort on empty.
2) Dedupe: compute `sha256` over the normalized body; if a `containers.sha256` match exists and `options.dedupe=true`, short‑circuit with existing `container_id` and 0/0 counts.
3) Create `containers` row: `source_uri` (`"raw:text"` for raw; `file://…` for upload_ref), `mime_type` (`text/plain` or `text/markdown`), `sha256`, `title` (hint → heuristic → `untitled-<sha8>`).
4) Create one `pages` row (page_no=1, text=body).
5) Segment body:
   - Short: if `len(body) < threshold` (default 240 chars) → one `blob` segment.
   - Else: `sentence_window` windows with defaults from settings (k=4, overlap=1, soft_max≈1200 chars). Optional `title` segment if extracted.
   - Insert into `text_segments` with {container_id, page_no=1, object_type, section_path=None, bbox=None, text} and set `text_fts = to_tsvector('simple', text)`.
6) Embeddings: computed by a background task after commit (OpenAI `text-embedding-3-small`, 1536‑dim).
   - The route enqueues the task; the runner batches inputs (default 256) and updates `emb_v1`, `emb_model`, `emb_version`.
   - Retrieval should filter on `emb_v1 IS NOT NULL` for ANN until embeddings complete.
7) Collections: if `collection_id` provided, ensure `collections` row exists and link via `containers_collections` join before commit.
8) Return counts.

### Container granularity (text sources)
- One container per ingest source:
  - Raw text (`raw_text`) → one container with `source_uri="raw:text"`.
  - File upload reference (`upload_ref` with `file://…`) → one container per file path.
- Each text container currently has exactly one `Page` (page_no=1) holding the body; all text segments point to (container_id, page_no=1).
 - Dedupe is global by normalized body hash, not per collection or file path. Different sources with identical normalized text will dedupe to the same container unless `dedupe=false` is specified.

## Ingestion (PDF Containers) – v1 implemented
1) Create container + artifacts root; render each page once at configurable DPI (default 200) and set `pages.image_uri`.
2) Layout detection via DocLayNet DETR (HF) to produce zones (text/table/figure/caption/other).
3) Per‑page routing using PyMuPDF span coverage and figure area: `digital | ocr | hybrid`; persist to `page_analysis` with signals.
4) Text extraction:
   - Prefer PyMuPDF vector spans per zone (digital/hybrid) with `text_source='vector'`; else OCR via GOT‑OCR HF with `text_source='ocr'`.
   - Insert `text_segments` with bbox; store per‑page fused text into `pages.text`.
5) Figures: insert `figures` with bbox and save crops under artifacts; populate `image_uri`.
6) Tables: insert `table_sets` placeholders (structure extraction pending Table Transformer integration).
7) Citations: detect “References” heading; parse `[n]` bibliography entries; create in‑text anchors `[n]` on prior pages with `target_bib` mapping when possible.
8) Enqueue text embeddings if enabled by settings.

Notes
- CLI pipeline (`scripts/process_pdf_local.py`) runs with dedupe off, always creating a new container; API `/v1/ingest` retains dedupe by sha256.
- Logging added across stages (routing, regions, vector vs OCR, figure crops, citations summary, embeddings enqueue).

## Retrieval (SQL‑first)
- Text ANN: `ORDER BY emb_v1 <=> :q_emb LIMIT :k` (cosine; no normalization needed).
- Text FTS: `WHERE text_fts @@ plainto_tsquery('simple', :q)`.
- Scope by `container_id`; fuse (RRF) in app if combining ANN + FTS; traverse `links` for graph expansion.

### Current Retrieval State (as of text ingestion)
- Capabilities in place:
  - FTS (GIN) on `text_segments.text_fts` with `plainto_tsquery('simple', :q)` and `ts_rank_cd` for ranking.
  - ANN on `text_segments.emb_v1` (pgvector HNSW) with `ORDER BY emb_v1 <=> :q_emb` (filter `emb_v1 IS NOT NULL`).
  - Collection scoping via `containers_collections` join; container/page scoping via `(container_id, page_no)`.
- Not implemented yet:
  - Links/graph traversal (no links created in text pipeline—by design).
  - Hybrid search endpoint (RRF/weighted blend) and snippet highlighting.
  - Vision retrieval for figures via SigLIP (DB column present; embedding computation to be wired).

### Search v1→v2 – Implemented Work
- FTS config swap to english without re‑ingest:
  - Added a concurrent GIN expression index on `to_tsvector('english', text)`; switched queries to use english via that index.
  - Migrated to a stored english TSVECTOR column and updated the FTS trigger to compute `to_tsvector('pg_catalog.english', unaccent(text))`.
  - FTS endpoint now uses `websearch_to_tsquery('pg_catalog.english', unaccent(:q))` with `ts_rank_cd`; guarded fallback constructs a small OR‑of‑prefix query with a minimum score to avoid “no results”.
- Async endpoints:
  - `/v1/search/fts` and `/v1/search/ann` implemented with `AsyncSession` and collection scoping.
  - `/v1/search/hybrid` implemented: runs FTS + ANN, fuses via RRF, optional reranker, returns unified hits.
- ANN improvements:
  - Query embedding via OpenAI (1536‑d). Proper vector binding for pgvector.
  - Optional `hnsw.ef_search` per request/session (uses `SET LOCAL hnsw.ef_search = <int>`; safe fallback if unavailable).
  - Returned a longer `text` field (up to ~1500 chars) for better reranker inputs.
- Hybrid search tunables (settings):
  - `HYBRID_N_SEM` (default 400), `HYBRID_N_LEX` (default 200), `HYBRID_RERANK_POOL` (default 256).
  - `HYBRID_ANN_EF_SEARCH` (default 96), `HYBRID_PER_CONTAINER_LIMIT` (diversity; 0 disables).
  - Reranker flags: `RERANK_ENABLED`, `RERANK_MODEL`, `RERANK_DEVICE` (MPS/CPU/CUDA), `RERANK_BATCH_SIZE`, `RERANK_MAX_LENGTH`.
- Reranker (BGE) scaffold and enablement:
  - `services/rerank/bge_reranker.py` with lazy import of `sentence-transformers` CrossEncoder.
  - `build_rerank_text` helper composes input from title/section/snippet (optionally longer text).
  - `/v1/search/hybrid` passes `use_reranker=settings.rerank_enabled`; reranks fused top‑M on a background thread; graceful fallback if not installed.
  - Warm cache script run to pre‑download `BAAI/bge-reranker-base` on MPS.
- Eval tooling:
  - `scripts/quick_eval_scifact.py` runs FTS/ANN/HYBRID against BEIR SciFact, uses qrels/test.tsv, and looks up container IDs via title segments.
  - Computes Hit@k, MRR@k, Recall@k (doc‑level), now also aggregates a doc‑level confusion matrix (TP/FP/FN + precision/recall/F1).
  - Generates a timestamped Markdown report in `reports/` with summary metrics and sample queries.
- Tuning changes that improved performance:
  - Larger ANN pool and rerank pool; optional `ef_search` for better recall.
  - Switched to k=20 for higher recall; kept agent‑friendly diversity.
  - Disabled pre‑rerank container dedup (HYBRID_PER_CONTAINER_LIMIT=0) for SciFact so the reranker can select the best passage per document.

### Open Items / Future Improvements
- Weighted RRF (favor ANN w.r.t. FTS) to reduce fusion noise.
- Optional union‑then‑rerank strategy (e.g., ANN@100 ∪ FTS@50) for latency‑sensitive paths.
- Move dedup to post‑rerank to keep final results diverse while allowing the reranker to see all passages per doc.
- Add title joins directly in FTS/ANN SELECTs for stronger rerank inputs.

## Decisions & Renames (summary)
- `documents → containers`, `doc_id → container_id` everywhere.
- `chunks → text_segments`, `chunk_id → segment_id`.
- Domain: `TextChunk → TextSegment`; `TextObjectType → TextSegmentType`.
- Domain: `CitationAnchor → CitationSegment`; `TextAnchor.chunk_id → segment_id`.
- API Search: `chunk_id → segment_id`, add `modality`, `container_id`.
- Removed Asset/asset_id from domain to keep it lean.
- IngestOptions thinned to `{ dedupe, modality_hint }` – pipelines decide specifics.
- Router renamed to `orchestrator.py`; routes now delegate `services.ingest.ingest(req)` to orchestrator.
- Domain `TableSet.schema` renamed to `table_schema` to avoid Pydantic shadowing warnings.
- `Page ↔ TextSegment` relationships use explicit composite joins (view‑only) in ORM; no composite FK at DB layer yet.
- Added GIN index on `text_segments.text_fts` and HNSW on `emb_v1` (already); added B‑tree index `idx_containers_sha256` for dedupe.
- Added `collections` and `containers_collections` tables; pipeline links containers to provided `collection_id`.
- Settings centralize env config; added ingest tunables and embed config.

## Runtime Model (Async vs Sync)
- Current DB stack is synchronous (`psycopg2`). FastAPI runs sync endpoints/background tasks in a threadpool; the event loop stays free.
- Background embedding runner is sync by design (calls the async OpenAI client via `asyncio.run`), so DB I/O does not block the event loop.
- Future migration to asyncpg/AsyncSession planned (ingestion first, retrieval later) for higher concurrency:
  - Switch to `create_async_engine` (`postgresql+asyncpg://`) + `AsyncSession` and `await session.execute/flush/commit`.
  - Make orchestrator/pipeline/routes async; keep Alembic sync.
  - Move background runner DB to `AsyncSession` and `await embed_many` directly (no `asyncio.run`).

### Planned Async Migration (staged)
1) Add async session module (no behavior change): provide `get_async_db()` and `AsyncSession` factory while existing sync flow remains.
2) Implement an async embedding runner with bounded concurrency (TaskGroup/Semaphore) and batch commits; keep it unused initially.
3) Convert ingestion path to async:
   - API route `async def ingest(..., db: AsyncSession)` → `await orchestrator.ingest(db, req)`.
   - Orchestrator + text pipeline use `AsyncSession` and `select()/await session.execute(...)` patterns.
   - Schedule the async embedding runner as the background task.
4) Convert search endpoints to async later (keep SQL the same; just `await session.execute`).

### Bounded Concurrency Strategy (embeddings)
- Batch size: default 256 texts per `embed_many` call (tunable).
- Concurrency: default 2–4 concurrent batches per container (tunable), using `asyncio.TaskGroup` (Py 3.11) and `asyncio.Semaphore` to bound rate.
- Retry/backoff: exponential backoff + jitter on 429/5xx; commit per batch for incremental progress.
- Idempotent selection: always process `WHERE emb_v1 IS NULL` to allow safe re-runs.

## Open Items / Next Steps
- Container pipelines: implement HTML/PDF/DOCX ingestion (multi‑page, multi‑modal) and call segment‑specific ingesters.
- Remote URIs: either remove from API for now or implement a resolver that fetches text/html and routes to appropriate pipeline.
- Retrieval: add ANN endpoint using `emb_v1 <=> :q_emb`, optional fusion with FTS; add collection‑scoped queries.
- Embeddings: consider async worker for high‑volume ingestion.
- Collections: admin API to create/list; retrieval scoping by collection.
- Extend `segments_all` view to union other modalities as they come online.
 - Migrate DB to asyncpg/AsyncSession once text/image/PDF pipelines are stable; keep background embeddings and ingestion flow intact.

## References
- Docs: `docs/Raw_Text_Ingestion_Pipeline.md` (raw `.txt` pipeline details).
- README: schema summary, typical SQL (FTS/ANN), setup steps.
 - Update (Nov 2025):
   - PDF v1 implemented with DocLayNet layout + GOT‑OCR HF; per‑page routing and `page_analysis` table.
   - `text_segments.text_source` added for provenance; figure crops exported; citations parsed.
   - New CLI `scripts/process_pdf_local.py` for local PDF runs with logging.

## Concurrency & Performance Optimization

### Python GIL Mechanics
- **GIL (Global Interpreter Lock)**: A mutex that prevents multiple threads from executing Python bytecode simultaneously.
- **Relay Race Analogy**: Threads pass the GIL like a baton—only the holder can execute Python code.
- **GIL Release**: Happens every ~100 bytecodes, on I/O operations, and when calling into C extensions.
- **Key Insight**: The GIL prevents data races in pure Python but also prevents true parallelism for Python code.

### C-Extension Parallelism
- **PyTorch/NumPy/PyMuPDF**: Written in C/C++, these libraries release the GIL during heavy computations.
- **True Parallelism**: When GIL is released, multiple threads can execute C/C++ code simultaneously.
- **Pattern**: `Py_BEGIN_ALLOW_THREADS` → C++ computation → `Py_END_ALLOW_THREADS`
- **Benefit**: ThreadPoolExecutor with C extensions gives actual parallel speedup, not just concurrency.

### ThreadPoolExecutor Pattern
```python
# For C-bound work (rendering, models)
executor = ThreadPoolExecutor(max_workers=4)
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(executor, c_bound_function, args)
```
- **When to use**: PyMuPDF rendering, PyTorch inference, PIL image operations
- **Why it works**: These operations release GIL → threads run in parallel
- **Avoid for**: Pure Python compute (no GIL release → no speedup)

### AsyncIO Orchestration
- **Event Loop**: Single thread managing I/O operations and scheduling coroutines
- **Pattern**: Event loop for I/O orchestration + thread pool for CPU-bound work
- **Benefits**: Responsive system, efficient I/O handling, true parallelism for compute

### Comparison with C++/Rust
- **C++**: No GIL, manual memory management, fine-grained locking
  - Pro: True parallelism by default
  - Con: Manual synchronization, race conditions, memory safety issues
- **Rust**: No GIL, ownership system enforces thread safety at compile time
  - Pro: True parallelism + compile-time safety
  - Con: Steep learning curve, borrow checker complexity
- **Python**: GIL enforces sequential execution of Python code
  - Pro: No races in Python code, simple mental model
  - Con: No parallelism for pure Python, must use C extensions or processes

## PDF Pipeline Optimization Strategy

### Current Bottlenecks
1. **Sequential Page Rendering** (~200ms per page × 40 pages = 8s sequential)
   - PyMuPDF renders each page one by one in a loop
   
2. **Sequential Page Processing** (layout + OCR + extraction per page)
   - DETR layout detection
   - GOT-OCR inference
   - Text extraction and region processing
   - All happening serially

3. **Multiple DB Sessions**
   - Opening/closing connections repeatedly
   - Each page creates its own session for PageAnalysis
   - Separate sessions for segments, figures, tables

4. **No Batching**
   - Models process one item at a time
   - No leverage of GPU batch efficiency

### Optimization Approach

#### Stage 1: Parallel Rendering (ThreadPool)
```python
# Render all pages concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    page_images = list(executor.map(render_page, page_infos))
```
- Expected: 4× speedup (40 pages / 4 workers = 10 batches vs 40 sequential)

#### Stage 2: Concurrent Page Processing (AsyncIO + ThreadPool)
```python
async def process_page(page_info, image):
    # Layout detection in thread
    regions = await run_in_executor(layout_detector, page_info)
    # OCR in thread
    text = await run_in_executor(ocr_provider, regions)
    return results

# Process all pages concurrently
async with asyncio.TaskGroup() as tg:
    for page in pages:
        tg.create_task(process_page(page))
```
- Semaphore to limit concurrent pages (memory bound)
- ThreadPool for model inference (GIL release)

#### Stage 3: Batch DB Operations
```python
# Collect all results first
results = await process_all_pages_concurrent()

# Single transaction
async with AsyncSessionLocal() as session:
    # Batch insert all data
    session.add_all(page_analyses)
    session.add_all(text_segments)
    session.add_all(figures)
    await session.commit()  # One commit!
```

### Expected Improvements
- **40-page PDF current**: ~40 seconds (sequential everything)
- **40-page PDF optimized**: ~10-12 seconds
  - Rendering: 40 pages ÷ 4 workers = 10 seconds
  - Processing: Concurrent with rendering + overhead
  - DB: Single transaction < 1 second
- **Overall**: ~3-4× speedup expected

## Performance Optimization Todo List

### Immediate Tasks
- [ ] Update memory.md with concurrency discussion ✓
- [ ] Add profiling decorator to process_pdf_local.py
- [ ] Get baseline timing measurements
- [ ] Add per-provider ThreadPoolExecutor wrapper for compute-bound methods
- [ ] Make max_workers configurable in settings  

### Core Optimizations
- [ ] Add TaskGroups for concurrent page rendering
- [ ] Add TaskGroups for concurrent page processing
- [ ] Optimize DB session usage (batch operations)
- [ ] Add semaphores for concurrency control

### Fine Tuning
- [ ] Add optional batching for page-level inference
- [ ] Set sensible PyTorch thread/env defaults for MPS (avoid oversubscription)
- [ ] Profile throughput/latency trade-offs and document recommendations

### Configuration
```python
# New settings to add
PDF_RENDER_MAX_WORKERS = 4       # Rendering threads
PDF_PROCESS_MAX_WORKERS = 2      # Model inference threads  
PDF_MAX_CONCURRENT_PAGES = 8     # Semaphore for memory control
TORCH_NUM_THREADS = 1            # Prevent oversubscription
```

## Stage 2: Cross-Request Batch Aggregation (Future)

For production APIs serving concurrent requests, implement cross-request batching to optimize GPU utilization while keeping memory per request low:

### Architecture
```python
class GPUBatchAggregator:
    """Collect pages from multiple requests and batch them together for GPU ops"""
    def __init__(self, batch_size=8, batch_timeout=100):
        self.pending_items = []  # [(request_id, page_data)]
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout  # ms
        
    async def add_item(self, request_id, item):
        self.pending_items.append((request_id, item))
        
        # Trigger batch processing on size or timeout
        if len(self.pending_items) >= self.batch_size:
            await self._process_batch()
            
    async def _process_batch(self):
        # Mix items from different requests into one GPU batch
        batch_items = self.pending_items[:self.batch_size]
        batch_data = [item[1] for item in batch_items]
        
        # Single GPU call for items from multiple requests
        results = await model_inference(batch_data)
        
        # Route results back to correct requests
        for (req_id, _), result in zip(batch_items, results):
            await route_to_request(req_id, result)
```

### Benefits
- **Low memory per request**: Sequential processing within each request
- **High GPU utilization**: Batching across requests
- **Fair scheduling**: Round-robin or priority-based
- **Configurable latency**: Timeout ensures bounded wait time

### Deployment Considerations
- Use for high-traffic APIs (10+ concurrent requests)
- Monitor p50/p95/p99 latencies
- Adjust batch size based on traffic patterns
- Consider separate queues per model type

## Memory & Cost Analysis

### Single Request Memory Profile
Based on profiling a 25-page PDF:
- **Page images (200 DPI)**: ~15MB per page in memory (tensor form)
- **Model memory**: ~2.8GB constant (DETR + GOT-OCR + Table Transformer + DePlot)
- **Sequential processing**: 2.8GB + 15MB = ~2.85GB peak
- **Batched processing (25 pages)**: 2.8GB + 375MB = ~3.2GB peak
- **700-page textbook batched**: 2.8GB + 10.5GB = ~13.3GB minimum

### AWS Deployment Costs
For 10-20 concurrent users:
- **GPU instances**: p3.2xlarge ($3.06/hr) or g5.xlarge ($1.01/hr)
- **Memory needs**: 20 users × 3GB = 60GB minimum
- **Better approach**: Queue + worker pool pattern
  - API servers (CPU only): Handle requests, queue jobs
  - GPU workers (1-2 instances): Process queued jobs
  - Auto-scale workers based on queue depth

### Optimization for Cost
1. **Use spot instances** for GPU workers (70% cheaper)
2. **Sequential processing** to minimize memory
3. **Cross-request batching** to maximize GPU usage
4. **Aggressive model unloading** when idle
5. **Consider serverless GPU** (Runpod, Banana, Modal)

## Next Steps
- **Immediate**: Add memory/GPU profiling to understand exact usage
- **Stage 1**: Optimize single-request pipeline (current focus)
  - Concurrent page rendering ✓
  - Batch processing within request (future)
  - Memory-aware batch sizing
- **Stage 2**: Cross-request batch aggregation for APIs
- **Stage 3**: Distributed processing for large documents

### Other Next Steps
- Tables: integrate Microsoft Table Transformer
  - Detection: refine/confirm table zone bbox.
  - Structure recognition: extract grid (rows/cols, spans) and OCR per‑cell as needed.
  - Persistence: insert `table_rows(row_text, row_json, row_index)`; index `row_text_fts`.
- Charts → data (DePlot)
  - Run `google/deplot` on figure crops classified as charts; parse linearized table to `table_rows`.
  - Link provenance (figure → table) via `links` or a `source_figure_id` column on `table_sets`.
- Vision embeddings (SigLIP)
  - Implement an embedding runner for `figures.emb_siglip` (e.g., `google/siglip-base-patch16-224`).
  - Add an image/hybrid search endpoint and (optionally) fuse into hybrid RRF.
- Persist raw layout zones
  - Optional `layout_zones(container_id, page_no, zone_id, label, bbox, confidence)` or artifact JSON URI on `pages` for overlays/audits.
- Routing quality
  - Calibrate thresholds; add image coverage via PDF XObjects; consider rotation handling.
- Vector text reliability (PyMuPDF spans)
  - Investigate PDFs where vector text coverage is unexpectedly zero.
  - Try `page.get_text('words')`/`'blocks'` fallback; handle rotated pages; font encoding; invisible glyph layers.
  - Keep pdfminer.six as a secondary extractor if needed.
- OCR performance & quality
  - Batch crops; concurrency; tune MPS dtype and generation params; consider stop strings/templates.
- Citations v2
  - Add author‑year detection; better char_offset mapping with de‑hyphenation; exclude common footers from references.
- Dedupe/CLI ergonomics
  - Optional `--dedupe` flag for local CLI; job visibility/state table for orchestration.
- API surfaces
  - `get_page_context` (zones, segments, figures, tables); `get_figure_context` (image_uri, caption, derived rows).

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
  - text_segments(segment_id, container_id, page_no, object_type enum, section_path, bbox, text, text_fts, emb_v1, emb_model, emb_version, chunk_version)
  - table_sets(table_id, container_id, name, n_rows, n_cols, schema JSON, page_no, bbox)
  - table_rows(row_id, table_id, row_index, row_json, row_text, row_text_fts, emb_v1, emb_model, emb_version)
  - figures(figure_id, container_id, page_no, bbox, caption_segment_id, image_uri, emb_v1,…)
  - audio_segments, video_segments (with emb_v1)
  - bibliography_entries(bib_id, container_id,…)
  - citation_anchors(anchor_id, container_id, page_no, char_offset, marker, target_bib)
  - links(link_id, src_segment_id, src_modality, dst_segment_id, dst_modality, relation, scope,…)
  - link_anchors(link_id, atype, anchor JSON)
- Indexes/FTS/Vector:
  - text_segments: GIN on `text_fts`; HNSW on `emb_v1` (cosine).
  - table_rows: GIN on `row_text_fts`; HNSW on `emb_v1`.
  - figures/audio/video: HNSW on `emb_v1`.
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
    - Current scope: text‑only; supports `RawTextSource` and `UploadRefSource` (local `file://` or absolute paths). Remote URIs intentionally not implemented.
  - `ingest/text_pipeline.py`: text pipeline implementation (`ingest_raw_text_pipeline`) + helpers.
  - `ingest/container_pipeline.py`: container (PDF/DOCX/HTML) pipeline (skeleton; not implemented yet).
  - `ingest/types.py`: (not required right now; we route directly from DTOs).
- Embeddings:
  - Store raw vectors in per‑table `emb_v1` columns; cosine HNSW index.
  - Provider under `services/embeddings/oai_embeddings.py` (OpenAI, async client).
  - Background task pattern: API route schedules `embed_container_segments(container_id)` via FastAPI BackgroundTasks.
    - Runner: `services/embeddings/embed_runner.py` (sync function today) batches (256) segments per request,
      selects `text_segments` with `emb_v1 IS NULL` for the given `container_id`, calls `embed_many`, and
      updates `emb_v1/emb_model/emb_version` with per‑batch commits. Idempotent and safe to re‑run.
    - Reasoning: keep request latency low and DB transaction short; embeddings complete shortly after commit.
- Scripts:
  - `scripts/ingest_text.py` (CLI: raw text or file) and `scripts/search_text.py` (FTS sanity checks).

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

## Retrieval (SQL‑first)
- Text ANN: `ORDER BY emb_v1 <=> :q_emb LIMIT :k` (cosine; no normalization needed).
- Text FTS: `WHERE text_fts @@ plainto_tsquery('simple', :q)`.
- Scope by `container_id`; fuse (RRF) in app if combining ANN + FTS; traverse `links` for graph expansion.

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

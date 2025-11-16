
As such, we should first begin with the schemas and domain models that such a system would entail. I have chosen to break those down into 3 distinct layers. 

1) API DTOs (Data Transfer Objects). These are basically just the fastAPI schemas, because right now I am still thinking of this as a common corpus of all my agents to be able to query knowledge without worrying about underlying implementations. Both post Documents and Query Intelligence. 
2) Core Domain Models 
3) Corresponding DB models 


**API DTOs** 

Ingestion 

IngestSourceType = Literal["raw_text", "remote_uri", "data_uri", "upload_ref"]

1) Raw Text- inline bytes as a UTF-8 string in the request body. Small only text docs handled via API (rare, more for testing)
2) Remote_URI- https url 
3) Data_URI - https.pdf url or something like that 
4) upload_ref- S3, minio, etc (the preferred manner)

These give us our Source Types. Now we go to what Ingest Options look like. 

The ingest option is essentially the optionality that any front end system client can request; it provides knobs for the domain models. Right now we're only offering a modality type hint, because we expect the modality hint to contain all the logic necessary to determine which ingestion pipeline to use. I actually don't want the agent or client to customize much about the retrieval at the moment, the whole purpose of knowledge-lib being a separate API/service is that I want it to be a black box for external agents. 



Domain Models


At the top level, we have modalities. These are inferred by source types and also the modality_hints that are given in the ingestion DTOs. A modality is a determination of what pipeline to use for ingestion. 

For now I have defined 7 modalities. 

1) text: raw text content
2) table: structured tables from CSVs/Excels and their rows (excel would be at a container level though I suppose?)
3) citation: in-text citation markers (and their references)
4) image: figures/ROI (regions of interest) on pages.
5) video: obvious 
6) audio: obvious 
7) container

A container is obviously a combination of multiple modalities, like a PDF, or a book, or a word doc, or an html page or idk a power point or something like that. The idea is that you can call different pipeline services in conjunction for a container. If you happen to want to just embed an individual modality (an image, or an audio file, or something of the sort, then you can do that also). So then we continue down the line. 

Containers

- Document: top-level container (provenance, title, language, scanned).
- Page: 1-based page index, dimensions, optional raw page text/render.
- Asset (optional): source blob metadata (URI/MIME); can be backed by object storage.



Within each modality, you have "segments". Segments are just chunks in common nomenclature (like document chunks) but because we are considering multiple modalities, I call them segments. For each of the following modalities, we have the following segments. 



Segments By Modality

- Text
    - TextSegment: retrievable unit of text.
    - Fields: segment_id, modality="text", doc_id, page_no, object_type (TextSegmentType: title | heading | paragraph | caption | footnote | sentence_window | blob), section_path?, bbox?, text, emb_model/version?.
    - Use: FTS and vector search; segmentation by paragraph/sentence/window.
- Table
    - TableSet: table container (schema, dims, optional page/bbox); not a segment.
    - TableRow: retrievable unit; row_json (typed), row_text (for FTS), row_index, embeddings optional.
- Citation
    - CitationSegment: in-text marker (e.g., “[12]”), with doc_id/page_no/char_offset, marker, optional target_bib.
    - BibliographyEntry: parsed/reference metadata (title, authors, DOI/URL).
    - Future: optional ReferenceSegment (segment for references) to support agentic expansion.
- Image
    - Figure: ROI or whole figure; bbox normalized [0,1]; optional caption_segment_id; image_uri; embeddings optional (e.g., CLIP).
- Audio
    - AudioSegment: time window [t0_ms, t1_ms]; optional transcript_segment_id; embeddings optional.
- Video
    - VideoSegment: shot/frame/time window; keyframe_uri; embeddings optional.
- Container
    - Not a segment. Ingest pipeline detects/creates Document + Pages, then extracts and persists segments (text/table/image/citation/audio/video).

Links

- Link: typed relation between segments (src SegmentRef → dst SegmentRef).
- Relation: supports | contradicts | refers_to | caption_of | mentions_entity | same_as | similar_to.
- Scope: document | collection | global.
- LinkCandidate: offline proposal for batch/ML workflows.

Anchors

- Purpose: precise locators that ground a Link in evidence.
- TextAnchor: segment_id + [start,end) char span.
- BBoxAnchor: segment_id + bbox ([0,1] page-space).
- TableAnchor: table_id + row/col indices.
- AVAnchor: segment_id + [t0_ms, t1_ms].
- CitationRef: segment_id of a CitationSegment (the marker itself as the locator).

Identifiers & Types

- segment_id: UUID for every segment; used across APIs/links.
- modality: discriminant across all segments; carried in SegmentRef and API results.
- TextSegmentType: classification for text segments; only text needs this enum.
- Segment: discriminated union by modality across all segment types.
- SegmentRef: { segment_id, modality, doc_id? } for graph edges and lookups.

Routing & Options

- Source types: raw_text | remote_uri | data_uri | upload_ref (with content_type_hint).
    - dedupe: bool (default true)
    - modality_hint: optional Modality (routing hint; pipelines may ignore)
- Pipelines decide specifics (segmentation granularity, OCR, embeddings). As routing evolves, add hints—not client-side controls.

DB Mapping

- Document → documents
- Page → pages
- TextSegment → chunks (FTS trigger on text_fts; HNSW on emb_v1; object_type is Postgres enum)
- TableSet → table_sets
- TableRow → table_rows (FTS trigger on row_text_fts; HNSW on emb_v1)
- Figure → figures (HNSW on emb_v1)
- CitationSegment → citation_anchors
- BibliographyEntry → bibliography_entries
- AudioSegment → audio_segments (HNSW on emb_v1)
- VideoSegment → video_segments (HNSW on emb_v1)
- Link → links; anchors → link_anchors

Operational Notes

- BBoxes: normalized [0,1]; convert to pixels with page dims when needed.
- Embeddings: per-modality dims/models may differ; each table has its own HNSW index.
- FTS: triggers maintain tsvector columns for text and table rows.
- Dedupe: indexed documents.sha256; choose policy (reuse vs. re-ingest + link).
- Unified API: search returns { modality, segment_id, doc_id, page_no?, score, snippet? }. Ingest uses a single DTO with a discriminated source type.
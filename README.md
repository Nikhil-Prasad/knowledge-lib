# knowledge-lib

Knowledge-lib is an attempt at standardizing ingestion and retrieval via an API interface. My rationale for doing this is: 

1) I plan on building many multi-agent systems and having a centralized search and retrieval mechanism makes some sense to me. 
2) Most of the vectorDB + search systems are unneeded/redundant. 
3) Retrieval is fundamentally a data engineering problem. Solving that problem 100 times for 100 different apps/agent set ups seems unneeded. 
4) I prefer unification of modalities

We need to think about this from the perspective of, what exactly should a system specialized on "knowledge" entail? We know that LLMs are primarily text machines, so if you are building a system where an LLM must generate based on some passed context, you ultimately have to build a system/pipeline that converts multiple modalities into textual representations (columns, embeddings, key words, metadata, summaries, graphs, etc.) that you can retrieve across various RAG apps as you build them. For this purpose; you need a unified ingestion service. This service can (and will) utilize multiple models, techniques in order to transform N modalities (text, audio, video, PDFs, etc) into a textual representation. 

As such we should first begin by properly representing the "objects" and data flow transfer that such an ingestion system would entail. Details on those are in /docs/schemas.md

Secondly, we must think about retrieval. Assuming that all forms of data are ingested into a set of representations that can then be retrieved via SQL queries on the corresponding table that each schema holds. This allows for a lot of cross cutting, as the SQL layer is also matched with a minio/s3 layer, and since the models are multimodal, you can augment generation with actually passing in images (if needed) and parts of PDFs. But the primary layer is text.

Standardizing this into one API service, while it may add a bit of latency, unifies an approach for all long running agents. 

## Overview

- Storage: Postgres 16 + pgvector (via Docker Compose)
- ORM/Migrations: SQLAlchemy + Alembic
- Env management: single root `.env` used by app/Alembic and Docker Compose (Makefile passes `--env-file .env`)
- Features:
  - Vector embeddings column `emb_v1` with HNSW index (`vector_cosine_ops`)
  - Full‑text search (FTS) with `text_fts` tsvector + GIN index
  - Trigger keeps `text_fts` in sync with `text` on INSERT/UPDATE
  - Cascading relationships between `documents`, `chunks`, and `pages`

## Quick Start (local)

1) Start Postgres with Docker:
- make up

2) Create your root `.env` (app + Compose):
- cp .env.example .env

3) Install deps (via uv):
- uv sync

4) Initialize DB schema (Alembic):
- uv run alembic upgrade head

5) Sanity checks:
- uv run alembic current -v
- PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "SELECT 1"

Makefile convenience:
- make up    # docker compose up -d (Postgres + pgAdmin)
- make down  # docker compose down
- make api   # run FastAPI locally (uvicorn)
- make prepare-beir-scifact  # export BEIR SciFact to evals/datasets as per-doc .txt + manifest

## Prepare BEIR SciFact (raw text)

Export SciFact into per-document `.txt` files (no external deps):
- make prepare-beir-scifact

Outputs (under `evals/datasets`):
- `evals/datasets/beir/processed/scifact/docs/*.txt` — UTF-8 text files, one per BEIR doc (title + blank line + body)
- `evals/datasets/beir/processed/scifact/manifest.jsonl` — metadata per doc (source_uri, sha256, etc.)
- `evals/datasets/beir/eval/scifact/queries.jsonl` — BEIR queries for the chosen split (if present)
- `evals/datasets/beir/eval/scifact/qrels/test.tsv` — relevance labels (or split fallback)


## API Endpoints

- Run server:
  - `uv run uvicorn src.app.main:app --reload`

- Health (unversioned):
  - `GET /health/live` — liveness (process up; no DB).
  - `GET /health/ready` — readiness (checks DB with `SELECT 1`).
  - Alias: `GET /health/healthz` (compat; same as readiness).

- v1 (business endpoints):
  - `POST /v1/ingest`
    - Request: `{ "source": { "source_type": "raw_text", "title": "Demo", "text": "hello world ..." } }`
    - Response: `{ "container_id": "<uuid>", "pages_created": 1, "segments_created": 1 }`
  - `POST /v1/search`
    - Request: `{ "query": "hello", "k": 20 }`
    - Response: `{ "results": [ { "modality": "text", "segment_id": "...", "container_id": "...", "page_no": 1, "score": 0.42, "snippet": "..." } ] }`

## Environment Files

- Root `.env` (single source of truth for app/Alembic/Compose):
  - DATABASE_URL=postgresql+psycopg2://kl:klpass@localhost:5432/knowledge
  - PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE (optional convenience variables)
  - POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_PORT (used by Docker Compose)
  - Example committed: `.env.example`
  - `.env` is ignored (do not commit secrets)

Alembic and the app load the root `.env` (via python‑dotenv). Docker Compose also reads it via the Makefile (`--env-file .env`).

## Schema Summary

Tables:
- containers (PK: container_id, metadata, timestamps)
- chunks (PK: chunk_id; FK to containers; `text` + `text_fts` TSVECTOR with GIN index; `emb_v1` VECTOR(1536) with HNSW index)
- pages (PK: container_id, page_no; FK to containers)

Indexes:
- GIN on `chunks.text_fts`
- HNSW on `chunks.emb_v1` (`vector_cosine_ops`)
- BTree on `(container_id, page_no)` and `container_id`

FTS maintenance:
- Trigger `trg_chunks_fts` computes:
  - `NEW.text_fts := to_tsvector('simple'::regconfig, unaccent(coalesce(NEW.text,'')))`

Extensions (auto‑created by migration):
- vector, unaccent, pg_trgm

## Typical Queries

Lexical (FTS):
```
SELECT segment_id, ts_rank_cd(text_fts, to_tsquery('simple', 'kyc:*')) AS score
FROM text_segments
WHERE text_fts @@ to_tsquery('simple', 'kyc:*')
ORDER BY score DESC
LIMIT 50;
```

Vector (ANN):
```
SELECT segment_id, 1 - (emb_v1 <=> :q_emb) AS cosine_sim
FROM text_segments
ORDER BY emb_v1 <=> :q_emb
LIMIT 50;
```

Hybrid: fetch top‑N from each, fuse (e.g., RRF), then rerank.

## Migrations Workflow

- Create a revision (autogenerate):
  - uv run alembic revision --autogenerate -m "your message"
- Apply latest:
  - uv run alembic upgrade head
- Show current:
  - uv run alembic current -v

Rule of thumb: do not edit an already‑applied migration; add a new revision for changes.

## Commit Policy

- Commit definitions and examples:
  - infra/docker/docker-compose.yml
  - .env.example (root)
  - Alembic migrations, SQL/trigger logic
  - Application code
- Do NOT commit:
  - data/ (Docker volumes, Postgres data) — already ignored
  - Any `.env` with secrets — the example is committed instead

## Troubleshooting

- Cannot connect to `localhost:5432`:
  - Ensure containers are up: `make up`
  - Check logs: `docker logs kl-postgres`
  - Health: `docker inspect -f '{{.State.Health.Status}}' kl-postgres`
- Migrations complain about FTS immutability:
  - This repo uses a trigger‑maintained `text_fts`, not a generated column, to comply with Postgres immutability rules.

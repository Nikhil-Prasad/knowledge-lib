# knowledge-lib

Self‑contained API for ingestion and hybrid retrieval using Postgres (pgvector) and MinIO. Local Postgres is run via Docker Compose to keep setup simple and reproducible.

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
- documents (PK: doc_id, metadata, timestamps)
- chunks (PK: chunk_id; FK to documents; `text` + `text_fts` TSVECTOR with GIN index; `emb_v1` VECTOR(1536) with HNSW index)
- pages (PK: doc_id, page_no; FK to documents)

Indexes:
- GIN on `chunks.text_fts`
- HNSW on `chunks.emb_v1` (`vector_cosine_ops`)
- BTree on `(doc_id, page_no)` and `doc_id`

FTS maintenance:
- Trigger `trg_chunks_fts` computes:
  - `NEW.text_fts := to_tsvector('simple'::regconfig, unaccent(coalesce(NEW.text,'')))`

Extensions (auto‑created by migration):
- vector, unaccent, pg_trgm

## Typical Queries

Lexical (FTS):
```
SELECT chunk_id, ts_rank_cd(text_fts, to_tsquery('simple', 'kyc:*')) AS score
FROM chunks
WHERE text_fts @@ to_tsquery('simple', 'kyc:*')
ORDER BY score DESC
LIMIT 50;
```

Vector (ANN):
```
SELECT chunk_id, 1 - (emb_v1 <=> :q_emb) AS cosine_sim
FROM chunks
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

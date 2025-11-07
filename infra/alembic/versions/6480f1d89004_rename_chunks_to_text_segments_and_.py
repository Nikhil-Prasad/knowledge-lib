"""rename chunks to text_segments and align FKs/indexes/triggers/views

Revision ID: 6480f1d89004
Revises: 1cdb00b0ff5a
Create Date: 2025-11-05 22:10:27.733989

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6480f1d89004'
down_revision: Union[str, Sequence[str], None] = '1cdb00b0ff5a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Rename chunks table to text_segments
    op.rename_table('chunks', 'text_segments')

    # Rename primary key column chunk_id -> segment_id
    op.alter_column('text_segments', 'chunk_id', new_column_name='segment_id')

    # Rename indexes for clarity (best-effort; ignore if names differ)
    op.execute("DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_chunks_doc_page') THEN ALTER INDEX idx_chunks_doc_page RENAME TO idx_text_segments_doc_page; END IF; END $$;")
    op.execute("DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_chunks_fts') THEN ALTER INDEX idx_chunks_fts RENAME TO idx_text_segments_fts; END IF; END $$;")
    op.execute("DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_chunks_emb_v1') THEN ALTER INDEX idx_chunks_emb_v1 RENAME TO idx_text_segments_emb_v1; END IF; END $$;")
    op.execute("DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_class WHERE relname='ix_chunks_doc_id') THEN ALTER INDEX ix_chunks_doc_id RENAME TO ix_text_segments_doc_id; END IF; END $$;")

    # Rename trigger/function for FTS maintenance
    op.execute("DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='trg_chunks_fts') THEN ALTER TRIGGER trg_chunks_fts ON text_segments RENAME TO trg_text_segments_fts; END IF; END $$;")
    op.execute("DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_proc WHERE proname='chunks_fts_trigger') THEN ALTER FUNCTION chunks_fts_trigger() RENAME TO text_segments_fts_trigger; END IF; END $$;")

    # Update FKs in dependent tables: figures, audio_segments
    # figures.caption_chunk_id -> caption_segment_id and point to text_segments.segment_id
    with op.batch_alter_table('figures') as batch_op:
        # Drop old FK if exists
        batch_op.drop_constraint('figures_caption_chunk_id_fkey', type_='foreignkey')
        batch_op.alter_column('caption_chunk_id', new_column_name='caption_segment_id')
        batch_op.create_foreign_key(None, 'text_segments', ['caption_segment_id'], ['segment_id'])

    # audio_segments.transcript_chunk_id -> transcript_segment_id
    with op.batch_alter_table('audio_segments') as batch_op:
        batch_op.drop_constraint('audio_segments_transcript_chunk_id_fkey', type_='foreignkey')
        batch_op.alter_column('transcript_chunk_id', new_column_name='transcript_segment_id')
        batch_op.create_foreign_key(None, 'text_segments', ['transcript_segment_id'], ['segment_id'])

    # Update views to reference text_segments
    op.execute("DROP VIEW IF EXISTS segments_all")
    op.execute("DROP VIEW IF EXISTS segments_text")
    op.execute(
        """
        CREATE VIEW segments_text AS
        SELECT
            segment_id,
            'text'::text AS modality,
            doc_id,
            page_no,
            text,
            emb_v1
        FROM text_segments;
        """
    )
    op.execute("CREATE VIEW segments_all AS SELECT * FROM segments_text;")


def downgrade() -> None:
    """Downgrade schema."""
    # Revert view changes
    op.execute("DROP VIEW IF EXISTS segments_all")
    op.execute("DROP VIEW IF EXISTS segments_text")

    # Revert FK/column renames in audio_segments
    with op.batch_alter_table('audio_segments') as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.alter_column('transcript_segment_id', new_column_name='transcript_chunk_id')
        batch_op.create_foreign_key(None, 'chunks', ['transcript_chunk_id'], ['chunk_id'])

    # Revert FK/column renames in figures
    with op.batch_alter_table('figures') as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.alter_column('caption_segment_id', new_column_name='caption_chunk_id')
        batch_op.create_foreign_key(None, 'chunks', ['caption_chunk_id'], ['chunk_id'])

    # Revert trigger/function rename
    op.execute("DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_proc WHERE proname='text_segments_fts_trigger') THEN ALTER FUNCTION text_segments_fts_trigger() RENAME TO chunks_fts_trigger; END IF; END $$;")
    op.execute("DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='trg_text_segments_fts') THEN ALTER TRIGGER trg_text_segments_fts ON text_segments RENAME TO trg_chunks_fts; END IF; END $$;")

    # Revert index renames (best-effort)
    op.execute("DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_text_segments_doc_page') THEN ALTER INDEX idx_text_segments_doc_page RENAME TO idx_chunks_doc_page; END IF; END $$;")
    op.execute("DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_text_segments_fts') THEN ALTER INDEX idx_text_segments_fts RENAME TO idx_chunks_fts; END IF; END $$;")
    op.execute("DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_text_segments_emb_v1') THEN ALTER INDEX idx_text_segments_emb_v1 RENAME TO idx_chunks_emb_v1; END IF; END $$;")
    op.execute("DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_class WHERE relname='ix_text_segments_doc_id') THEN ALTER INDEX ix_text_segments_doc_id RENAME TO ix_chunks_doc_id; END IF; END $$;")

    # Revert PK column rename
    op.alter_column('text_segments', 'segment_id', new_column_name='chunk_id')
    # Rename table back
    op.rename_table('text_segments', 'chunks')

"""unify segments minimal: enum, sha idx, updated_at, views

Revision ID: c7db4e8a3c7d
Revises: b0127cefc16c
Create Date: 2025-11-02 23:09:57.083043

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7db4e8a3c7d'
down_revision: Union[str, Sequence[str], None] = 'b0127cefc16c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1) Enum for chunks.object_type (text object types)
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'text_object_type') THEN
                CREATE TYPE text_object_type AS ENUM (
                    'title','heading','paragraph','caption','footnote','sentence_window','blob'
                );
            END IF;
        END$$;
        """
    )
    op.execute(
        """
        ALTER TABLE chunks
        ALTER COLUMN object_type TYPE text_object_type
        USING object_type::text_object_type
        """
    )

    # 2) Dedupe index on documents.sha256
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256)")

    # 3) Keep updated_at accurate via trigger on documents
    op.execute(
        """
        CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          NEW.updated_at := now();
          RETURN NEW;
        END$$;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_documents_updated ON documents")
    op.execute(
        """
        CREATE TRIGGER trg_documents_updated
        BEFORE UPDATE ON documents
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """
    )

    # 4) Convenience views for segments (start with text only)
    op.execute(
        """
        CREATE OR REPLACE VIEW segments_text AS
        SELECT
            chunk_id AS segment_id,
            'text'::text AS modality,
            doc_id,
            page_no,
            text,
            emb_v1
        FROM chunks;
        """
    )
    op.execute(
        """
        CREATE OR REPLACE VIEW segments_all AS
        SELECT * FROM segments_text;
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop views (all depends on text)
    op.execute("DROP VIEW IF EXISTS segments_all")
    op.execute("DROP VIEW IF EXISTS segments_text")

    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS trg_documents_updated ON documents")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at()")

    # Drop index
    op.execute("DROP INDEX IF EXISTS idx_documents_sha256")

    # Revert enum change
    op.execute(
        """
        ALTER TABLE chunks
        ALTER COLUMN object_type TYPE TEXT
        USING object_type::text
        """
    )
    op.execute("DROP TYPE IF EXISTS text_object_type")

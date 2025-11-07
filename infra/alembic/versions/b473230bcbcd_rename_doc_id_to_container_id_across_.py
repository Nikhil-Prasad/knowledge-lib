"""rename doc_id to container_id across container/pages/segments

Revision ID: b473230bcbcd
Revises: 6480f1d89004
Create Date: 2025-11-05 22:23:32.033813

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b473230bcbcd'
down_revision: Union[str, Sequence[str], None] = '6480f1d89004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1) containers.doc_id -> containers.container_id
    op.alter_column('containers', 'doc_id', new_column_name='container_id')

    # 2) pages.doc_id -> pages.container_id (part of PK)
    op.alter_column('pages', 'doc_id', new_column_name='container_id')

    # 3) text_segments.doc_id -> text_segments.container_id
    if op.get_bind().dialect.has_table(op.get_bind(), 'text_segments'):
        op.alter_column('text_segments', 'doc_id', new_column_name='container_id')

    # 4) figures.doc_id -> figures.container_id
    op.alter_column('figures', 'doc_id', new_column_name='container_id')

    # 5) table_sets.doc_id -> table_sets.container_id
    op.alter_column('table_sets', 'doc_id', new_column_name='container_id')

    # 6) bibliography_entries.doc_id -> bibliography_entries.container_id
    op.alter_column('bibliography_entries', 'doc_id', new_column_name='container_id')

    # 7) citation_anchors.doc_id -> citation_anchors.container_id
    op.alter_column('citation_anchors', 'doc_id', new_column_name='container_id')

    # 8) audio_segments.doc_id -> audio_segments.container_id
    op.alter_column('audio_segments', 'doc_id', new_column_name='container_id')

    # 9) video_segments.doc_id -> video_segments.container_id
    op.alter_column('video_segments', 'doc_id', new_column_name='container_id')

    # 10) Recreate views with container_id
    op.execute("DROP VIEW IF EXISTS segments_all")
    op.execute("DROP VIEW IF EXISTS segments_text")
    op.execute(
        """
        CREATE VIEW segments_text AS
        SELECT
            segment_id,
            'text'::text AS modality,
            container_id,
            page_no,
            text,
            emb_v1
        FROM text_segments;
        """
    )
    op.execute("CREATE VIEW segments_all AS SELECT * FROM segments_text;")


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column('video_segments', 'container_id', new_column_name='doc_id')
    op.alter_column('audio_segments', 'container_id', new_column_name='doc_id')
    op.alter_column('citation_anchors', 'container_id', new_column_name='doc_id')
    op.alter_column('bibliography_entries', 'container_id', new_column_name='doc_id')
    op.alter_column('table_sets', 'container_id', new_column_name='doc_id')
    op.alter_column('figures', 'container_id', new_column_name='doc_id')
    if op.get_bind().dialect.has_table(op.get_bind(), 'text_segments'):
        op.alter_column('text_segments', 'container_id', new_column_name='doc_id')
    op.alter_column('pages', 'container_id', new_column_name='doc_id')
    op.alter_column('containers', 'container_id', new_column_name='doc_id')

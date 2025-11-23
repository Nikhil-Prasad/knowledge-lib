"""add page_analysis table and text_source on text_segments

Revision ID: f1a2b3c4d5e6
Revises: 3f7a2b1c8c22
Create Date: 2025-11-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, Sequence[str], None] = '3f7a2b1c8c22'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) Create enum type for text_source
    op.execute("DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'text_source_type') THEN CREATE TYPE text_source_type AS ENUM ('vector','ocr','fused'); END IF; END $$;")

    # 2) Add column to text_segments
    op.add_column('text_segments', sa.Column('text_source', sa.Enum('vector', 'ocr', 'fused', name='text_source_type', native_enum=True), nullable=True))

    # 3) Create page_analysis table
    op.create_table(
        'page_analysis',
        sa.Column('container_id', sa.UUID(), sa.ForeignKey('containers.container_id', ondelete='CASCADE'), primary_key=True, nullable=False),
        sa.Column('page_no', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('route', sa.String(), nullable=False),
        sa.Column('text_coverage', sa.Float(), nullable=True),
        sa.Column('image_coverage', sa.Float(), nullable=True),
        sa.Column('sandwich_score', sa.Float(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('timings', sa.JSON(), nullable=True),
        sa.Column('version', sa.Text(), nullable=True),
    )
    op.create_index('idx_page_analysis_container_page', 'page_analysis', ['container_id', 'page_no'])

    # 4) Add emb_siglip (768-d) to figures and HNSW index
    op.add_column('figures', sa.Column('emb_siglip', pgvector.sqlalchemy.vector.VECTOR(dim=768), nullable=True))
    op.create_index(
        'idx_figures_emb_siglip',
        'figures',
        ['emb_siglip'],
        unique=False,
        postgresql_using='hnsw',
        postgresql_ops={'emb_siglip': 'vector_cosine_ops'}
    )


def downgrade() -> None:
    # Drop page_analysis
    op.drop_index('idx_page_analysis_container_page', table_name='page_analysis')
    op.drop_table('page_analysis')

    # Drop SigLIP index and column
    op.drop_index('idx_figures_emb_siglip', table_name='figures')
    op.drop_column('figures', 'emb_siglip')

    # Drop column and type
    op.drop_column('text_segments', 'text_source')
    op.execute("DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'text_source_type') THEN DROP TYPE text_source_type; END IF; END $$;")

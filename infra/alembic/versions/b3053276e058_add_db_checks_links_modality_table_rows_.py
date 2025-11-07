"""add DB checks: links modality, table_rows unique, pages page_no check

Revision ID: b3053276e058
Revises: b473230bcbcd
Create Date: 2025-11-05 22:41:08.485706

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3053276e058'
down_revision: Union[str, Sequence[str], None] = 'b473230bcbcd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1) Links modality CHECK
    op.execute(
        """
        ALTER TABLE links
        ADD CONSTRAINT chk_links_modality
        CHECK (
          src_modality IN ('text','table','citation','image','audio','video') AND
          dst_modality IN ('text','table','citation','image','audio','video')
        )
        """
    )

    # 2) Pages page_no >= 1 CHECK
    op.execute(
        """
        ALTER TABLE pages
        ADD CONSTRAINT chk_pages_page_no
        CHECK (page_no >= 1)
        """
    )

    # 3) table_rows unique (table_id, row_index)
    # Drop the existing non-unique index to avoid redundancy
    op.drop_index('idx_table_rows_table_idx', table_name='table_rows')
    op.create_unique_constraint('uq_table_rows_table_row_index', 'table_rows', ['table_id', 'row_index'])


def downgrade() -> None:
    """Downgrade schema."""
    # Revert unique on table_rows; recreate non-unique index
    op.drop_constraint('uq_table_rows_table_row_index', table_name='table_rows', type_='unique')
    op.create_index('idx_table_rows_table_idx', 'table_rows', ['table_id', 'row_index'], unique=False)

    # Drop pages CHECK
    op.execute("ALTER TABLE pages DROP CONSTRAINT IF EXISTS chk_pages_page_no")

    # Drop links modality CHECK
    op.execute("ALTER TABLE links DROP CONSTRAINT IF EXISTS chk_links_modality")

"""rename documents to containers and update fks

Revision ID: 1cdb00b0ff5a
Revises: c7db4e8a3c7d
Create Date: 2025-11-05 21:56:46.640043

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1cdb00b0ff5a'
down_revision: Union[str, Sequence[str], None] = 'c7db4e8a3c7d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1) Rename table documents -> containers
    op.rename_table('documents', 'containers')

    # 2) Rename trigger on documents to containers for clarity (if exists)
    # Note: trigger stays attached on table rename; we just rename it cosmetically.
    op.execute("ALTER TRIGGER trg_documents_updated ON containers RENAME TO trg_containers_updated")

    # FKs referencing documents are automatically updated by table rename.
    # No further action needed as column names remain the same.


def downgrade() -> None:
    """Downgrade schema."""
    # Rename trigger back
    op.execute("ALTER TRIGGER trg_containers_updated ON containers RENAME TO trg_documents_updated")
    # Rename table back containers -> documents
    op.rename_table('containers', 'documents')

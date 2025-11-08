"""add idx_containers_sha256 on containers.sha256

Revision ID: d6a8a1c4a1a9
Revises: b473230bcbcd
Create Date: 2025-11-07 14:30:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "d6a8a1c4a1a9"
down_revision: Union[str, Sequence[str], None] = "b473230bcbcd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create dedupe index on containers.sha256 (bytea)
    op.execute("CREATE INDEX IF NOT EXISTS idx_containers_sha256 ON containers(sha256)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_containers_sha256")


"""add collections and containers_collections join

Revision ID: 6e2f5c9b2a10
Revises: 1bd139b8b741
Create Date: 2025-11-07 21:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "6e2f5c9b2a10"
down_revision: Union[str, Sequence[str], None] = "1bd139b8b741"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "collections",
        sa.Column("collection_id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.Text()),
        sa.Column("description", sa.Text()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    op.create_table(
        "containers_collections",
        sa.Column("collection_id", sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.collection_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("container_id", sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey("containers.container_id", ondelete="CASCADE"), primary_key=True),
    )
    op.create_index("idx_containers_collections_collection", "containers_collections", ["collection_id"])
    op.create_index("idx_containers_collections_container", "containers_collections", ["container_id"])


def downgrade() -> None:
    op.drop_index("idx_containers_collections_container", table_name="containers_collections")
    op.drop_index("idx_containers_collections_collection", table_name="containers_collections")
    op.drop_table("containers_collections")
    op.drop_table("collections")


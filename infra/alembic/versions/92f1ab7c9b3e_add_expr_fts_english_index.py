from alembic import op

# revision identifiers, used by Alembic.
revision = "92f1ab7c9b3e"
down_revision = "6e2f5c9b2a10"
branch_labels = None
depends_on = None


def upgrade():
    # Run outside a transaction so CREATE INDEX CONCURRENTLY is allowed
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_text_segments_fts_en_expr "
            "ON text_segments USING GIN (to_tsvector('english', text))"
        )


def downgrade():
    # Run outside a transaction so DROP INDEX CONCURRENTLY is allowed
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_text_segments_fts_en_expr")

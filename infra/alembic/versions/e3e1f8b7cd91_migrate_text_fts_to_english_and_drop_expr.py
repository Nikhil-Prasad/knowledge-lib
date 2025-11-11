from alembic import op

# revision identifiers, used by Alembic.
revision = "e3e1f8b7cd91"
down_revision = "92f1ab7c9b3e"
branch_labels = None
depends_on = None


def upgrade():
    # 1) Recompute stored tsvector column to english
    op.execute("UPDATE text_segments SET text_fts = to_tsvector('english', text)")

    # 2) Drop the temporary english expression index (no longer needed)
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX IF EXISTS idx_text_segments_fts_en_expr")


def downgrade():
    # 1) We cannot recover prior 'simple' tsvector content deterministically; keep english values.
    # 2) Recreate the english expression index for compatibility if needed.
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_text_segments_fts_en_expr "
            "ON text_segments USING GIN (to_tsvector('english', text))"
        )


from alembic import op

# revision identifiers, used by Alembic.
revision = "3f7a2b1c8c22"
down_revision = "e3e1f8b7cd91"
branch_labels = None
depends_on = None


def upgrade():
    # Ensure unaccent extension exists
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")

    # Replace trigger function to compute english tsvector with unaccent
    op.execute(
        """
        CREATE OR REPLACE FUNCTION text_segments_fts_trigger() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          NEW.text_fts := to_tsvector('pg_catalog.english'::regconfig, unaccent(coalesce(NEW.text,'')));
          RETURN NEW;
        END$$;
        """
    )

    # Backfill existing rows to ensure stored column matches new logic
    op.execute(
        "UPDATE text_segments SET text_fts = to_tsvector('pg_catalog.english', unaccent(coalesce(text,'')))"
    )


def downgrade():
    # Revert trigger function to simple+unaccent
    op.execute(
        """
        CREATE OR REPLACE FUNCTION text_segments_fts_trigger() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          NEW.text_fts := to_tsvector('simple'::regconfig, unaccent(coalesce(NEW.text,'')));
          RETURN NEW;
        END$$;
        """
    )


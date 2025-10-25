# infra/alembic/env.py
from __future__ import annotations
import os, sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

# --- Make 'src/' importable for model imports ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import your declarative Base (mapped models)
from src.app.db.models.models import Base  

# Alembic Config
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Prefer DATABASE_URL env; fallback to alembic.ini sqlalchemy.url
DB_URL = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))

target_metadata = Base.metadata

def include_object(obj, name, type_, reflected, compare_to):
    # migrate everything; keep hook for future filters if needed
    return True

def run_migrations_offline() -> None:
    context.configure(
        url=DB_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,              # detect Vector(1536) changes, etc.
        compare_server_default=True,    # detect Computed/DEFAULT changes
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    connectable = create_engine(DB_URL, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

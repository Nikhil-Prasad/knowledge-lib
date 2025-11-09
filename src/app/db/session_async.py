from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.app.settings import get_settings

settings = get_settings()

# Async engine (not yet wired into the app routes; provided for staged migration)
ASYNC_DATABASE_URL = settings.sqlalchemy_url.replace(
    "postgresql+psycopg2://", "postgresql+asyncpg://"
)

async_engine = create_async_engine(ASYNC_DATABASE_URL, pool_pre_ping=True, future=True)

AsyncSessionLocal = sessionmaker(
    bind=async_engine, class_=AsyncSession, autoflush=False, autocommit=False, expire_on_commit=False
)


async def get_async_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

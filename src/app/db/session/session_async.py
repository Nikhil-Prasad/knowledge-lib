from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator

from src.app.settings import get_settings

settings = get_settings()

# Async engine for app runtime (uses ASYNC_DATABASE_URL if set, otherwise derives from sync URL)
async_engine = create_async_engine(settings.async_sqlalchemy_url, pool_pre_ping=True, future=True)

AsyncSessionLocal = sessionmaker(
    bind=async_engine, class_=AsyncSession, autoflush=False, autocommit=False, expire_on_commit=False
)


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

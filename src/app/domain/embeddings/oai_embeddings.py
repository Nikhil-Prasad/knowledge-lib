from openai import AsyncOpenAI
from functools import lru_cache
import asyncio
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from settings import get_settings

from typing import AsyncIterator, List, Optional, Sequence, Tuple, Dict

settings = get_settings()


@lru_cache(maxsize=1)
def _openai_compat(base_url: Optional[str] = None) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=base_url,
        api_key=settings.openai_api_key
    )

def get_client() -> AsyncOpenAI:
    return _openai_compat()

async def embed_one(
        text: str, 
        model: str = "text-embedding-3-small",
        ) -> List[float]:
    """Returns a single embedding for one input string"""
    client = get_client()

    embed = await client.embeddings.create(
        model=model,
        input=[text],
    )

async def embed_many(
        texts: Sequence[str], 
        model: str = "text-embedding-3-small",
) -> List[list[float]]:
    """Returns one embedding per input string, preserves order"""
    
    if not texts:
        return []

    client = get_client()
    batch = await client.embeddings.create(
        model=model,
        input=list(texts)
    )

    return [d.embedding for d in batch.data]


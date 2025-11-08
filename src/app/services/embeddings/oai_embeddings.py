from openai import AsyncOpenAI
from functools import lru_cache
from typing import List, Optional, Sequence

from src.app.settings import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def _openai_compat(base_url: Optional[str] = None) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key=settings.openai_api_key)


def get_client() -> AsyncOpenAI:
    return _openai_compat()


async def embed_one(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Return a single embedding for one input string."""
    client = get_client()
    resp = await client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding  # type: ignore[return-value]


async def embed_many(texts: Sequence[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Return one embedding per input string, preserves order."""
    if not texts:
        return []
    client = get_client()
    resp = await client.embeddings.create(model=model, input=list(texts))
    return [d.embedding for d in resp.data]  # type: ignore[list-item]

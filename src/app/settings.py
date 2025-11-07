from typing import Optional
from functools import lru_cache

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: Optional[str] = Field(default=None)


@lru_cache(maxsize=1)
def get_settings():
    return Settings

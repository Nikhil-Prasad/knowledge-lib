from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load environment from .env for local dev; pydantic-settings also reads .env
load_dotenv()

# Centralized settings for DB URL
from src.app.settings import get_settings

_settings = get_settings()

engine = create_engine(
    _settings.sqlalchemy_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

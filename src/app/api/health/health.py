from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from src.app.api.deps import get_db


router = APIRouter(prefix="/health", tags=["health"])


class HealthStatus(BaseModel):
    status: str


class ReadinessStatus(BaseModel):
    status: str
    db: str


@router.get("/live", response_model=HealthStatus)
def live() -> HealthStatus:
    return HealthStatus(status="ok")


@router.get("/ready", response_model=ReadinessStatus)
def ready(db: Session = Depends(get_db)) -> ReadinessStatus:
    try:
        db.execute(sql_text("SELECT 1"))
        return ReadinessStatus(status="ok", db="ok")
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"db error: {e}")


# Back-compat alias if someone calls /healthz
@router.get("/healthz", include_in_schema=False)
def healthz(db: Session = Depends(get_db)) -> ReadinessStatus:  # pragma: no cover
    return ready(db)


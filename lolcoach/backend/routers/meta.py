"""GET/POST /api/meta — meta refresh status and trigger."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlmodel import Session, func, select

from ..config import DEFAULT_RATE_LIMIT, META_REFRESH_RATE_LIMIT
from ..db.database import get_session
from ..db.models_db import MetaCache
from ..models import MetaStatusResponse
from ..security import require_token
from ..services.meta_scraper import scrape_champion_builds

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/api/meta/status", response_model=MetaStatusResponse)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def meta_status(
    request,
    _token: str = Depends(require_token),
    session: Session = Depends(get_session),
):
    # Most recent entry
    stmt = select(MetaCache).order_by(MetaCache.updated_at.desc())
    latest = session.execute(stmt).scalars().first()

    count_stmt = select(func.count()).select_from(MetaCache)
    count = session.execute(count_stmt).scalar_one()

    return MetaStatusResponse(
        last_refresh=latest.updated_at if latest else None,
        patch=latest.patch if latest else None,
        champion_count=count,
    )


@router.post("/api/meta/refresh")
@limiter.limit(META_REFRESH_RATE_LIMIT)
async def trigger_meta_refresh(
    request,
    _token: str = Depends(require_token),
):
    updated = await scrape_champion_builds()
    return {"status": "ok", "updated": updated}

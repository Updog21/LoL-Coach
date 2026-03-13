"""POST /api/ask — build recommendation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlmodel import Session, select

from ..config import ASK_RATE_LIMIT
from ..db.database import get_session
from ..db.models_db import Setting
from ..models import AskRequest, BuildResponse
from ..security import require_token, validate_timestamp
from ..services.ai_router import query_build
from ..services.meta_scraper import get_meta_context
from ..services.rag import retrieve_context
from ..services.riot_poller import current_game_state

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/api/ask", response_model=BuildResponse)
@limiter.limit(ASK_RATE_LIMIT)
async def ask_build(
    request,
    body: AskRequest,
    _token: str = Depends(require_token),
    session: Session = Depends(get_session),
):
    # Replay protection
    validate_timestamp(body.timestamp)

    # Resolve provider from settings
    provider_row = session.get(Setting, "provider")
    provider = provider_row.value if provider_row else "openai"

    # Use live game state if not provided in request
    game_state = body.game_state or current_game_state

    # Gather context
    meta_ctx = get_meta_context(body.champion, body.role.value)
    rag_ctx = await retrieve_context(body.champion, body.question, provider=provider)

    # Query AI
    result = await query_build(
        provider=provider,
        question=body.question,
        champion=body.champion,
        role=body.role.value,
        game_state=game_state,
        meta_context=meta_ctx,
        rag_context=rag_ctx,
        screenshot_b64=body.screenshot_b64,
    )

    return result

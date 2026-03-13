"""GET/POST /api/settings — provider and API key management."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlmodel import Session, select

from ..config import DEFAULT_RATE_LIMIT, OLLAMA_BASE_URL
from ..db.database import get_session
from ..db.models_db import Setting
from ..models import ProviderName, SettingsResponse
from ..security import load_api_key, require_token, save_api_key

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class UpdateSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: ProviderName
    api_key: str | None = None
    ollama_base_url: str | None = None


@router.get("/api/settings", response_model=SettingsResponse)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_settings(
    request,
    _token: str = Depends(require_token),
    session: Session = Depends(get_session),
):
    provider_row = session.get(Setting, "provider")
    provider = provider_row.value if provider_row else "openai"
    has_key = load_api_key(provider) is not None

    ollama_url = None
    if provider == "ollama":
        url_row = session.get(Setting, "ollama_base_url")
        ollama_url = url_row.value if url_row else OLLAMA_BASE_URL

    return SettingsResponse(
        provider=ProviderName(provider),
        has_api_key=has_key,
        ollama_base_url=ollama_url,
    )


@router.post("/api/settings", response_model=SettingsResponse)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def update_settings(
    request,
    body: UpdateSettingsRequest,
    _token: str = Depends(require_token),
    session: Session = Depends(get_session),
):
    # Save provider
    _upsert_setting(session, "provider", body.provider.value)

    # Save API key to keyring if provided
    if body.api_key:
        save_api_key(body.provider.value, body.api_key)

    # Save Ollama URL if applicable
    if body.ollama_base_url:
        _upsert_setting(session, "ollama_base_url", body.ollama_base_url)

    session.commit()

    has_key = load_api_key(body.provider.value) is not None
    return SettingsResponse(
        provider=body.provider,
        has_api_key=has_key,
        ollama_base_url=body.ollama_base_url,
    )


def _upsert_setting(session: Session, key: str, value: str) -> None:
    existing = session.get(Setting, key)
    if existing:
        existing.value = value
    else:
        session.add(Setting(key=key, value=value))

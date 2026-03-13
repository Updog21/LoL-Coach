"""CRUD /api/resources — resource manager endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlmodel import Session, select

from ..config import DEFAULT_RATE_LIMIT, RESOURCE_RATE_LIMIT
from ..db.database import get_session
from ..db.models_db import Resource, Setting
from ..models import AddResourceRequest, ResourceResponse
from ..security import require_token
from ..services.rag import delete_resource_chunks, ingest_text
from ..services.resource_ingestor import extract_text

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/api/resources", response_model=list[ResourceResponse])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def list_resources(
    request,
    champion: str | None = None,
    _token: str = Depends(require_token),
    session: Session = Depends(get_session),
):
    stmt = select(Resource)
    if champion:
        stmt = stmt.where(Resource.champion == champion.lower())
    stmt = stmt.order_by(Resource.created_at.desc())
    results = session.execute(stmt).scalars().all()
    return results


@router.post("/api/resources", response_model=ResourceResponse, status_code=201)
@limiter.limit(RESOURCE_RATE_LIMIT)
async def add_resource(
    request,
    body: AddResourceRequest,
    _token: str = Depends(require_token),
    session: Session = Depends(get_session),
):
    champion = body.champion.lower()

    # Create the DB record
    resource = Resource(
        champion=champion,
        url=str(body.url) if body.url else None,
        label=body.label,
        type=body.type.value,
        status="pending",
    )
    session.add(resource)
    session.commit()
    session.refresh(resource)

    # Extract text from the resource
    try:
        text = await extract_text(body.type.value, str(body.url) if body.url else None, body.content)
    except Exception as e:
        resource.status = "error"
        session.add(resource)
        session.commit()
        logger.error("Failed to extract text for resource %d: %s", resource.id, e)
        raise HTTPException(status_code=422, detail=f"Failed to extract text: {e}")

    # Get provider for embeddings
    provider_row = session.get(Setting, "provider")
    provider = provider_row.value if provider_row else "openai"

    # Ingest into ChromaDB
    try:
        await ingest_text(
            champion=champion,
            text=text,
            source_url=str(body.url) if body.url else "",
            label=body.label,
            resource_id=resource.id,
            provider=provider,
        )
        resource.status = "ready"
    except Exception as e:
        resource.status = "error"
        logger.error("Failed to ingest resource %d: %s", resource.id, e)

    session.add(resource)
    session.commit()
    session.refresh(resource)
    return resource


@router.delete("/api/resources/{resource_id}", status_code=204)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def delete_resource(
    request,
    resource_id: int,
    _token: str = Depends(require_token),
    session: Session = Depends(get_session),
):
    resource = session.get(Resource, resource_id)
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")

    # Remove from ChromaDB
    await delete_resource_chunks(resource.champion, resource_id)

    # Remove from SQLite
    session.delete(resource)
    session.commit()

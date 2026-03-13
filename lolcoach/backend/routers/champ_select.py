"""GET /api/champ-select — live draft state + pick recommendations."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlmodel import Session, select

from ..config import ASK_RATE_LIMIT, DEFAULT_RATE_LIMIT
from ..db.database import get_session
from ..db.models_db import Setting
from ..models import ChampSelectResponse, PickRecommendation
from ..security import require_token
from ..services import lcu_poller
from ..services.ai_router import _get_chat_model
from ..services.meta_scraper import get_meta_context

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


PICK_ADVISOR_PROMPT = """\
You are a League of Legends draft coach. Given the current draft state, recommend \
the best champion picks for the player.

Current draft state:
- Player's assigned role: {role}
- Player's hovered champion: {hover}
- Ally picks so far: {allies}
- Enemy picks so far: {enemies}
- Banned champions: {bans}

{meta_context}

Rules:
1. Recommend 3-5 champions for the player's role.
2. For each, explain WHY it's a good pick given the enemy comp and ally comp.
3. Consider: counter-picks to visible enemies, team composition gaps (AP/AD balance, \
engage, peel), and current meta strength.
4. If the player is hovering a champion, evaluate that pick first — tell them if it's \
good or bad for this game and why.
5. Note any champions the team is missing (e.g. "team lacks engage" or "all AD").

Respond in valid JSON:
{{
  "hover_evaluation": "Good pick because... / Consider swapping because...",
  "team_needs": "Team lacks AP damage and hard engage",
  "recommendations": [
    {{
      "champion": "ChampionName",
      "role": "ROLE",
      "reason": "Why this pick is strong here",
      "win_rate": 52.5,
      "meta_tier": "S",
      "counters": ["EnemyChamp1", "EnemyChamp2"]
    }}
  ]
}}
"""


@router.get("/api/champ-select", response_model=ChampSelectResponse)
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_champ_select_state(
    request,
    _token: str = Depends(require_token),
):
    """Return the current champ select state (no AI, just draft data)."""
    state = lcu_poller.current_champ_select
    if state is None:
        return ChampSelectResponse(phase="NONE", my_role="")

    # Get meta context for the hovered champion
    hover_meta = ""
    if state.my_champion_name and state.my_role:
        hover_meta = get_meta_context(state.my_champion_name, state.my_role)

    return ChampSelectResponse(
        phase=state.phase,
        my_role=state.my_role,
        hover=state.my_champion_name or None,
        hover_meta=hover_meta or None,
    )


@router.post("/api/champ-select/recommend", response_model=ChampSelectResponse)
@limiter.limit(ASK_RATE_LIMIT)
async def recommend_picks(
    request,
    _token: str = Depends(require_token),
    session: Session = Depends(get_session),
):
    """AI-powered pick recommendations based on current draft state."""
    state = lcu_poller.current_champ_select
    if state is None:
        return ChampSelectResponse(phase="NONE", my_role="")

    # Resolve provider
    provider_row = session.get(Setting, "provider")
    provider = provider_row.value if provider_row else "openai"

    # Gather meta context for likely picks in this role
    meta_lines: list[str] = []
    if state.my_champion_name and state.my_role:
        ctx = get_meta_context(state.my_champion_name, state.my_role)
        if ctx:
            meta_lines.append(f"Hovered champion meta:\n{ctx}")

    # Add enemy meta context
    for enemy in state.enemy_picks:
        # We don't know enemy roles, so just get their general meta
        for role in ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"]:
            ctx = get_meta_context(enemy, role)
            if ctx:
                meta_lines.append(f"{enemy} meta:\n{ctx}")
                break

    prompt = PICK_ADVISOR_PROMPT.format(
        role=state.my_role or "UNKNOWN",
        hover=state.my_champion_name or "None",
        allies=", ".join(state.ally_picks) if state.ally_picks else "None yet",
        enemies=", ".join(state.enemy_picks) if state.enemy_picks else "None yet",
        bans=", ".join(state.banned) if state.banned else "None",
        meta_context="\n".join(meta_lines) if meta_lines else "No meta data available.",
    )

    # Query the AI
    try:
        llm = _get_chat_model(provider)
        messages = [
            {"role": "system", "content": "You are a League of Legends draft analyst."},
            {"role": "user", "content": prompt},
        ]
        response = await llm.ainvoke(messages)
        content = response.content

        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        parsed = json.loads(content.strip())

        recommendations = [
            PickRecommendation(**rec)
            for rec in parsed.get("recommendations", [])
        ]

        return ChampSelectResponse(
            phase=state.phase,
            my_role=state.my_role,
            hover=state.my_champion_name or None,
            hover_meta=parsed.get("hover_evaluation"),
            recommendations=recommendations,
            team_needs=parsed.get("team_needs", ""),
        )

    except Exception as exc:
        logger.error("Pick recommendation failed: %s", exc)
        # Return state without AI recommendations on failure
        return ChampSelectResponse(
            phase=state.phase,
            my_role=state.my_role,
            hover=state.my_champion_name or None,
        )

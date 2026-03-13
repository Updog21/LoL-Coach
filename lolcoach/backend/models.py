"""Pydantic models — strict, extra=forbid."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Role(str, Enum):
    TOP = "TOP"
    JUNGLE = "JUNGLE"
    MID = "MID"
    BOT = "BOT"
    SUPPORT = "SUPPORT"


class ResourceType(str, Enum):
    YOUTUBE = "youtube"
    ARTICLE = "article"
    PDF = "pdf"
    NOTE = "note"


class ProviderName(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


# ---------------------------------------------------------------------------
# Game state (from Riot Live Client API)
# ---------------------------------------------------------------------------
class ItemSlim(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    id: int = Field(..., alias="itemID")
    name: str = Field(..., alias="displayName")


class PlayerSlim(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    summoner_name: str = Field(..., alias="summonerName")
    champion_name: str = Field(..., alias="championName")
    team: str
    position: str
    is_dead: bool = Field(..., alias="isDead")
    items: list[ItemSlim] = Field(default_factory=list)


class ActivePlayer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    summoner_name: str = Field(..., alias="summonerName")
    champion_name: str = Field(..., alias="championName")
    level: int
    current_gold: float = Field(..., alias="currentGold")


class GameState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    active_player: ActivePlayer = Field(..., alias="activePlayer")
    allies: list[PlayerSlim]
    enemies: list[PlayerSlim]
    game_time: float = Field(..., alias="gameTime")
    game_mode: str = Field(..., alias="gameMode")


# ---------------------------------------------------------------------------
# API requests
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    question: str = Field(..., max_length=500)
    champion: str = Field(..., max_length=32)
    role: Role
    game_state: Optional[GameState] = Field(None, alias="gameState")
    screenshot_b64: Optional[str] = Field(None, alias="screenshotB64")
    timestamp: datetime


class AddResourceRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    champion: str = Field(..., max_length=32, pattern=r"^[a-zA-Z]+$")
    url: Optional[HttpUrl] = None
    label: str = Field(..., max_length=120)
    type: ResourceType
    content: Optional[str] = None  # for notes


# ---------------------------------------------------------------------------
# API responses
# ---------------------------------------------------------------------------
class ItemRecommendation(BaseModel):
    name: str
    cost: int
    reason: str


class RuneRecommendation(BaseModel):
    keystone: str
    secondary: str
    shards: str


class BuildResponse(BaseModel):
    items: list[ItemRecommendation]
    runes: Optional[RuneRecommendation] = None
    skill_order: str = Field(..., alias="skillOrder")
    rationale: str


class ResourceResponse(BaseModel):
    id: int
    champion: str
    url: Optional[str] = None
    label: str
    type: ResourceType
    status: str
    created_at: datetime


class SettingsResponse(BaseModel):
    provider: ProviderName
    has_api_key: bool
    ollama_base_url: Optional[str] = None


class MetaStatusResponse(BaseModel):
    last_refresh: Optional[datetime] = None
    patch: Optional[str] = None
    champion_count: int = 0

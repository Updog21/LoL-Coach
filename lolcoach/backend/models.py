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


class PlayerRunes(BaseModel):
    model_config = ConfigDict(extra="allow")
    keystone: str = ""
    primary_tree: str = Field("", alias="primaryTree")
    secondary_tree: str = Field("", alias="secondaryTree")


class PlayerScores(BaseModel):
    model_config = ConfigDict(extra="allow")
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    creep_score: int = Field(0, alias="creepScore")
    ward_score: float = Field(0.0, alias="wardScore")


class PlayerSlim(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    summoner_name: str = Field(..., alias="summonerName")
    champion_name: str = Field(..., alias="championName")
    team: str
    position: str
    level: int = 1
    is_dead: bool = Field(..., alias="isDead")
    items: list[ItemSlim] = Field(default_factory=list)
    runes: PlayerRunes = Field(default_factory=PlayerRunes)
    scores: PlayerScores = Field(default_factory=PlayerScores)


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
# Champ select state (from LCU API)
# ---------------------------------------------------------------------------
class ChampSelectAction(BaseModel):
    """A single pick/ban action in champ select."""
    model_config = ConfigDict(extra="allow")
    id: int
    actor_cell_id: int = Field(..., alias="actorCellId")
    champion_id: int = Field(..., alias="championId")
    type: str  # "pick" | "ban"
    completed: bool
    is_in_progress: bool = Field(False, alias="isInProgress")


class ChampSelectTeammate(BaseModel):
    """A teammate slot in champ select."""
    model_config = ConfigDict(extra="allow")
    cell_id: int = Field(..., alias="cellId")
    champion_id: int = Field(..., alias="championId")
    summoner_id: int = Field(0, alias="summonerId")
    assigned_position: str = Field("", alias="assignedPosition")
    spell1_id: int = Field(0, alias="spell1Id")
    spell2_id: int = Field(0, alias="spell2Id")


class ChampSelectState(BaseModel):
    """Current champion select session state."""
    model_config = ConfigDict(extra="allow")
    phase: str = ""  # PLANNING | BAN_PICK | FINALIZATION
    local_player_cell_id: int = Field(0, alias="localPlayerCellId")
    my_team: list[ChampSelectTeammate] = Field(default_factory=list, alias="myTeam")
    their_team: list[ChampSelectTeammate] = Field(default_factory=list, alias="theirTeam")
    actions: list[list[ChampSelectAction]] = Field(default_factory=list)
    # Derived fields (populated by our poller)
    my_role: str = ""
    my_champion_id: int = 0
    my_champion_name: str = ""
    ally_picks: list[str] = Field(default_factory=list)
    enemy_picks: list[str] = Field(default_factory=list)
    banned: list[str] = Field(default_factory=list)


class PickRecommendation(BaseModel):
    """A single champion pick recommendation."""
    champion: str
    role: str
    reason: str
    win_rate: float = 0.0
    meta_tier: str = ""  # S, A, B, C
    counters: list[str] = Field(default_factory=list)  # who it's strong against


class ChampSelectResponse(BaseModel):
    """Response for the champ select recommendation endpoint."""
    phase: str
    my_role: str
    hover: Optional[str] = None
    hover_meta: Optional[str] = None  # meta context for hovered champ
    recommendations: list[PickRecommendation] = Field(default_factory=list)
    team_needs: str = ""  # e.g. "Team lacks AP damage and engage"


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


class LanePreview(BaseModel):
    """Early game laning phase breakdown."""
    summary: str  # 1-2 sentence overview of what to expect
    power_spikes: str  # when you're strong vs when they're strong
    threats: str  # what to watch out for (their kill combo, gank setup, etc.)
    tips: str  # actionable advice (spacing, trading patterns, wave management)
    lane_partner_synergy: Optional[str] = Field(None, alias="lanePartnerSynergy")  # bot lane only


class BuildResponse(BaseModel):
    items: list[ItemRecommendation]
    runes: Optional[RuneRecommendation] = None
    skill_order: str = Field(..., alias="skillOrder")
    rationale: str
    lane_preview: Optional[LanePreview] = Field(None, alias="lanePreview")


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

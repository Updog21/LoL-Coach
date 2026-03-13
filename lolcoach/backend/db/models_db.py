"""SQLModel ORM models — Resource, Setting, MetaCache."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class Resource(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    champion: str = Field(index=True)
    url: Optional[str] = None
    label: str
    type: str  # youtube | article | pdf | note
    status: str = Field(default="pending")  # pending | ready | error
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Setting(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str


class MetaCache(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    champion: str = Field(index=True)
    role: str
    patch: str
    source: str = Field(default="u.gg")  # u.gg | probuilds
    core_items: str  # JSON list of item names
    rune_page: str  # JSON
    win_rate: float
    pick_rate: float
    pro_builds: Optional[str] = None  # JSON list of recent pro builds from probuilds.net
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

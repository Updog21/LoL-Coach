"""u.gg meta scraper + Riot Data Dragon sync."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup
from sqlmodel import Session, select

from ..db.database import engine
from ..db.models_db import MetaCache

logger = logging.getLogger(__name__)

DATA_DRAGON_VERSIONS = "https://ddragon.leagueoflegends.com/api/versions.json"

ROLES = ["top", "jungle", "mid", "adc", "support"]
ROLE_MAP = {"top": "TOP", "jungle": "JUNGLE", "mid": "MID", "adc": "BOT", "support": "SUPPORT"}


async def get_current_patch() -> str:
    """Fetch the latest patch version from Data Dragon."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(DATA_DRAGON_VERSIONS)
        resp.raise_for_status()
        versions = resp.json()
        return versions[0]


async def scrape_champion_builds() -> int:
    """Scrape top win-rate builds from u.gg for all champions. Returns count updated."""
    patch = await get_current_patch()
    updated = 0

    async with httpx.AsyncClient(timeout=15.0) as client:
        champ_url = f"https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/champion.json"
        resp = await client.get(champ_url)
        resp.raise_for_status()
        champions = list(resp.json()["data"].keys())

        for champ in champions:
            for role in ROLES:
                try:
                    build_data = await _scrape_champion_role(client, champ, role, patch)
                    if build_data:
                        _upsert_meta(champ, ROLE_MAP[role], patch, build_data)
                        updated += 1
                except Exception as exc:
                    logger.debug("Failed to scrape %s/%s: %s", champ, role, exc)
                    continue

    logger.info("Meta refresh complete: %d entries updated for patch %s", updated, patch)
    return updated


async def _scrape_champion_role(
    client: httpx.AsyncClient, champion: str, role: str, patch: str
) -> dict | None:
    """Scrape a single champion+role build page from u.gg."""
    url = f"https://u.gg/lol/champions/{champion.lower()}/build/{role}"
    try:
        resp = await client.get(url, follow_redirects=True)
        if resp.status_code != 200:
            return None
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    core_items: list[str] = []
    rune_page = ""
    win_rate = 0.0
    pick_rate = 0.0

    for item_el in soup.select(".recommended-build_items .item-img"):
        alt = item_el.get("alt", "")
        if alt:
            core_items.append(alt)

    for wr_el in soup.select(".win-rate"):
        text = wr_el.get_text(strip=True).replace("%", "")
        try:
            win_rate = float(text)
            break
        except ValueError:
            continue

    if not core_items:
        return None

    return {
        "core_items": core_items[:6],
        "rune_page": rune_page,
        "win_rate": win_rate,
        "pick_rate": pick_rate,
    }


def _upsert_meta(champion: str, role: str, patch: str, data: dict) -> None:
    """Insert or update a MetaCache entry."""
    with Session(engine) as session:
        stmt = select(MetaCache).where(
            MetaCache.champion == champion,
            MetaCache.role == role,
            MetaCache.patch == patch,
        )
        existing = session.exec(stmt).first()

        if existing:
            existing.core_items = json.dumps(data["core_items"])
            existing.rune_page = data["rune_page"]
            existing.win_rate = data["win_rate"]
            existing.pick_rate = data["pick_rate"]
            existing.updated_at = datetime.now(timezone.utc)
        else:
            entry = MetaCache(
                champion=champion,
                role=role,
                patch=patch,
                core_items=json.dumps(data["core_items"]),
                rune_page=data["rune_page"],
                win_rate=data["win_rate"],
                pick_rate=data["pick_rate"],
            )
            session.add(entry)

        session.commit()


def get_meta_context(champion: str, role: str) -> str:
    """Retrieve formatted meta build context for a champion+role from SQLite."""
    with Session(engine) as session:
        stmt = (
            select(MetaCache)
            .where(MetaCache.champion == champion, MetaCache.role == role)
            .order_by(MetaCache.updated_at.desc())
        )
        meta = session.exec(stmt).first()

    if not meta:
        return ""

    items = json.loads(meta.core_items)
    return (
        f"Meta build (patch {meta.patch}, {meta.win_rate:.1f}% WR):\n"
        f"Core items: {', '.join(items)}\n"
        f"Runes: {meta.rune_page or 'N/A'}"
    )

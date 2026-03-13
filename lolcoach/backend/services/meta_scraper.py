"""Meta scraper — u.gg (SSR data) + probuilds.net + Riot Data Dragon."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup
from sqlmodel import Session, select

from ..db.database import engine
from ..db.models_db import MetaCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Riot Data Dragon
# ---------------------------------------------------------------------------
DATA_DRAGON_VERSIONS = "https://ddragon.leagueoflegends.com/api/versions.json"
DATA_DRAGON_ITEMS = "https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/item.json"
DATA_DRAGON_RUNES = "https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/runesReforged.json"
DATA_DRAGON_CHAMPS = "https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/champion.json"
DATA_DRAGON_CHAMP_DETAIL = (
    "https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/champion/{champion_id}.json"
)

ROLES = ["top", "jungle", "mid", "adc", "support"]
ROLE_MAP = {"top": "TOP", "jungle": "JUNGLE", "mid": "MID", "adc": "BOT", "support": "SUPPORT"}

# Shared HTTP headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ---------------------------------------------------------------------------
# Item / rune ID → name caches (loaded from Data Dragon once per refresh)
# ---------------------------------------------------------------------------
_item_id_to_name: dict[str, str] = {}
_rune_id_to_name: dict[str, str] = {}
_champion_lookup: dict[str, str] = {}
_champion_kit_cache: dict[str, dict] = {}


async def get_current_patch() -> str:
    """Fetch the latest patch version from Data Dragon."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(DATA_DRAGON_VERSIONS)
        resp.raise_for_status()
        return resp.json()[0]


async def _load_item_names(client: httpx.AsyncClient, patch: str) -> None:
    """Populate item ID → name mapping from Data Dragon."""
    global _item_id_to_name
    if _item_id_to_name:
        return
    resp = await client.get(DATA_DRAGON_ITEMS.format(patch=patch))
    resp.raise_for_status()
    data = resp.json()["data"]
    _item_id_to_name = {item_id: info["name"] for item_id, info in data.items()}
    logger.info("Loaded %d item names from Data Dragon", len(_item_id_to_name))


async def _load_rune_names(client: httpx.AsyncClient, patch: str) -> None:
    """Populate rune ID → name mapping from Data Dragon."""
    global _rune_id_to_name
    if _rune_id_to_name:
        return
    resp = await client.get(DATA_DRAGON_RUNES.format(patch=patch))
    resp.raise_for_status()
    for tree in resp.json():
        _rune_id_to_name[str(tree["id"])] = tree["name"]
        for slot in tree["slots"]:
            for rune in slot["runes"]:
                _rune_id_to_name[str(rune["id"])] = rune["name"]
    logger.info("Loaded %d rune names from Data Dragon", len(_rune_id_to_name))


def _normalize_champion_ref(value: str) -> str:
    """Normalize champion ids/names into a lookup-friendly key."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


async def _load_champion_lookup(client: httpx.AsyncClient, patch: str) -> None:
    """Populate champion name/id aliases from Data Dragon."""
    global _champion_lookup
    if _champion_lookup:
        return

    resp = await client.get(DATA_DRAGON_CHAMPS.format(patch=patch))
    resp.raise_for_status()
    for champ in resp.json()["data"].values():
        champ_id = champ["id"]
        aliases = {
            champ_id,
            champ.get("name", ""),
            champ.get("key", ""),
        }
        for alias in aliases:
            normalized = _normalize_champion_ref(alias)
            if normalized:
                _champion_lookup[normalized] = champ_id

    logger.info("Loaded %d champion aliases from Data Dragon", len(_champion_lookup))


def _clean_html_text(value: str) -> str:
    """Convert Data Dragon HTML snippets to compact plain text."""
    text = BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _truncate_summary(value: str, limit: int = 200) -> str:
    """Keep spell descriptions compact enough for prompt use."""
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


async def _load_champion_kit(
    client: httpx.AsyncClient, patch: str, champion: str
) -> dict | None:
    """Fetch and cache Data Dragon champion ability data."""
    await _load_champion_lookup(client, patch)

    champion_id = _champion_lookup.get(_normalize_champion_ref(champion))
    if not champion_id:
        return None

    if champion_id in _champion_kit_cache:
        return _champion_kit_cache[champion_id]

    resp = await client.get(DATA_DRAGON_CHAMP_DETAIL.format(patch=patch, champion_id=champion_id))
    resp.raise_for_status()
    raw = resp.json()["data"][champion_id]

    payload = {
        "id": champion_id,
        "name": raw.get("name", champion),
        "title": raw.get("title", ""),
        "tags": raw.get("tags", []),
        "passive": {
            "name": raw.get("passive", {}).get("name", ""),
            "description": _truncate_summary(
                _clean_html_text(raw.get("passive", {}).get("description", ""))
            ),
        },
        "spells": [
            {
                "slot": slot,
                "name": spell.get("name", ""),
                "description": _truncate_summary(_clean_html_text(spell.get("description", ""))),
            }
            for slot, spell in zip(("Q", "W", "E", "R"), raw.get("spells", []), strict=False)
        ],
    }
    _champion_kit_cache[champion_id] = payload
    return payload


def _format_champion_kit(heading: str, champion_kit: dict) -> str:
    """Format one champion's kit for the AI prompt."""
    tags = ", ".join(champion_kit.get("tags", [])) or "Unknown"
    title = champion_kit.get("title", "")
    prefix = f"{champion_kit['name']} ({title})" if title else champion_kit["name"]

    lines = [f"{heading}: {prefix} | Tags: {tags}"]
    passive = champion_kit.get("passive", {})
    if passive.get("name") and passive.get("description"):
        lines.append(f"Passive - {passive['name']}: {passive['description']}")
    for spell in champion_kit.get("spells", []):
        if spell.get("name") and spell.get("description"):
            lines.append(f"{spell['slot']} - {spell['name']}: {spell['description']}")
    return "\n".join(lines)


async def get_ability_context(
    champion: str,
    role: str,
    enemy_champions: list[str],
    lane_opponent: str = "",
    lane_partner: str = "",
    enemy_jungler: str = "",
    ally_jungler: str = "",
) -> str:
    """Build champion-kit context for the player's champ and current threats."""
    if not champion:
        return ""

    patch = await get_current_patch()
    lines: list[str] = []
    seen: set[str] = set()

    async with httpx.AsyncClient(timeout=15.0) as client:
        requests: list[tuple[str, str]] = [("YOUR CHAMPION", champion)]
        if lane_opponent:
            requests.append(("LANE OPPONENT", lane_opponent))
        if lane_partner:
            requests.append(("LANE PARTNER", lane_partner))
        if ally_jungler:
            requests.append(("ALLY JUNGLER", ally_jungler))
        if enemy_jungler:
            requests.append(("ENEMY JUNGLER", enemy_jungler))
        for enemy in enemy_champions:
            if enemy:
                requests.append(("ENEMY THREAT", enemy))

        tasks = []
        for heading, champ_name in requests:
            normalized = _normalize_champion_ref(champ_name)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            tasks.append((heading, champ_name, _load_champion_kit(client, patch, champ_name)))

        results = await asyncio.gather(*(task for _, _, task in tasks), return_exceptions=True)

    for (heading, champ_name, _task), result in zip(tasks, results, strict=False):
        if isinstance(result, Exception):
            logger.debug("Failed to load ability data for %s: %s", champ_name, result)
            continue
        if result is None:
            continue

        if heading == "ENEMY THREAT":
            heading = f"ENEMY THREAT ({role})"
        lines.append(_format_champion_kit(heading, result))

    return "\n\n".join(lines)


def _item_name(item_id: str) -> str:
    """Resolve item ID to name, falling back to the ID itself."""
    return _item_id_to_name.get(str(item_id), f"Item#{item_id}")


def _rune_name(rune_id: str) -> str:
    """Resolve rune ID to name, falling back to the ID itself."""
    return _rune_id_to_name.get(str(rune_id), f"Rune#{rune_id}")


# ---------------------------------------------------------------------------
# u.gg scraper — extract embedded __SSR_DATA__
# ---------------------------------------------------------------------------
async def _scrape_ugg(
    client: httpx.AsyncClient, champion: str, role: str, patch: str
) -> dict | None:
    """Extract build data from u.gg's embedded SSR JSON."""
    url = f"https://u.gg/lol/champions/{champion.lower()}/build/{role}"
    try:
        resp = await client.get(url, follow_redirects=True, headers=HEADERS)
        if resp.status_code != 200:
            return None
    except Exception:
        return None

    html = resp.text

    # Extract __SSR_DATA__ JSON blob
    match = re.search(r"window\.__SSR_DATA__\s*=\s*({.+?})\s*;?\s*</script>", html, re.DOTALL)
    if not match:
        # Fallback: try parsing visible HTML for any embedded stats
        return _parse_ugg_html(html)

    try:
        ssr = json.loads(match.group(1))
    except json.JSONDecodeError:
        return _parse_ugg_html(html)

    # Navigate the SSR data structure for build info
    # Structure varies — search for champion build data in the payload
    return _extract_ugg_build(ssr, champion, role)


def _extract_ugg_build(ssr: dict, champion: str, role: str) -> dict | None:
    """Walk the u.gg SSR data tree to find build stats."""
    # The SSR data contains page props with champion overview data
    # Try common paths in the nested structure
    try:
        # Search recursively for win_rate / items data
        text = json.dumps(ssr)

        # Look for item arrays (lists of item IDs)
        item_ids: list[str] = []
        rune_ids: list[str] = []
        win_rate = 0.0
        pick_rate = 0.0

        # Find core items pattern — typically arrays of item IDs
        item_matches = re.findall(r'"recommended_build_ids?":\s*\[([^\]]+)\]', text)
        if item_matches:
            for m in item_matches:
                ids = [x.strip().strip('"') for x in m.split(",")]
                item_ids = [_item_name(i) for i in ids if i.isdigit()]
                if item_ids:
                    break

        # Find win rate
        wr_match = re.search(r'"win_rate":\s*([\d.]+)', text)
        if wr_match:
            wr = float(wr_match.group(1))
            win_rate = wr if wr < 1 else wr  # could be 0.55 or 55.0
            if win_rate < 1:
                win_rate *= 100

        # Find pick rate
        pr_match = re.search(r'"pick_rate":\s*([\d.]+)', text)
        if pr_match:
            pr = float(pr_match.group(1))
            pick_rate = pr if pr > 1 else pr * 100

        # Find rune/keystone IDs
        rune_match = re.search(r'"keystone_id":\s*(\d+)', text)
        if rune_match:
            rune_ids.append(rune_match.group(1))

        if not item_ids:
            return None

        return {
            "core_items": item_ids[:6],
            "rune_page": ", ".join(_rune_name(r) for r in rune_ids) if rune_ids else "",
            "win_rate": win_rate,
            "pick_rate": pick_rate,
        }
    except Exception as exc:
        logger.debug("Failed to extract u.gg SSR data: %s", exc)
        return None


def _parse_ugg_html(html: str) -> dict | None:
    """Fallback: parse u.gg HTML for any visible build data."""
    soup = BeautifulSoup(html, "html.parser")

    # Extract item images — item IDs are in image src URLs
    item_ids: list[str] = []
    for img in soup.find_all("img"):
        src = img.get("src", "")
        # Match patterns like /lol/item/3031.png or /item/3031.webp
        m = re.search(r"/item/(\d+)\.", src)
        if m and m.group(1) not in item_ids:
            item_ids.append(m.group(1))

    # Extract rune images
    rune_ids: list[str] = []
    for img in soup.find_all("img"):
        src = img.get("src", "")
        m = re.search(r"/rune(?:s)?/(?:all/)?(\d+)\.", src)
        if m and m.group(1) not in rune_ids:
            rune_ids.append(m.group(1))

    # Extract win rate from text
    win_rate = 0.0
    wr_text = soup.find(string=re.compile(r"\d+\.?\d*\s*%"))
    if wr_text:
        m = re.search(r"([\d.]+)\s*%", wr_text)
        if m:
            win_rate = float(m.group(1))

    if not item_ids:
        return None

    return {
        "core_items": [_item_name(i) for i in item_ids[:6]],
        "rune_page": ", ".join(_rune_name(r) for r in rune_ids[:2]) if rune_ids else "",
        "win_rate": win_rate,
        "pick_rate": 0.0,
    }


# ---------------------------------------------------------------------------
# probuilds.net scraper — HTML parse (Astro SSR, data in markup)
# ---------------------------------------------------------------------------
async def _scrape_probuilds(
    client: httpx.AsyncClient, champion: str, role: str
) -> list[dict]:
    """Scrape recent pro builds from probuilds.net. Returns list of build dicts."""
    # probuilds uses PascalCase champion names and uppercase role in URL
    role_url = role.upper() if role != "adc" else "ADC"
    url = f"https://probuilds.net/champions/details/{champion}/{role_url}"

    try:
        resp = await client.get(url, follow_redirects=True, headers=HEADERS)
        if resp.status_code != 200:
            return []
    except Exception:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    builds: list[dict] = []

    # --- Aggregate common items (top of page) ---
    common_items: list[str] = []
    aggregate_keystone = ""

    # Common items are img tags with src containing /lol/item/{id}.webp
    for section in soup.find_all(attrs={"class": re.compile(r"aggregate")}):
        for img in section.find_all("img"):
            src = img.get("src", "")
            m = re.search(r"/item/(\d+)\.", src)
            if m:
                common_items.append(_item_name(m.group(1)))
            # Keystone rune
            m = re.search(r"/runes?/(?:all/)?(\d+)\.", src)
            if m and not aggregate_keystone:
                aggregate_keystone = _rune_name(m.group(1))

    # Win rate from aggregate section
    agg_win_rate = 0.0
    for el in soup.find_all(string=re.compile(r"\d+%")):
        m = re.search(r"(\d+)%", str(el))
        if m:
            agg_win_rate = float(m.group(1))
            break

    if common_items:
        builds.append({
            "source": "probuilds_aggregate",
            "items": common_items[:6],
            "keystone": aggregate_keystone,
            "win_rate": agg_win_rate,
            "player": "aggregate",
        })

    # --- Individual match builds ---
    # Each match is a link with class containing "match"
    match_links = soup.find_all(attrs={"class": re.compile(r"match")})
    seen = 0
    for match_el in match_links:
        if seen >= 10:  # Cap at 10 recent pro games
            break

        items: list[str] = []
        runes: list[str] = []

        for img in match_el.find_all("img"):
            src = img.get("src", "")
            m = re.search(r"/item/(\d+)\.", src)
            if m:
                items.append(_item_name(m.group(1)))
            m = re.search(r"/runes?/(?:all/)?(\d+)\.", src)
            if m:
                runes.append(_rune_name(m.group(1)))

        if not items:
            continue

        # Extract KDA text
        kda = ""
        kda_el = match_el.find(string=re.compile(r"\d+\s*/\s*\d+\s*/\s*\d+"))
        if kda_el:
            kda = kda_el.strip()

        builds.append({
            "source": "probuilds_match",
            "items": items[:6],
            "keystone": runes[0] if runes else "",
            "kda": kda,
            "player": "",  # player name extraction is fragile, skip
        })
        seen += 1

    return builds


# ---------------------------------------------------------------------------
# Orchestrator — scrape all sources
# ---------------------------------------------------------------------------
async def scrape_champion_builds() -> int:
    """Scrape builds from u.gg + probuilds.net for all champions. Returns count updated."""
    patch = await get_current_patch()
    updated = 0

    async with httpx.AsyncClient(timeout=20.0) as client:
        # Load ID → name mappings
        await _load_item_names(client, patch)
        await _load_rune_names(client, patch)

        # Get champion list from Data Dragon
        resp = await client.get(DATA_DRAGON_CHAMPS.format(patch=patch))
        resp.raise_for_status()
        champions = list(resp.json()["data"].keys())

        for champ in champions:
            for role in ROLES:
                try:
                    # u.gg — primary meta stats
                    ugg_data = await _scrape_ugg(client, champ, role, patch)

                    # probuilds.net — pro player builds
                    pro_builds = await _scrape_probuilds(client, champ, role)

                    if ugg_data or pro_builds:
                        _upsert_meta(
                            champion=champ,
                            role=ROLE_MAP[role],
                            patch=patch,
                            ugg_data=ugg_data,
                            pro_builds=pro_builds,
                        )
                        updated += 1
                except Exception as exc:
                    logger.debug("Failed to scrape %s/%s: %s", champ, role, exc)
                    continue

    logger.info("Meta refresh complete: %d entries updated for patch %s", updated, patch)
    return updated


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------
def _upsert_meta(
    champion: str,
    role: str,
    patch: str,
    ugg_data: dict | None,
    pro_builds: list[dict],
) -> None:
    """Insert or update a MetaCache entry with combined data."""
    core_items = ugg_data["core_items"] if ugg_data else []
    rune_page = ugg_data.get("rune_page", "") if ugg_data else ""
    win_rate = ugg_data.get("win_rate", 0.0) if ugg_data else 0.0
    pick_rate = ugg_data.get("pick_rate", 0.0) if ugg_data else 0.0

    # If u.gg had no items but probuilds did, use probuilds aggregate
    if not core_items and pro_builds:
        for pb in pro_builds:
            if pb.get("source") == "probuilds_aggregate" and pb.get("items"):
                core_items = pb["items"]
                rune_page = pb.get("keystone", "")
                win_rate = pb.get("win_rate", 0.0)
                break

    pro_json = json.dumps(pro_builds) if pro_builds else None

    with Session(engine) as session:
        stmt = select(MetaCache).where(
            MetaCache.champion == champion,
            MetaCache.role == role,
            MetaCache.patch == patch,
        )
        existing = session.execute(stmt).scalars().first()

        if existing:
            existing.core_items = json.dumps(core_items)
            existing.rune_page = rune_page
            existing.win_rate = win_rate
            existing.pick_rate = pick_rate
            existing.pro_builds = pro_json
            existing.source = "u.gg+probuilds"
            existing.updated_at = datetime.now(timezone.utc)
        else:
            entry = MetaCache(
                champion=champion,
                role=role,
                patch=patch,
                source="u.gg+probuilds",
                core_items=json.dumps(core_items),
                rune_page=rune_page,
                win_rate=win_rate,
                pick_rate=pick_rate,
                pro_builds=pro_json,
            )
            session.add(entry)

        session.commit()


# ---------------------------------------------------------------------------
# Context retrieval (used by AI router)
# ---------------------------------------------------------------------------
def get_meta_context(champion: str, role: str) -> str:
    """Retrieve formatted meta + pro build context for the AI prompt."""
    with Session(engine) as session:
        stmt = (
            select(MetaCache)
            .where(MetaCache.champion == champion, MetaCache.role == role)
            .order_by(MetaCache.updated_at.desc())
        )
        meta = session.execute(stmt).scalars().first()

    if not meta:
        return ""

    parts: list[str] = []

    # u.gg meta stats
    items = json.loads(meta.core_items) if meta.core_items else []
    if items:
        parts.append(
            f"Meta build (patch {meta.patch}, {meta.win_rate:.1f}% WR, "
            f"{meta.pick_rate:.1f}% pick rate):\n"
            f"Core items: {', '.join(items)}\n"
            f"Runes: {meta.rune_page or 'N/A'}"
        )

    # Pro builds from probuilds.net
    if meta.pro_builds:
        pro_data = json.loads(meta.pro_builds)
        pro_parts: list[str] = []
        for pb in pro_data[:5]:  # Limit to 5 most relevant
            src = pb.get("source", "")
            items_str = ", ".join(pb.get("items", []))
            if src == "probuilds_aggregate":
                pro_parts.append(
                    f"  Pro aggregate: {items_str} "
                    f"(keystone: {pb.get('keystone', 'N/A')}, "
                    f"{pb.get('win_rate', 0):.0f}% WR)"
                )
            elif items_str:
                kda = pb.get("kda", "")
                pro_parts.append(
                    f"  Pro game: {items_str} "
                    f"(keystone: {pb.get('keystone', 'N/A')}"
                    f"{', KDA: ' + kda if kda else ''})"
                )

        if pro_parts:
            parts.append("Pro player builds (probuilds.net):\n" + "\n".join(pro_parts))

    return "\n\n".join(parts)

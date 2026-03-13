"""Async poller for the Riot Live Client Data API (https://127.0.0.1:2999)."""

from __future__ import annotations

import asyncio
import logging

import httpx

from ..config import RIOT_API_BASE, RIOT_POLL_INTERVAL
from ..models import GameState
from ..ws.manager import ws_manager

logger = logging.getLogger(__name__)

# Shared state — latest game state available to routers via import.
current_game_state: GameState | None = None


def _transform_riot_payload(raw: dict) -> dict:
    """Extract and reshape the Riot payload into our GameState schema."""
    active = raw.get("activePlayer", {})
    all_players = raw.get("allPlayers", [])
    active_name = active.get("summonerName", "")

    allies = []
    enemies = []
    my_team = None

    # First pass: find active player's team
    for p in all_players:
        if p.get("summonerName") == active_name:
            my_team = p.get("team")
            break

    # Second pass: split into allies/enemies
    for p in all_players:
        slim = {
            "summonerName": p.get("summonerName", ""),
            "championName": p.get("championName", ""),
            "team": p.get("team", ""),
            "position": p.get("position", ""),
            "isDead": p.get("isDead", False),
            "items": [
                {"itemID": it.get("itemID", 0), "displayName": it.get("displayName", "")}
                for it in p.get("items", [])
            ],
        }
        if p.get("team") == my_team:
            allies.append(slim)
        else:
            enemies.append(slim)

    return {
        "activePlayer": {
            "summonerName": active_name,
            "championName": active.get("championName", ""),
            "level": active.get("level", 1),
            "currentGold": active.get("currentGold", 0.0),
        },
        "allies": allies,
        "enemies": enemies,
        "gameTime": raw.get("gameData", {}).get("gameTime", 0.0),
        "gameMode": raw.get("gameData", {}).get("gameMode", "CLASSIC"),
    }


async def poll_riot_api() -> None:
    """Long-running coroutine: polls Riot API every RIOT_POLL_INTERVAL seconds."""
    global current_game_state

    async with httpx.AsyncClient(verify=False, timeout=3.0) as client:
        while True:
            try:
                resp = await client.get(f"{RIOT_API_BASE}/liveclientdata/allgamedata")
                resp.raise_for_status()
                transformed = _transform_riot_payload(resp.json())
                current_game_state = GameState(**transformed)
                await ws_manager.broadcast_json(current_game_state.model_dump(by_alias=True))
            except httpx.ConnectError:
                # Game not running — normal condition
                current_game_state = None
            except Exception as exc:
                logger.debug("Riot poller error: %s", exc)
                current_game_state = None

            await asyncio.sleep(RIOT_POLL_INTERVAL)

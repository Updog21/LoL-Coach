"""LCU poller — monitors champ select and broadcasts draft state over WebSocket."""

from __future__ import annotations

import asyncio
import logging

from ..config import LCU_POLL_INTERVAL
from ..models import ChampSelectAction, ChampSelectState, ChampSelectTeammate
from ..ws.manager import ws_manager
from .lcu import LCUConnection, find_lockfile, lcu_get

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Champion ID → name mapping (loaded from LCU on connect)
# ---------------------------------------------------------------------------
_champ_id_to_name: dict[int, str] = {}

# Shared state — current champ select state available to routers via import.
current_champ_select: ChampSelectState | None = None


async def _load_champion_map(conn: LCUConnection) -> None:
    """Load champion ID → name mapping from LCU."""
    global _champ_id_to_name
    if _champ_id_to_name:
        return

    data = await lcu_get(conn, "/lol-champions/v1/champions")
    if not data or not isinstance(data, list):
        return

    for champ in data:
        cid = champ.get("id", 0)
        name = champ.get("name", "")
        if cid and name and name != "None":
            _champ_id_to_name[cid] = name

    logger.info("Loaded %d champion names from LCU", len(_champ_id_to_name))


def _champ_name(champ_id: int) -> str:
    """Resolve champion ID to name."""
    if champ_id <= 0:
        return ""
    return _champ_id_to_name.get(champ_id, f"Champion#{champ_id}")


def _parse_champ_select(raw: dict) -> ChampSelectState:
    """Parse LCU champ select session into our model."""
    local_cell = raw.get("localPlayerCellId", 0)

    # Parse team rosters
    my_team = [ChampSelectTeammate(**t) for t in raw.get("myTeam", [])]
    their_team = [ChampSelectTeammate(**t) for t in raw.get("theirTeam", [])]

    # Parse actions (picks and bans)
    raw_actions = raw.get("actions", [])
    actions: list[list[ChampSelectAction]] = []
    for action_group in raw_actions:
        actions.append([ChampSelectAction(**a) for a in action_group])

    # Find my role and hover
    my_role = ""
    my_champion_id = 0
    for teammate in my_team:
        if teammate.cell_id == local_cell:
            my_role = _normalize_role(teammate.assigned_position)
            my_champion_id = teammate.champion_id
            break

    # Collect ally picks (completed picks from my team, excluding me)
    ally_picks: list[str] = []
    for teammate in my_team:
        if teammate.cell_id != local_cell and teammate.champion_id > 0:
            name = _champ_name(teammate.champion_id)
            if name:
                ally_picks.append(name)

    # Collect enemy picks (whatever is visible)
    enemy_picks: list[str] = []
    for enemy in their_team:
        if enemy.champion_id > 0:
            name = _champ_name(enemy.champion_id)
            if name:
                enemy_picks.append(name)

    # Collect bans from all actions
    banned: list[str] = []
    for action_group in actions:
        for action in action_group:
            if action.type == "ban" and action.completed and action.champion_id > 0:
                name = _champ_name(action.champion_id)
                if name:
                    banned.append(name)

    # Determine phase
    phase = raw.get("timer", {}).get("phase", "UNKNOWN")
    if not phase or phase == "UNKNOWN":
        # Infer from actions
        has_incomplete = any(
            not a.completed
            for group in actions
            for a in group
        )
        phase = "BAN_PICK" if has_incomplete else "FINALIZATION"

    return ChampSelectState(
        phase=phase,
        localPlayerCellId=local_cell,
        myTeam=my_team,
        theirTeam=their_team,
        actions=actions,
        my_role=my_role,
        my_champion_id=my_champion_id,
        my_champion_name=_champ_name(my_champion_id),
        ally_picks=ally_picks,
        enemy_picks=enemy_picks,
        banned=banned,
    )


def _normalize_role(position: str) -> str:
    """Normalize LCU position strings to our Role enum values."""
    mapping = {
        "top": "TOP",
        "jungle": "JUNGLE",
        "middle": "MID",
        "bottom": "BOT",
        "utility": "SUPPORT",
        "support": "SUPPORT",
        "mid": "MID",
        "adc": "BOT",
    }
    return mapping.get(position.lower(), position.upper())


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------
async def poll_lcu() -> None:
    """Long-running coroutine: discovers LCU, polls champ select session."""
    global current_champ_select
    conn: LCUConnection | None = None
    was_in_select = False

    while True:
        # Try to find LCU if we don't have a connection
        if conn is None:
            conn = find_lockfile()
            if conn:
                await _load_champion_map(conn)

        if conn is not None:
            try:
                session = await lcu_get(conn, "/lol-champ-select/v1/session")

                if session and isinstance(session, dict):
                    state = _parse_champ_select(session)
                    current_champ_select = state

                    # Broadcast champ select state to frontend
                    await ws_manager.broadcast_json({
                        "type": "champ_select",
                        "phase": state.phase,
                        "myRole": state.my_role,
                        "myChampion": state.my_champion_name,
                        "myChampionId": state.my_champion_id,
                        "allyPicks": state.ally_picks,
                        "enemyPicks": state.enemy_picks,
                        "banned": state.banned,
                    })
                    was_in_select = True
                else:
                    # Not in champ select
                    if was_in_select:
                        # Just left champ select — notify frontend
                        await ws_manager.broadcast_json({"type": "champ_select_end"})
                        was_in_select = False
                    current_champ_select = None

            except Exception as exc:
                # LCU connection lost (client closed, port changed, etc.)
                logger.debug("LCU poll error: %s", exc)
                conn = None
                current_champ_select = None
                if was_in_select:
                    await ws_manager.broadcast_json({"type": "champ_select_end"})
                    was_in_select = False

        await asyncio.sleep(LCU_POLL_INTERVAL)

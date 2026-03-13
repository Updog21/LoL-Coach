"""Game advisor — detects meaningful state changes and triggers build updates.

Sits between the riot poller (raw state every 5s) and the AI router. Compares
consecutive game states to decide WHEN a new build recommendation should fire,
then pushes it over WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
import time

from sqlmodel import Session

from ..config import BUILD_UPDATE_COOLDOWN, EARLY_GAME_END, MID_GAME_END
from ..db.database import engine
from ..db.models_db import Setting
from ..models import GameState
from ..ws.manager import ws_manager
from .ai_router import query_build
from .meta_scraper import get_meta_context
from .rag import retrieve_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot of what we last sent a recommendation for
# ---------------------------------------------------------------------------
class _GameSnapshot:
    """Captures the aspects of game state we care about for change detection."""

    def __init__(self, state: GameState, role: str) -> None:
        self.game_time = state.game_time
        self.my_champion = state.active_player.champion_name
        self.my_level = state.active_player.level
        self.my_gold = state.active_player.current_gold
        self.phase = _game_phase(state.game_time)

        # Track completed items per player (ignore components / consumables
        # by only counting items that appear as "completed" — we approximate
        # by tracking the full item list length, since Riot API doesn't flag
        # completion. A new item name appearing is the real signal.)
        self.my_items: set[str] = set()
        self.lane_opponent_items: set[str] = set()
        self.enemy_items: dict[str, set[str]] = {}
        self.fed_enemy_threats: dict[str, int] = {}
        self.fed_enemy_summaries: dict[str, str] = {}
        self.priority_enemy = ""
        self.priority_enemy_summary = ""

        from .ai_router import _identify_lane_opponent

        active_ally = None
        for ally in state.allies:
            if ally.summoner_name == state.active_player.summoner_name:
                active_ally = ally
                self.my_items = {i.name for i in ally.items if i.name}
                break

        self.lane_opponent = _identify_lane_opponent(role, state.enemies)
        if self.lane_opponent:
            self.lane_opponent_items = {i.name for i in self.lane_opponent.items if i.name}
            self.lane_opponent_name = self.lane_opponent.champion_name
        else:
            self.lane_opponent_name = ""

        for enemy in state.enemies:
            self.enemy_items[enemy.champion_name] = {i.name for i in enemy.items if i.name}
            counterpart = _find_role_counterpart(enemy.position, state.allies)
            threat_score, summary = _score_enemy_threat(
                enemy=enemy,
                counterpart=counterpart,
                game_time=state.game_time,
                my_level=active_ally.level if active_ally else state.active_player.level,
            )
            if _should_consider_enemy_threat(
                my_role=role,
                enemy_role=_normalize_role(enemy.position),
                game_time=state.game_time,
                threat_score=threat_score,
            ):
                self.fed_enemy_threats[enemy.champion_name] = threat_score
                self.fed_enemy_summaries[enemy.champion_name] = summary

        if self.fed_enemy_threats:
            self.priority_enemy = max(
                self.fed_enemy_threats,
                key=lambda champ: self.fed_enemy_threats[champ],
            )
            self.priority_enemy_summary = self.fed_enemy_summaries[self.priority_enemy]


def _game_phase(game_time: float) -> str:
    if game_time < EARLY_GAME_END:
        return "early"
    if game_time < MID_GAME_END:
        return "mid"
    return "late"


_ROLE_ALIASES = {
    "top": "TOP",
    "jungle": "JUNGLE",
    "middle": "MID",
    "mid": "MID",
    "bottom": "BOT",
    "bot": "BOT",
    "utility": "SUPPORT",
    "support": "SUPPORT",
}
_FED_THREAT_THRESHOLD = 4


def _normalize_role(position: str) -> str:
    return _ROLE_ALIASES.get(position.lower(), position.upper())


def _find_role_counterpart(position: str, allies: list) -> object | None:
    """Find the ally in the same role as the enemy threat."""
    expected_role = _normalize_role(position)
    for ally in allies:
        if _normalize_role(ally.position) == expected_role:
            return ally
    return None


def _score_enemy_threat(enemy, counterpart, game_time: float, my_level: int) -> tuple[int, str]:
    """Estimate how urgently the build should adapt to this enemy snowballing."""
    kills = enemy.scores.kills
    deaths = enemy.scores.deaths
    assists = enemy.scores.assists
    creep_score = enemy.scores.creep_score

    score = 0
    notes: list[str] = []

    if kills >= 4:
        score += 2
        notes.append(f"{kills}/{deaths}/{assists} scoreline")
    if kills - deaths >= 3:
        score += 2
    if enemy.level - my_level >= 2:
        score += 1
        notes.append(f"{enemy.level - my_level} levels up on you")

    if counterpart is not None:
        kill_gap = kills - counterpart.scores.kills
        death_gap = counterpart.scores.deaths - deaths
        level_gap = enemy.level - counterpart.level
        cs_gap = creep_score - counterpart.scores.creep_score
        item_gap = len(enemy.items) - len(counterpart.items)

        if kill_gap >= 3:
            score += 2
        if death_gap >= 3:
            score += 1
        if level_gap >= 2:
            score += 2
            notes.append(f"{level_gap} levels ahead of enemy {counterpart.position.lower()}")
        if game_time >= 600 and cs_gap >= 25:
            score += 1
            notes.append(f"{cs_gap} CS lead")
        if item_gap >= 1:
            score += 1
    else:
        if kills >= 6:
            score += 1
        if game_time >= 900 and creep_score >= 120:
            score += 1

    summary_parts = [enemy.champion_name]
    if notes:
        summary_parts.append(", ".join(notes))
    else:
        summary_parts.append(f"{kills}/{deaths}/{assists} with {creep_score} CS")

    return score, " - ".join(summary_parts)


def _should_consider_enemy_threat(
    my_role: str, enemy_role: str, game_time: float, threat_score: int
) -> bool:
    """Gate fed-threat reactions based on role-specific itemization best practice."""
    my_role = my_role.upper()
    enemy_role = enemy_role.upper()

    # Same-role opponents and junglers are core adaptation targets for every role.
    if enemy_role in {my_role, "JUNGLE"}:
        return threat_score >= _FED_THREAT_THRESHOLD

    # Supports should react earlier to threats that can reach or burst carries.
    if my_role == "SUPPORT":
        if enemy_role in {"BOT", "MID"}:
            return threat_score >= _FED_THREAT_THRESHOLD
        return game_time >= EARLY_GAME_END and threat_score >= _FED_THREAT_THRESHOLD + 1

    # ADCs/BOT carries care early about lane/jungle and then other diving carries.
    if my_role == "BOT":
        if enemy_role in {"SUPPORT", "MID"}:
            return game_time >= EARLY_GAME_END and threat_score >= _FED_THREAT_THRESHOLD + 1
        return game_time >= EARLY_GAME_END and threat_score >= _FED_THREAT_THRESHOLD + 2

    # Mids adapt readily to jungle/mid pressure, then to side-lane threats after lane.
    if my_role == "MID":
        if enemy_role == "BOT":
            return game_time >= EARLY_GAME_END and threat_score >= _FED_THREAT_THRESHOLD + 1
        return game_time >= EARLY_GAME_END and threat_score >= _FED_THREAT_THRESHOLD + 2

    # Tops usually stay role-first unless the map threat is clearly warping fights.
    if my_role == "TOP":
        return game_time >= EARLY_GAME_END and threat_score >= _FED_THREAT_THRESHOLD + 2

    # Junglers need broad map awareness, but off-role adaptations still need clear evidence.
    if my_role == "JUNGLE":
        if enemy_role in {"MID", "BOT"}:
            return threat_score >= _FED_THREAT_THRESHOLD
        return game_time >= EARLY_GAME_END and threat_score >= _FED_THREAT_THRESHOLD + 1

    return threat_score >= _FED_THREAT_THRESHOLD + 2


# ---------------------------------------------------------------------------
# Change detection — returns a list of trigger reasons (empty = no update)
# ---------------------------------------------------------------------------
def detect_changes(prev: _GameSnapshot | None, curr: _GameSnapshot) -> list[str]:
    """Compare two snapshots and return reasons a build update should fire."""
    if prev is None:
        return ["game_start"]

    reasons: list[str] = []

    # 1. Game phase transition (early → mid → late)
    if curr.phase != prev.phase:
        reasons.append(f"phase_change:{prev.phase}>{curr.phase}")

    # 2. Lane opponent bought a new completed item
    if curr.lane_opponent_name:
        new_opp_items = curr.lane_opponent_items - prev.lane_opponent_items
        # Filter out consumables and wards
        significant = {i for i in new_opp_items if not _is_consumable(i)}
        if significant:
            reasons.append(f"lane_opponent_item:{','.join(significant)}")

    # 3. Any enemy bought a new completed item (check all enemies)
    for champ, items in curr.enemy_items.items():
        prev_items = prev.enemy_items.get(champ, set())
        new_items = items - prev_items
        significant = {i for i in new_items if not _is_consumable(i)}
        if significant and champ != curr.lane_opponent_name:
            reasons.append(f"enemy_item:{champ}:{','.join(significant)}")

    # 4. Player completed a new item (needs next-item guidance)
    new_my_items = curr.my_items - prev.my_items
    significant_my = {i for i in new_my_items if not _is_consumable(i)}
    if significant_my:
        reasons.append(f"player_item:{','.join(significant_my)}")

    # 5. Significant level milestone (6, 11, 16 — ult upgrades)
    for milestone in (6, 11, 16):
        if prev.my_level < milestone <= curr.my_level:
            reasons.append(f"level_spike:{milestone}")

    # 6. A previously manageable enemy has become fed enough to demand itemization changes
    for champ, score in curr.fed_enemy_threats.items():
        prev_score = prev.fed_enemy_threats.get(champ, 0)
        if prev_score < _FED_THREAT_THRESHOLD <= score:
            reasons.append(f"fed_enemy:{champ}")
        elif prev_score >= _FED_THREAT_THRESHOLD and score - prev_score >= 2:
            reasons.append(f"fed_enemy_spike:{champ}")

    # 7. The highest-priority enemy threat changed
    if (
        curr.priority_enemy
        and curr.priority_enemy != prev.priority_enemy
        and curr.priority_enemy in curr.fed_enemy_threats
    ):
        reasons.append(f"priority_threat:{curr.priority_enemy}")

    return reasons


# Common consumable / ward item names to ignore
_CONSUMABLES = {
    "Health Potion", "Refillable Potion", "Corrupting Potion",
    "Control Ward", "Stealth Ward", "Oracle Lens", "Farsight Alteration",
    "Elixir of Iron", "Elixir of Sorcery", "Elixir of Wrath",
    "Total Biscuit of Everlasting Will",
}


def _is_consumable(item_name: str) -> bool:
    return item_name in _CONSUMABLES


# ---------------------------------------------------------------------------
# The advisor loop — runs alongside the riot poller
# ---------------------------------------------------------------------------
# Shared state so the frontend can read the latest recommendation
latest_build_response: dict | None = None

# The role the player is playing — set once at game start from position data
_detected_role: str = ""


def _detect_my_role(state: GameState) -> str:
    """Figure out the player's role from ally position data."""
    for ally in state.allies:
        if ally.summoner_name == state.active_player.summoner_name:
            pos = ally.position.lower()
            role_map = {
                "top": "TOP", "jungle": "JUNGLE",
                "middle": "MID", "mid": "MID",
                "bottom": "BOT", "bot": "BOT",
                "utility": "SUPPORT", "support": "SUPPORT",
            }
            return role_map.get(pos, pos.upper())
    return ""


def _get_provider() -> str:
    with Session(engine) as session:
        row = session.get(Setting, "provider")
        return row.value if row else "openai"


async def run_advisor(get_state) -> None:
    """Long-running coroutine that watches game state and pushes build updates.

    Args:
        get_state: callable returning the current GameState or None
    """
    global latest_build_response, _detected_role

    last_sent_snapshot: _GameSnapshot | None = None
    last_update_time: float = 0
    game_active = False

    while True:
        state: GameState | None = get_state()

        if state is None:
            # No game running — reset
            if game_active:
                last_sent_snapshot = None
                latest_build_response = None
                _detected_role = ""
                last_update_time = 0
                game_active = False
                await ws_manager.broadcast_json({"type": "build_update", "active": False})
            await asyncio.sleep(3)
            continue

        game_active = True

        # Detect role once at game start
        if not _detected_role:
            _detected_role = _detect_my_role(state)
            if not _detected_role:
                await asyncio.sleep(5)
                continue
            logger.info("Detected player role: %s", _detected_role)

        # Build current snapshot
        curr_snapshot = _GameSnapshot(state, _detected_role)

        # Compare against the last snapshot we actually sent to avoid
        # dropping changes that happen during the cooldown window.
        reasons = detect_changes(last_sent_snapshot, curr_snapshot)

        now = time.monotonic()
        cooldown_ok = (now - last_update_time) >= BUILD_UPDATE_COOLDOWN

        if reasons and cooldown_ok:
            logger.info("Build update triggered: %s", ", ".join(reasons))

            try:
                provider = _get_provider()
                champion = state.active_player.champion_name
                meta_ctx = get_meta_context(champion, _detected_role)
                rag_ctx = await retrieve_context(champion, "build recommendation", provider=provider)

                # Craft a question that includes the trigger reasons
                question = _build_question(reasons, curr_snapshot)

                result = await query_build(
                    provider=provider,
                    question=question,
                    champion=champion,
                    role=_detected_role,
                    game_state=state,
                    meta_context=meta_ctx,
                    rag_context=rag_ctx,
                )

                response_data = {
                    "type": "build_update",
                    "active": True,
                    "triggers": reasons,
                    "phase": curr_snapshot.phase,
                    "build": result.model_dump(by_alias=True),
                }
                latest_build_response = response_data
                await ws_manager.broadcast_json(response_data)
                last_update_time = now
                last_sent_snapshot = curr_snapshot

            except Exception as exc:
                logger.error("Build advisor update failed: %s", exc)

        await asyncio.sleep(RIOT_POLL_INTERVAL)


# Re-import here to avoid circular at module level
from ..config import RIOT_POLL_INTERVAL


def _build_question(reasons: list[str], snap: _GameSnapshot) -> str:
    """Turn trigger reasons into a natural question for the AI."""
    parts: list[str] = []

    for reason in reasons:
        if reason == "game_start":
            parts.append("Game just started. Give me my opening build path and laning strategy.")

        elif reason.startswith("phase_change:"):
            phase = reason.split(">")[1]
            if phase == "mid":
                parts.append(
                    "We're entering mid game. Should I adjust my build for "
                    "teamfights and rotations?"
                )
            elif phase == "late":
                parts.append(
                    "Late game now. What's my optimal final build for closing "
                    "out or defending?"
                )

        elif reason.startswith("lane_opponent_item:"):
            items = reason.split(":")[1]
            parts.append(
                f"My lane opponent just bought {items}. "
                f"Do I need to adjust my build to counter this?"
            )

        elif reason.startswith("enemy_item:"):
            _, champ, items = reason.split(":", 2)
            parts.append(f"Enemy {champ} just completed {items}.")

        elif reason.startswith("player_item:"):
            items = reason.split(":")[1]
            parts.append(
                f"I just completed {items}. What should I build next and why?"
            )

        elif reason.startswith("level_spike:"):
            level = reason.split(":")[1]
            parts.append(f"I just hit level {level}. Any build or playstyle changes?")

        elif reason.startswith("fed_enemy_spike:"):
            champ = reason.split(":")[1]
            summary = snap.fed_enemy_summaries.get(champ, champ)
            parts.append(
                f"{summary} is snowballing even harder. Re-evaluate my build only if "
                f"best practice for my role says I should adapt to that threat now."
            )

        elif reason.startswith("fed_enemy:") or reason.startswith("priority_threat:"):
            champ = reason.split(":")[1]
            summary = snap.fed_enemy_summaries.get(champ, champ)
            parts.append(
                f"{summary} has become a fed enemy threat. Adjust my build only if that "
                f"is role-correct best practice instead of blindly abandoning my standard path."
            )

    return " ".join(parts) if parts else "Update my build recommendation for the current game state."

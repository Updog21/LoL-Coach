"""Multi-provider LLM routing via LangChain."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..config import OLLAMA_BASE_URL
from ..models import BuildResponse, GameState, ProviderName
from ..security import load_api_key
from .meta_scraper import get_ability_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider → LangChain model factory
# ---------------------------------------------------------------------------

def _get_chat_model(provider: str, vision: bool = False):
    """Return a LangChain chat model for the given provider."""
    key = load_api_key(provider)

    if provider == ProviderName.OPENAI:
        from langchain_openai import ChatOpenAI

        model = "gpt-4o" if vision else "gpt-4o"
        return ChatOpenAI(model=model, api_key=key, temperature=0.3)

    if provider == ProviderName.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = "gemini-2.0-flash"
        return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0.3)

    if provider == ProviderName.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        model = "claude-sonnet-4-6"
        return ChatAnthropic(model=model, api_key=key, temperature=0.3)

    if provider == ProviderName.OLLAMA:
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(model="llama3", base_url=OLLAMA_BASE_URL, temperature=0.3)

    raise ValueError(f"Unsupported provider: {provider}")


def _get_embedding_model(provider: str):
    """Return an embedding model for RAG ingestion/retrieval."""
    key = load_api_key(provider)

    if provider == ProviderName.OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-large", api_key=key)

    if provider == ProviderName.GEMINI:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=key
        )

    if provider == ProviderName.ANTHROPIC:
        # Anthropic doesn't have embeddings — fall back to a local model via ChromaDB default
        return None

    if provider == ProviderName.OLLAMA:
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)

    return None


# ---------------------------------------------------------------------------
# Lane opponent detection
# ---------------------------------------------------------------------------
# Maps our role names to Riot "position" strings used in the Live Client API.
_ROLE_TO_POSITIONS = {
    "TOP": {"top", "TOP"},
    "JUNGLE": {"jungle", "JUNGLE"},
    "MID": {"mid", "middle", "MID", "MIDDLE"},
    "BOT": {"bot", "bottom", "BOT", "BOTTOM"},
    "SUPPORT": {"support", "utility", "SUPPORT", "UTILITY"},
}


def _identify_lane_opponent(
    role: str, enemies: list,
) -> "PlayerSlim | None":
    """Find the enemy most likely in the same lane based on role/position."""
    expected_positions = _ROLE_TO_POSITIONS.get(role.upper(), set())
    for enemy in enemies:
        if enemy.position.lower() in {p.lower() for p in expected_positions}:
            return enemy
    return None


def _identify_lane_partner(
    role: str, allies: list, active_player_name: str,
) -> "PlayerSlim | None":
    """Find the ally sharing your lane (BOT ↔ SUPPORT)."""
    partner_role = _BOT_LANE_PARTNER.get(role.upper())
    if not partner_role:
        return None
    partner_positions = _ROLE_TO_POSITIONS.get(partner_role, set())
    for ally in allies:
        if ally.summoner_name == active_player_name:
            continue
        if ally.position.lower() in {p.lower() for p in partner_positions}:
            return ally
    return None


def _identify_teammate_by_role(
    role: str, allies: list, active_player_name: str,
) -> "PlayerSlim | None":
    """Find an ally in the requested role, excluding the active player."""
    expected_positions = _ROLE_TO_POSITIONS.get(role.upper(), set())
    for ally in allies:
        if ally.summoner_name == active_player_name:
            continue
        if ally.position.lower() in {p.lower() for p in expected_positions}:
            return ally
    return None


# BOT and SUPPORT share a lane
_BOT_LANE_PARTNER = {
    "BOT": "SUPPORT",
    "SUPPORT": "BOT",
}


def _format_player_detail(player) -> str:
    """Format a single player's champion, items, and runes for the prompt."""
    items_str = ", ".join(i.name for i in player.items) if player.items else "no items"
    runes_str = ""
    if player.runes and player.runes.keystone:
        runes_str = (
            f" | Runes: {player.runes.keystone}"
            f" ({player.runes.primary_tree}/{player.runes.secondary_tree})"
        )
    score_str = ""
    if getattr(player, "scores", None):
        score_str = (
            f" | Score: {player.scores.kills}/{player.scores.deaths}/{player.scores.assists}"
            f", CS {player.scores.creep_score}"
        )
    level_str = f" | Level {player.level}" if getattr(player, "level", 0) else ""
    return f"{player.champion_name}{level_str}{score_str} — Items: [{items_str}]{runes_str}"


def _summarize_fed_threats(game_state: GameState) -> list[str]:
    """Summarize enemy champions who are materially ahead."""
    summaries: list[str] = []
    for enemy in game_state.enemies:
        counterpart = None
        for ally in game_state.allies:
            if ally.position.lower() == enemy.position.lower():
                counterpart = ally
                break

        lead_notes: list[str] = []
        if enemy.scores.kills >= 4 and enemy.scores.kills - enemy.scores.deaths >= 2:
            lead_notes.append(f"{enemy.scores.kills}/{enemy.scores.deaths}/{enemy.scores.assists}")
        if counterpart:
            if enemy.level - counterpart.level >= 2:
                lead_notes.append(f"+{enemy.level - counterpart.level} levels on {counterpart.champion_name}")
            if enemy.scores.creep_score - counterpart.scores.creep_score >= 25:
                lead_notes.append(
                    f"+{enemy.scores.creep_score - counterpart.scores.creep_score} CS on {counterpart.champion_name}"
                )
            if len(enemy.items) > len(counterpart.items):
                lead_notes.append(f"+{len(enemy.items) - len(counterpart.items)} item slots")

        if lead_notes:
            summaries.append(f"{enemy.champion_name}: " + ", ".join(lead_notes))
    return summaries


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are LoL Build Coach, an expert League of Legends item, rune, laning, and matchup advisor.

You will receive detailed information about the player's lane opponent (champion, \
runes, items), their lane partner (if bot lane), the full enemy team, and grounded \
champion-kit summaries from Riot Data Dragon.

Your FIRST priority is the lane matchup — build to win lane, THEN consider the \
broader team composition.

Priority order for item recommendations:
1. LANE OPPONENT — their champion, keystone rune, and items dictate your early/core \
   build. For example, if the enemy laner runs Conqueror on a bruiser, you may need \
   anti-sustain; if they run Electrocute on an assassin, you may need early defense.
2. ENEMY TEAM COMPOSITION — secondary factor. Check for heavy AP/AD, CC-heavy comps, \
   or specific threats from other lanes that require late-game itemization.
3. META / CUSTOM RESOURCES — use as a baseline, but override with matchup-specific \
   choices when needed.
4. ABILITY INTERACTIONS — explain how the player's kit functions into the current \
   enemy champions. Call out which enemy tools can block, dodge, out-range, cleanse, \
   or punish the player's key abilities.
5. JUNGLE PRESSURE — explicitly account for the enemy jungler's gank pattern, crowd \
   control, burst, pathing threat, and objective/skirmish impact. If relevant, also \
   explain how the allied jungler changes your windows to fight or set up plays.
6. FED THREATS — if an enemy champion is snowballing, you must adapt the build path to \
   respect that lead only when role-correct best practice says the threat materially \
   changes your job, survivability needs, or fight pattern. Do not overreact just \
   because someone is fed in another lane.

Rules:
1. Recommend exactly 6 items (full build path for current game state).
2. For each item, explain WHY it's good against the specific enemies in THIS game.
3. The first 1-2 items MUST directly address the lane opponent matchup.
4. If custom resource context is provided, prefer its advice over generic meta \
   when they conflict — cite the source.
5. Always include rune recommendations tailored to the lane matchup.
6. Be concise and actionable.
7. ALWAYS include a "lanePreview" section describing what to expect early in lane.
8. In both "rationale" and "lanePreview", explicitly mention the player's key ability \
   patterns, which enemies they are weak into, and what enemy abilities must be \
   tracked before committing.
9. In "threats" and "tips", explicitly call out the enemy jungler if their kit makes \
   lane positioning, ward timing, or wave state dangerous.
10. Treat pro/meta builds as the default baseline for this role, but override them when \
    a fed enemy or specific matchup interaction demands defensive, anti-burst, anti-heal, \
    anti-CC, or anti-tank adaptation.
11. If you choose NOT to adapt to a fed off-role threat, say that clearly and keep the \
    recommended item path aligned with normal best practice for this role.

Lane preview guidelines — be DIRECT and OPINIONATED. Tell the player exactly what \
to do and what NOT to do. Never be vague.
- "summary": Start with a clear verdict: "You lose this lane early — play safe and \
  farm until X" or "You hard-win levels 1-3, look for early kills" or "Skill matchup, \
  whoever lands X first wins trades." Include WHO has lane priority and WHY \
  (their champion's base stats, abilities, or rune choice makes them stronger/weaker \
  early). Call out if the enemy's rune choice changes the matchup (e.g. "They took \
  Fleet Footwork instead of Conqueror — they want to sustain, not fight, so you can \
  trade aggressively early").
- "powerSpikes": Be specific with levels and items. Say "You are weaker levels 1-5 \
  but spike hard at level 6 with ult — avoid extended trades before then" or "Their \
  champion hits a huge spike at BF Sword purchase — if you don't have a kill or CS \
  lead by then, concede and farm safely." Call out the FIRST major item for both sides.
- "threats": Name their exact kill combo (e.g. "Renekton dashes in, stuns with W, Qs \
  for empowered damage — if you eat this at 50% HP you die"). Call out gank setup \
  (e.g. "Their CC makes ganks lethal for you — ward river at 2:45"). Warn about \
  cheese strats (e.g. "Darius may try to all-in level 1 with ghost — do NOT fight \
  in the minion wave"). Also name enemy abilities that invalidate your engage/trade \
  pattern, such as spell shields, parries, untargetability, displacement, or cleanse.
- "tips": Very specific dos and don'ts. Say things like "Stay behind your caster \
  minions to block their Q", "Trade only after they use X ability (Y second \
  cooldown)", "Freeze the wave near your tower — they have no escape if jungler \
  ganks", "Do NOT push the wave — you lose all-ins and become gank-vulnerable." \
  Include spacing distances when relevant (e.g. "Stay at max auto range to avoid \
  their E engage"). Explicitly say when your own crowd control / burst combo is good, \
  and which enemy champions can deny it.
- "lanePartnerSynergy" (BOT/SUPPORT lanes ONLY): Describe how the duo should play \
  the 2v2 together. Example: "Your Leona can engage at level 2 — push the first wave \
  hard to hit 2 first, then all-in when she lands E" or "Your Yuumi offers no early \
  pressure — play safe, farm, and avoid 2v2 fights until you have BF Sword. The enemy \
  Draven/Thresh WILL look for early kills — respect their level 2." If the partner is \
  not provided, omit this field.

Respond in valid JSON matching this schema:
{
  "items": [{"name": "...", "cost": 0, "reason": "..."}],
  "runes": {"keystone": "...", "secondary": "...", "shards": "..."},
  "skillOrder": "Q > W > E",
  "rationale": "...",
  "lanePreview": {
    "summary": "...",
    "powerSpikes": "...",
    "threats": "...",
    "tips": "...",
    "lanePartnerSynergy": "... (only for BOT/SUPPORT)"
  }
}
"""


# ---------------------------------------------------------------------------
# Core query function
# ---------------------------------------------------------------------------
async def query_build(
    provider: str,
    question: str,
    champion: str,
    role: str,
    game_state: GameState | None = None,
    meta_context: str = "",
    rag_context: str = "",
    screenshot_b64: str | None = None,
) -> BuildResponse:
    """Send a build query to the configured AI provider and parse the response."""
    llm = _get_chat_model(provider, vision=screenshot_b64 is not None)

    # Build the user message
    parts: list[str] = []
    parts.append(f"Champion: {champion} | Role: {role}")
    parts.append(f"Question: {question}")

    # ----- Lane opponent (highest priority) -----
    is_bot_lane = role.upper() in ("BOT", "SUPPORT")

    if game_state:
        lane_opponent = _identify_lane_opponent(role, game_state.enemies)
        lane_partner = _identify_lane_partner(
            role, game_state.allies, game_state.active_player.summoner_name,
        )
        enemy_jungler = _identify_lane_opponent("JUNGLE", game_state.enemies)
        ally_jungler = _identify_teammate_by_role(
            "JUNGLE", game_state.allies, game_state.active_player.summoner_name,
        )

        if lane_opponent:
            parts.append(
                f"\n=== YOUR LANE OPPONENT (PRIMARY THREAT) ===\n"
                f"{_format_player_detail(lane_opponent)}\n"
                f"=== BUILD YOUR FIRST ITEMS TO BEAT THIS MATCHUP ==="
            )

        # ----- Bot lane: show the full 2v2 matchup -----
        if is_bot_lane:
            # Enemy bot duo
            enemy_bot = _identify_lane_opponent("BOT", game_state.enemies)
            enemy_sup = _identify_lane_opponent("SUPPORT", game_state.enemies)
            enemy_duo = [p for p in [enemy_bot, enemy_sup] if p is not None]
            if enemy_duo:
                parts.append("\n--- Enemy Bot Lane (2v2 matchup) ---")
                for p in enemy_duo:
                    parts.append(f"  {_format_player_detail(p)}")

            # Your lane partner
            if lane_partner:
                parts.append(
                    f"\n--- YOUR LANE PARTNER ---\n"
                    f"  {_format_player_detail(lane_partner)}\n"
                    f"  (Consider your combined kill pressure, engage patterns, "
                    f"and how your kits synergize)"
                )

        # ----- Full enemy team (secondary) -----
        parts.append("\n--- Enemy Team ---")
        for enemy in game_state.enemies:
            marker = " ← YOUR LANE" if (lane_opponent and enemy.champion_name == lane_opponent.champion_name) else ""
            parts.append(f"  {_format_player_detail(enemy)}{marker}")

        if enemy_jungler:
            parts.append(
                "\n--- Enemy Jungler (gank / objective threat) ---\n"
                f"  {_format_player_detail(enemy_jungler)}\n"
                "  Explain how this jungler's engage, CC, burst, or terrain access "
                "changes ward timing, wave management, and all-in windows."
            )

        # ----- Ally team (for comp context) -----
        parts.append("\n--- Your Team ---")
        for ally in game_state.allies:
            marker = " ← LANE PARTNER" if (lane_partner and ally.champion_name == lane_partner.champion_name) else ""
            parts.append(f"  {_format_player_detail(ally)}{marker}")

        if ally_jungler:
            parts.append(
                "\n--- Your Jungler (setup / follow-up) ---\n"
                f"  {_format_player_detail(ally_jungler)}\n"
                "  Explain whether your abilities and wave state can set up this "
                "jungler's ganks or objective fights."
            )

        fed_threats = _summarize_fed_threats(game_state)
        if fed_threats:
            parts.append(
                "\n--- Fed Enemy Threats ---\n"
                + "\n".join(f"  {summary}" for summary in fed_threats)
                + "\nAdjust the build path if these leads require earlier armor, MR, "
                  "anti-heal, burst protection, or anti-tank tools, but only when that "
                  "adaptation is actually best practice for this role."
            )

        parts.append(f"\nGame time: {game_state.game_time:.0f}s")

        try:
            ability_ctx = await get_ability_context(
                champion=champion,
                role=role,
                enemy_champions=[enemy.champion_name for enemy in game_state.enemies],
                lane_opponent=lane_opponent.champion_name if lane_opponent else "",
                lane_partner=lane_partner.champion_name if lane_partner else "",
                enemy_jungler=enemy_jungler.champion_name if enemy_jungler else "",
                ally_jungler=ally_jungler.champion_name if ally_jungler else "",
            )
        except Exception as exc:
            logger.debug("Failed to load ability context: %s", exc)
            ability_ctx = ""

        if ability_ctx:
            parts.append(
                "\n--- Champion Ability Reference (Riot Data Dragon) ---\n"
                f"{ability_ctx}\n"
                "Use this to explain: your key combo windows, enemies that can deny "
                "your abilities, and the specific enemy spells you must watch before "
                "fighting."
            )

    if meta_context:
        parts.append(f"\n--- Current Meta Data ---\n{meta_context}")

    if rag_context:
        parts.append(f"\n--- Custom Resources (HIGH PRIORITY) ---\n{rag_context}")

    user_msg = "\n".join(parts)

    # Build messages for the LLM
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if screenshot_b64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_msg},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                },
            ],
        })
    else:
        messages.append({"role": "user", "content": user_msg})

    response = await llm.ainvoke(messages)
    content = response.content

    # Parse JSON from response (handle markdown code blocks)
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    parsed = json.loads(content.strip())
    return BuildResponse(**parsed)

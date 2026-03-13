"""Multi-provider LLM routing via LangChain."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..config import OLLAMA_BASE_URL
from ..models import BuildResponse, GameState, ProviderName
from ..security import load_api_key

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
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are LoL Build Coach, an expert League of Legends item and rune advisor.

Given the player's champion, role, live enemy team composition (champions + items),
current meta data, and any custom resource context, provide a build recommendation.

Rules:
1. Recommend exactly 6 items (full build path for current game state).
2. For each item, explain WHY it's good in this specific game.
3. If custom resource context is provided, prefer its advice over generic meta
   when they conflict — cite the source.
4. Consider enemy damage types, CC, and their items when recommending.
5. Include rune recommendations if asked or if they differ from standard.
6. Be concise and actionable.

Respond in valid JSON matching this schema:
{
  "items": [{"name": "...", "cost": 0, "reason": "..."}],
  "runes": {"keystone": "...", "secondary": "...", "shards": "..."},
  "skillOrder": "Q > W > E",
  "rationale": "..."
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

    if game_state:
        enemies_str = ", ".join(
            f"{e.champion_name} ({', '.join(i.name for i in e.items) or 'no items'})"
            for e in game_state.enemies
        )
        parts.append(f"Enemy team: {enemies_str}")
        parts.append(f"Game time: {game_state.game_time:.0f}s")

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

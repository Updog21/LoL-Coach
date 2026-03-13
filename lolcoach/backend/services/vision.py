"""Screenshot → enemy data extraction via vision-capable AI models."""

from __future__ import annotations

import json
import logging

from .ai_router import _get_chat_model

logger = logging.getLogger(__name__)

VISION_PROMPT = """\
Analyze this League of Legends screenshot and extract enemy team information.

Return a JSON object with:
{
  "enemies": [
    {
      "championName": "Champion Name",
      "items": [{"displayName": "Item Name", "itemID": 0}],
      "estimatedGold": 0
    }
  ]
}

Only include enemies visible in the scoreboard/tab screen. If you cannot identify
an item, use "Unknown" as the name. If no enemy data is visible, return {"enemies": []}.
"""


async def extract_enemies_from_screenshot(
    screenshot_b64: str,
    provider: str,
) -> list[dict]:
    """Use a vision model to extract enemy champion/item data from a screenshot."""
    llm = _get_chat_model(provider, vision=True)

    messages = [
        {"role": "system", "content": "You are a League of Legends game state analyzer."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                },
            ],
        },
    ]

    response = await llm.ainvoke(messages)
    content = response.content

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    parsed = json.loads(content.strip())
    return parsed.get("enemies", [])

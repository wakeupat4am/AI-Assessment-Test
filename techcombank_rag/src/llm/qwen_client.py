from __future__ import annotations

from src.config import Settings
from src.llm.provider import OpenAIChatClient


class QwenClient(OpenAIChatClient):
    """Backward-compatible name for the original local Qwen baseline client."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)


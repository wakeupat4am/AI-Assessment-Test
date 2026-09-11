from __future__ import annotations

import unicodedata
from typing import Any


def normalize_utf8_text(value: Any) -> str:
    """Return NFC text that is always safe to serialize as UTF-8.

    Interactive shells can decode invalid input bytes with ``surrogateescape``.
    Such strings print back to the same terminal but fail later when an HTTP
    client serializes them as JSON. Preserve valid Unicode and replace only the
    undecodable bytes/code points instead of crashing the conversation.
    """
    text = value if isinstance(value, str) else str(value)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        try:
            raw = text.encode("utf-8", errors="surrogateescape")
            text = raw.decode("utf-8", errors="replace")
        except UnicodeEncodeError:
            text = text.encode("utf-8", errors="replace").decode("utf-8")
    return unicodedata.normalize("NFC", text)


def normalize_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "role": normalize_utf8_text(turn.get("role", "")),
            "content": normalize_utf8_text(turn.get("content", "")),
        }
        for turn in history
        if isinstance(turn, dict)
    ]

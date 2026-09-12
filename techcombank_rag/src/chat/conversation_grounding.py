from __future__ import annotations

import re
from typing import Any

from src.chat.query_rewriter import is_self_contained_question
from src.chat.text_safety import normalize_history, normalize_utf8_text


MULTI_REFERENCE_RE = re.compile(
    r"\b(?:các|những|hai|ba|cả hai|cả ba)\s+"
    r"(?:số liệu|chỉ tiêu|kết quả)(?:\s+(?:này|đó|trên|vừa nêu))?\b|"
    r"\b(?:phân biệt|so sánh)\s+(?:chúng|cả hai|cả ba)\b",
    re.IGNORECASE,
)


def referenced_query_count(question: str) -> int:
    lowered = normalize_utf8_text(question).casefold()
    if "ba số liệu" in lowered or "ba chỉ tiêu" in lowered or "cả ba" in lowered:
        return 3
    if "hai số liệu" in lowered or "hai chỉ tiêu" in lowered or "cả hai" in lowered:
        return 2
    return 3


def history_queries_for_multi_reference(
    question: str,
    history: list[dict[str, str]],
    *,
    maximum_queries: int = 3,
) -> list[str]:
    """Resolve plural references from user turns, never from assistant answers."""
    current = normalize_utf8_text(question).strip()
    if not MULTI_REFERENCE_RE.search(current):
        return []
    target = min(maximum_queries, referenced_query_count(current))
    selected: list[str] = []
    seen: set[str] = set()
    for turn in reversed(normalize_history(history)):
        if turn.get("role") != "user":
            continue
        candidate = turn.get("content", "").strip()
        key = candidate.casefold()
        if not candidate or key in seen or not is_self_contained_question(candidate):
            continue
        selected.append(candidate)
        seen.add(key)
        if len(selected) == target:
            break
    return list(reversed(selected)) if len(selected) >= 2 else []


def merge_round_robin(
    candidate_groups: list[list[dict[str, Any]]],
    queries: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Preserve at least one high-ranked result per decomposed query."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    maximum = max((len(group) for group in candidate_groups), default=0)
    for rank in range(maximum):
        for group_index, group in enumerate(candidate_groups):
            if rank >= len(group):
                continue
            row = dict(group[rank])
            chunk_id = str(row.get("chunk_id", ""))
            if chunk_id in seen:
                continue
            row["decomposition_query_index"] = group_index
            row["decomposition_rank"] = rank + 1
            if queries and group_index < len(queries):
                row["decomposition_query"] = queries[group_index]
            merged.append(row)
            seen.add(chunk_id)
    return merged

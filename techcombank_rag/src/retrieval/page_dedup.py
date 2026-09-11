from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


TOKEN_RE = re.compile(r"[\w%]+", re.UNICODE)


def candidate_page(candidate: dict[str, Any]) -> int | None:
    value = candidate.get("printed_page", candidate.get("page"))
    return int(value) if value is not None else None


def candidate_score(candidate: dict[str, Any]) -> float:
    return float(
        candidate.get(
            "retrieval_score",
            candidate.get("score", candidate.get("fusion_score", 0.0)),
        )
    )


def candidate_representation(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("granularity")
        or candidate.get("block_type")
        or candidate.get("representation")
        or candidate.get("retrieval_source")
        or "unknown"
    ).casefold()


@lru_cache(maxsize=8192)
def normalized_tokens(text: str) -> frozenset[str]:
    return frozenset(TOKEN_RE.findall(text.casefold()))


def content_similarity(left: str, right: str) -> float:
    """Lexical overlap robust to row/block/page containment."""
    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return float(left.strip().casefold() == right.strip().casefold())
    overlap = len(left_tokens & right_tokens)
    jaccard = overlap / len(left_tokens | right_tokens)
    containment = overlap / min(len(left_tokens), len(right_tokens))
    return max(jaccard, containment)


def deduplicate_by_page(
    candidates: list[dict[str, Any]],
    max_per_page: int = 2,
    similarity_threshold: float = 0.88,
) -> list[dict[str, Any]]:
    if max_per_page <= 0:
        raise ValueError("max_per_page must be positive")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")

    ranked = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda item: (-candidate_score(item), str(item.get("chunk_id", ""))),
    )
    kept: list[dict[str, Any]] = []
    by_page: dict[int | None, list[dict[str, Any]]] = {}
    seen_chunk_ids: set[str] = set()
    for candidate in ranked:
        chunk_id = str(candidate.get("chunk_id", ""))
        if chunk_id and chunk_id in seen_chunk_ids:
            continue
        page = candidate_page(candidate)
        page_items = by_page.setdefault(page, [])
        if len(page_items) >= max_per_page:
            continue
        text = str(candidate.get("text", ""))
        if any(
            content_similarity(text, str(existing.get("text", "")))
            >= similarity_threshold
            for existing in page_items
        ):
            continue
        candidate.setdefault("page", page)
        candidate.setdefault("retrieval_score", candidate_score(candidate))
        candidate.setdefault("representation", candidate_representation(candidate))
        kept.append(candidate)
        page_items.append(candidate)
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
    return kept

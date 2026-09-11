from __future__ import annotations

import math
import re
from typing import Any

from src.retrieval.page_dedup import (
    candidate_page,
    candidate_representation,
    candidate_score,
    content_similarity,
)


NUMERIC_QUERY_RE = re.compile(
    r"\b(?:bao nhiêu|tỷ lệ|tỷ suất|năm\s+20\d{2}|tăng trưởng|cagr|roe|roa|nim|casa|npl)\b|%",
    re.IGNORECASE,
)
EXPLANATORY_QUERY_RE = re.compile(
    r"\b(?:tại sao|vì sao|giải thích|nguyên nhân|chiến lược|chính sách|vai trò|mô tả)\b",
    re.IGNORECASE,
)

QUERY_STOPWORDS = frozenset(
    {
        "bao",
        "của",
        "cho",
        "là",
        "nào",
        "năm",
        "nhiêu",
        "tại",
        "theo",
        "thế",
        "thì",
        "và",
        "với",
        "được",
    }
)


def _representation_bonus(query: str, representation: str) -> float:
    numeric = bool(NUMERIC_QUERY_RE.search(query))
    explanatory = bool(EXPLANATORY_QUERY_RE.search(query))
    representation = representation.casefold()
    if numeric and not explanatory:
        if "row" in representation or "table_row" in representation:
            return 0.06
        if "block" in representation or "table" in representation:
            return 0.03
        if "page" in representation:
            return -0.03
    if explanatory:
        if "paragraph" in representation or "narrative" in representation:
            return 0.05
        if "block" in representation:
            return 0.03
        if "row" in representation:
            return -0.03
    return 0.0


def _lexical_query_coverage(query: str, text: str) -> float:
    query_terms = {
        token
        for token in re.findall(r"[\w%]+", query.casefold())
        if token not in QUERY_STOPWORDS and len(token) > 1
    }
    if not query_terms:
        return 0.0
    text_terms = set(re.findall(r"[\w%]+", text.casefold()))
    return len(query_terms & text_terms) / len(query_terms)


def _normalized_relevance(candidates: list[dict[str, Any]]) -> list[float]:
    scores = [candidate_score(candidate) for candidate in candidates]
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if math.isclose(low, high):
        return [1.0] * len(scores)
    if low >= 0 and high > 0:
        # Preserve absolute relevance gaps. Min-max would force the weakest
        # still-relevant candidate to zero and overpower MMR's novelty term.
        return [score / high for score in scores]
    return [(score - low) / (high - low) for score in scores]


def select_diverse_evidence(
    query: str,
    candidates: list[dict[str, Any]],
    final_k: int,
    lambda_relevance: float = 0.7,
    protected_k: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministic lexical MMR with page and representation penalties."""
    if final_k <= 0:
        return []
    if not 0 <= lambda_relevance <= 1:
        raise ValueError("lambda_relevance must be between 0 and 1")
    pool = [dict(candidate) for candidate in candidates]
    relevances = _normalized_relevance(pool)
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    while len(selected) < min(final_k, len(pool)):
        best_index = -1
        best_score = float("-inf")
        for index, candidate in enumerate(pool):
            if index in selected_indices:
                continue
            redundancy = 0.0
            page_penalty = 0.0
            representation_penalty = 0.0
            if selected:
                redundancy = max(
                    content_similarity(
                        str(candidate.get("text", "")),
                        str(existing.get("text", "")),
                    )
                    for existing in selected
                )
                if any(
                    candidate_page(candidate) == candidate_page(existing)
                    for existing in selected
                ):
                    page_penalty = 0.08
                representation = candidate_representation(candidate)
                if any(
                    representation == candidate_representation(existing)
                    for existing in selected
                ):
                    representation_penalty = 0.025
            mmr = (
                lambda_relevance * relevances[index]
                - (1 - lambda_relevance) * redundancy
                - page_penalty
                - representation_penalty
                + _representation_bonus(query, candidate_representation(candidate))
            )
            tie_breaker = candidate_score(candidate) * 1e-6
            score = mmr + tie_breaker
            if score > best_score:
                best_score = score
                best_index = index
        if best_index < 0:
            break
        item = pool[best_index]
        item["diversity_score"] = round(best_score, 9)
        item.setdefault("page", candidate_page(item))
        item.setdefault("retrieval_score", candidate_score(item))
        item.setdefault("representation", candidate_representation(item))
        selected.append(item)
        selected_indices.add(best_index)

    # Safety rail: MMR may prefer novelty when dense scores are tightly
    # clustered. Reserve one slot for the strongest direct lexical match so
    # diversity cannot crowd out a chunk that explicitly names the requested
    # metric/entity. This is applied only for a strong match and does not
    # otherwise alter the MMR result.
    if selected:
        evidence_window = min(protected_k or min(5, final_k), final_k)
        coverages = [
            _lexical_query_coverage(query, str(candidate.get("text", "")))
            for candidate in pool
        ]
        direct_index = max(
            range(len(pool)),
            key=lambda index: (coverages[index], candidate_score(pool[index])),
        )
        direct = pool[direct_index]
        direct_id = str(direct.get("chunk_id"))
        top_score = max(candidate_score(candidate) for candidate in pool)
        score_floor = 0.85 * top_score if top_score > 0 else top_score
        if (
            coverages[direct_index] >= 0.50
            and candidate_score(direct) >= score_floor
        ):
            current_position = next(
                (
                    index
                    for index, item in enumerate(selected)
                    if str(item.get("chunk_id")) == direct_id
                ),
                None,
            )
            if current_position is not None and current_position < evidence_window:
                return selected
            if current_position is not None:
                replacement = selected.pop(current_position)
            else:
                replacement = dict(direct)
                replacement["diversity_score"] = round(
                    lambda_relevance * relevances[direct_index], 9
                )
                replacement.setdefault("page", candidate_page(replacement))
                replacement.setdefault("retrieval_score", candidate_score(replacement))
                replacement.setdefault(
                    "representation", candidate_representation(replacement)
                )
            selected.insert(evidence_window - 1, replacement)
            del selected[final_k:]
    return selected


def estimate_tokens(text: str) -> int:
    # Conservative tokenizer-free approximation; avoids a model/tokenizer load.
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def select_with_token_budget(
    candidates: list[dict[str, Any]],
    *,
    max_chunks: int = 6,
    max_tokens: int = 5000,
) -> tuple[list[dict[str, Any]], int]:
    if max_chunks <= 0 or max_tokens <= 0:
        return [], 0
    selected: list[dict[str, Any]] = []
    used_tokens = 0
    for candidate in candidates:
        token_count = estimate_tokens(str(candidate.get("text", "")))
        if used_tokens + token_count > max_tokens:
            continue
        item = dict(candidate)
        item["estimated_tokens"] = token_count
        selected.append(item)
        used_tokens += token_count
        if len(selected) >= max_chunks:
            break
    return selected, used_tokens

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from src.retrieval.metric_identity import normalize_search_text, normalized_terms


CITATION_RE = re.compile(r"\[tr\.\s*\d+(?:\s*,\s*\d+)*\]", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![\w])\d+(?:\.\d{3})*(?:,\d+)?%?")
STOPWORDS = {
    "của", "và", "là", "được", "năm", "tại", "theo", "trong", "mức", "với",
    "techcombank", "thì", "cho", "các", "khoản", "từ", "hoạt", "động",
}


def _numbers(text: str) -> set[str]:
    values = set()
    for value in NUMBER_RE.findall(CITATION_RE.sub("", text)):
        raw = value.rstrip("%")
        if len(raw) == 4 and raw.isdigit() and 1900 <= int(raw) <= 2100:
            continue
        values.add(value.replace(" ", ""))
    return values


def _distinctive_terms(text: str) -> set[str]:
    return {
        term for term in normalized_terms(CITATION_RE.sub("", text))
        if term not in STOPWORDS and len(term) > 1 and not term.isdigit()
    }


def canonicalize_citation(
    answer: str,
    chunks: list[dict[str, Any]],
    *,
    minimum_term_coverage: float = 0.62,
) -> tuple[str, list[int]] | None:
    """Choose the smallest retrieved page that supports the complete answer.

    Numeric coverage is mandatory. If multiple pages fully support the same
    claim, the earliest printed page is a stable canonical tie-breaker.
    """
    page_texts: dict[int, list[str]] = defaultdict(list)
    for chunk in chunks:
        page_texts[int(chunk["printed_page"])].append(str(chunk.get("text", "")))
    required_numbers = _numbers(answer)
    required_terms = _distinctive_terms(answer)
    candidates: list[tuple[float, int]] = []
    for page, texts in page_texts.items():
        text = normalize_search_text(" ".join(texts)).casefold()
        compact = text.replace(" ", "")
        if required_numbers and not all(value.casefold() in compact for value in required_numbers):
            continue
        coverage = (
            sum(term in text for term in required_terms) / len(required_terms)
            if required_terms else 1.0
        )
        if coverage >= minimum_term_coverage:
            candidates.append((coverage, page))
    if not candidates:
        return None
    best_coverage = max(item[0] for item in candidates)
    near_best = [page for coverage, page in candidates if coverage >= best_coverage - 0.05]
    page = min(near_best)
    clean = CITATION_RE.sub("", answer).strip()
    clean = re.sub(r"\s+([.,;:])", r"\1", clean).rstrip()
    clean = clean.rstrip(".")
    return f"{clean} [tr. {page}].", [page]

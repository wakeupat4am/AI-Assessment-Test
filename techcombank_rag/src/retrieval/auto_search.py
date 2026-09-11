from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.retrieval.diversity import (
    select_diverse_evidence,
    select_with_token_budget,
)
from src.retrieval.page_dedup import candidate_page, candidate_score, deduplicate_by_page


YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?%?")
COMPARISON_RE = re.compile(
    r"\b(?:so với|so sánh|thay đổi|chênh lệch|năm trước|cùng kỳ|giai đoạn|cagr|tăng trưởng|tăng hay giảm)\b",
    re.IGNORECASE,
)
EXPLANATION_RE = re.compile(
    r"\b(?:tại sao|vì sao|nguyên nhân|giải thích|do đâu)\b", re.IGNORECASE
)
EXPLANATION_EVIDENCE_RE = re.compile(
    r"\b(?:do|nhờ|bởi|nguyên nhân|động lực|đến từ|hỗ trợ|thúc đẩy|phản ánh|chủ yếu)\b",
    re.IGNORECASE,
)
NUMERIC_INTENT_RE = re.compile(
    r"\b(?:bao nhiêu|mức nào|đạt|tỷ lệ|tỷ suất|tăng trưởng|cagr|số lượng|liệt kê)\b|%",
    re.IGNORECASE,
)
DEFINITION_RE = re.compile(r"\b(?:viết tắt|thuật ngữ gì|là gì)\b", re.IGNORECASE)

TRACKED_TERMS = (
    "lợi nhuận trước thuế",
    "tổng thu nhập hoạt động",
    "tổng tài sản",
    "tiền gửi không kỳ hạn",
    "phòng giao dịch",
    "tỉnh thành",
    "chi nhánh",
    "dư nợ",
    "tiền gửi",
    "nợ xấu",
    "xếp hạng tín nhiệm",
    "casa",
    "npl",
    "roe",
    "roa",
    "nim",
    "rbg",
    "s&p",
    "fitch",
    "techcombank rewards",
)

TERM_ALIASES = {
    "xếp hạng tín nhiệm": ("xếp hạng tín nhiệm", "xếp hạng"),
    "nợ xấu": ("nợ xấu", "npl"),
}


@dataclass
class SufficiencyResult:
    sufficient: bool
    confidence: float
    missing_information: list[str]
    followup_query: str | None
    reason: str


@dataclass
class SearchState:
    original_query: str
    search_round: int = 0
    queries_used: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_evidence: list[dict[str, Any]] = field(default_factory=list)
    sufficient: bool = False
    missing_information: list[str] = field(default_factory=list)
    stop_reason: str | None = None


@dataclass
class SearchResult:
    candidates: list[dict[str, Any]]
    selected_evidence: list[dict[str, Any]]
    state: SearchState
    trace: dict[str, Any]


def _query_terms(query: str) -> list[str]:
    lowered = query.casefold()
    return [term for term in TRACKED_TERMS if term in lowered]


def _non_year_numbers(text: str) -> list[str]:
    values: list[str] = []
    for value in NUMBER_RE.findall(text):
        raw = value.rstrip("%")
        if raw.isdigit() and len(raw) == 4 and 1900 <= int(raw) <= 2100:
            continue
        values.append(value)
    return values


def _focused_followup(
    query: str, missing: list[str], terms: list[str], years: list[str]
) -> str:
    metric = " ".join(terms[:2]) or "Techcombank"
    year_text = " ".join(years)
    if any(item.startswith("year:") for item in missing):
        missing_years = " ".join(item.split(":", 1)[1] for item in missing if item.startswith("year:"))
        return f"{metric} năm {missing_years} số liệu chính xác".strip()
    if "explanation" in missing:
        return f"nguyên nhân {metric} thay đổi {year_text}".strip()
    missing_terms = [item.split(":", 1)[1] for item in missing if item.startswith("term:")]
    if missing_terms:
        return f"{' '.join(missing_terms)} {year_text} Techcombank".strip()
    if "definition" in missing:
        return f"danh mục thuật ngữ {metric} viết tắt".strip()
    if "comparison_values" in missing:
        return f"{metric} {year_text} số liệu so sánh năm trước".strip()
    if "numeric_value" in missing:
        return f"{metric} {year_text} bảng số liệu".strip()
    return f"{metric} {year_text} thông tin báo cáo".strip()


def check_evidence_sufficiency(
    query: str, evidence: list[dict[str, Any]]
) -> SufficiencyResult:
    if not evidence:
        return SufficiencyResult(
            False, 0.0, ["no_evidence"], "Techcombank thông tin báo cáo", "no evidence"
        )
    combined = "\n".join(str(item.get("text", "")) for item in evidence)
    lowered = combined.casefold()
    years = list(dict.fromkeys(YEAR_RE.findall(query)))
    terms = _query_terms(query)
    missing: list[str] = []
    checks: list[bool] = []

    for year in years:
        present = year in combined
        checks.append(present)
        if not present:
            missing.append(f"year:{year}")
    for term in terms:
        present = any(alias in lowered for alias in TERM_ALIASES.get(term, (term,)))
        checks.append(present)
        if not present:
            missing.append(f"term:{term}")

    numeric_intent = bool(NUMERIC_INTENT_RE.search(query))
    evidence_numbers = _non_year_numbers(combined)
    if numeric_intent:
        numeric_ok = bool(evidence_numbers)
        checks.append(numeric_ok)
        if not numeric_ok:
            missing.append("numeric_value")

    comparison = bool(COMPARISON_RE.search(query)) or len(years) >= 2
    if comparison:
        comparison_ok = len(set(evidence_numbers)) >= 2
        checks.append(comparison_ok)
        if not comparison_ok:
            missing.append("comparison_values")

    if EXPLANATION_RE.search(query):
        explanation_ok = bool(EXPLANATION_EVIDENCE_RE.search(combined))
        checks.append(explanation_ok)
        if not explanation_ok:
            missing.append("explanation")

    if DEFINITION_RE.search(query):
        definition_ok = "viết tắt" in lowered or any(
            marker in lowered for marker in ("khối", "tiền gửi không kỳ hạn")
        )
        checks.append(definition_ok)
        if not definition_ok:
            missing.append("definition")

    # Similarity is useful only as a weak guard; it never proves sufficiency.
    top_score = max(candidate_score(item) for item in evidence)
    relevance_ok = top_score >= 0.50
    checks.append(relevance_ok)
    passed = sum(checks)
    confidence = passed / len(checks) if checks else 0.0
    sufficient = bool(checks) and all(checks) and not missing
    followup = None if sufficient else _focused_followup(query, missing, terms, years)
    reason = (
        "all deterministic coverage requirements satisfied"
        if sufficient
        else f"missing {', '.join(missing) if missing else 'whole-question support'}"
    )
    return SufficiencyResult(
        sufficient=sufficient,
        confidence=round(confidence, 6),
        missing_information=missing,
        followup_query=followup,
        reason=reason,
    )


def _merge_candidates(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for round_number, rows in ((1, first), (2, second)):
        for row in rows:
            key = str(row.get("chunk_id") or f"anonymous-{round_number}-{len(order)}")
            if key not in merged:
                item = dict(row)
                item["search_rounds_found"] = [round_number]
                merged[key] = item
                order.append(key)
            else:
                merged[key]["search_rounds_found"].append(round_number)
                if candidate_score(row) > candidate_score(merged[key]):
                    preserved_rounds = merged[key]["search_rounds_found"]
                    merged[key] = {**row, "search_rounds_found": preserved_rounds}
    original_position = {key: index for index, key in enumerate(order)}
    return sorted(
        (merged[key] for key in order),
        key=lambda item: (
            -candidate_score(item),
            original_position.get(str(item.get("chunk_id")), len(order)),
        ),
    )


def _call_retriever(
    retriever: Any,
    query: str,
    top_k: int,
    *,
    conversation_context: str | None = None,
    router_query: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, float]:
    started = time.perf_counter()
    if hasattr(retriever, "retrieve_routed"):
        rows = retriever.retrieve_routed(
            query,
            top_k,
            conversation_context=conversation_context,
            router_query=router_query or query,
        )
    else:
        rows = retriever.retrieve(query, top_k)
    elapsed = time.perf_counter() - started
    trace = getattr(retriever, "last_trace", None)
    return [dict(row) for row in rows], dict(trace) if trace else None, elapsed


def adaptive_search(
    query: str,
    retriever: Any,
    reranker: Any | None = None,
    max_rounds: int = 2,
    *,
    requested_k: int = 10,
    initial_top_k: int = 20,
    second_round_top_k: int = 12,
    max_per_page: int = 2,
    similarity_threshold: float = 0.88,
    diversity_final_k: int = 10,
    evidence_max_chunks: int = 6,
    evidence_max_tokens: int = 5000,
    lambda_relevance: float = 0.7,
    enable_auto_search: bool = True,
    enable_page_dedup: bool = True,
    enable_diversity_selection: bool = True,
    conversation_context: str | None = None,
    router_query: str | None = None,
) -> SearchResult:
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    max_rounds = min(max_rounds, 2)
    total_started = time.perf_counter()
    state = SearchState(original_query=query)
    retrieval_seconds = 0.0
    rerank_seconds = 0.0
    round_traces: list[dict[str, Any] | None] = []
    pool_k = initial_top_k if any(
        (enable_auto_search, enable_page_dedup, enable_diversity_selection)
    ) else requested_k
    initial, trace, elapsed = _call_retriever(
        retriever,
        query,
        pool_k,
        conversation_context=conversation_context,
        router_query=router_query,
    )
    retrieval_seconds += elapsed
    round_traces.append(trace)
    state.search_round = 1
    state.queries_used.append(query)

    def process(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
        after_dedup = (
            deduplicate_by_page(rows, max_per_page, similarity_threshold)
            if enable_page_dedup
            else [dict(row) for row in rows]
        )
        after_diversity = (
            select_diverse_evidence(
                query,
                after_dedup,
                max(requested_k, diversity_final_k),
                lambda_relevance,
                protected_k=evidence_max_chunks,
            )
            if enable_diversity_selection
            else after_dedup
        )
        return after_diversity, len(after_dedup), len(after_diversity)

    processed, after_dedup_count, after_diversity_count = process(initial)
    evidence, evidence_tokens = select_with_token_budget(
        processed, max_chunks=evidence_max_chunks, max_tokens=evidence_max_tokens
    )
    sufficiency = check_evidence_sufficiency(query, evidence)

    if enable_auto_search and not sufficiency.sufficient and max_rounds > 1:
        followup = sufficiency.followup_query
        if followup and followup.casefold().strip() != query.casefold().strip():
            second, trace, elapsed = _call_retriever(
                retriever,
                followup,
                second_round_top_k,
                conversation_context=conversation_context,
                router_query=followup,
            )
            retrieval_seconds += elapsed
            round_traces.append(trace)
            state.search_round = 2
            state.queries_used.append(followup)
            processed, after_dedup_count, after_diversity_count = process(
                _merge_candidates(initial, second)
            )
            if reranker is not None:
                rerank_started = time.perf_counter()
                processed = list(
                    reranker.rerank(query, processed, top_k=max(requested_k, diversity_final_k))
                )
                rerank_seconds += time.perf_counter() - rerank_started
            evidence, evidence_tokens = select_with_token_budget(
                processed, max_chunks=evidence_max_chunks, max_tokens=evidence_max_tokens
            )
            sufficiency = check_evidence_sufficiency(query, evidence)
            state.stop_reason = (
                "sufficient_after_round_2"
                if sufficiency.sufficient
                else "max_rounds_reached_insufficient"
            )
        else:
            state.stop_reason = "no_distinct_followup_query"
    elif sufficiency.sufficient:
        state.stop_reason = "sufficient_after_round_1"
    elif not enable_auto_search:
        state.stop_reason = "auto_search_disabled"
    else:
        state.stop_reason = "max_rounds_reached_insufficient"

    state.candidates = processed[:requested_k]
    state.selected_evidence = evidence
    state.sufficient = sufficiency.sufficient
    state.missing_information = sufficiency.missing_information
    final_pages = [
        page for page in (candidate_page(item) for item in evidence) if page is not None
    ]
    trace_payload = {
        "query": query,
        "search_rounds": state.search_round,
        "queries_used": state.queries_used,
        "initial_candidates": len(initial),
        "after_page_dedup": after_dedup_count,
        "after_diversity": after_diversity_count,
        "final_evidence_count": len(evidence),
        "final_pages": final_pages,
        "final_evidence_tokens": evidence_tokens,
        "sufficiency_confidence": sufficiency.confidence,
        "sufficient": sufficiency.sufficient,
        "missing_information": sufficiency.missing_information,
        "stop_reason": state.stop_reason,
        "retrieval_latency_ms": round(retrieval_seconds * 1000, 3),
        "rerank_latency_ms": round(rerank_seconds * 1000, 3),
        "total_retrieval_latency_ms": round(
            (time.perf_counter() - total_started) * 1000, 3
        ),
        "round_traces": round_traces,
    }
    return SearchResult(state.candidates, evidence, state, trace_payload)


class AdaptiveRetriever:
    """B2 wrapper that keeps the B1 retriever and answer layer unchanged."""

    def __init__(
        self,
        retriever: Any,
        *,
        enable_auto_search: bool = True,
        enable_page_dedup: bool = True,
        enable_diversity_selection: bool = True,
        initial_top_k: int = 20,
        second_round_top_k: int = 12,
        max_search_rounds: int = 2,
        max_per_page: int = 2,
        similarity_threshold: float = 0.88,
        final_k: int = 6,
        max_evidence_tokens: int = 5000,
        lambda_relevance: float = 0.7,
        log_path: Path | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.base = retriever
        self.enable_auto_search = enable_auto_search
        self.enable_page_dedup = enable_page_dedup
        self.enable_diversity_selection = enable_diversity_selection
        self.initial_top_k = initial_top_k
        self.second_round_top_k = second_round_top_k
        self.max_search_rounds = max_search_rounds
        self.max_per_page = max_per_page
        self.similarity_threshold = similarity_threshold
        self.final_k = final_k
        self.max_evidence_tokens = max_evidence_tokens
        self.lambda_relevance = lambda_relevance
        self.log_path = Path(log_path) if log_path else None
        self.reranker = reranker
        self.last_trace: dict[str, Any] | None = None
        self.last_selected_evidence: list[dict[str, Any]] | None = None
        self.last_router_call: Any | None = None
        self._log_lock = threading.Lock()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.retrieve_routed(query, top_k=top_k)

    def retrieve_routed(
        self,
        query: str,
        top_k: int | None = None,
        *,
        conversation_context: str | None = None,
        router_query: str | None = None,
    ) -> list[dict[str, Any]]:
        result = adaptive_search(
            query,
            self.base,
            self.reranker,
            self.max_search_rounds,
            requested_k=top_k or 10,
            initial_top_k=self.initial_top_k,
            second_round_top_k=self.second_round_top_k,
            max_per_page=self.max_per_page,
            similarity_threshold=self.similarity_threshold,
            diversity_final_k=max(top_k or 10, self.final_k),
            evidence_max_chunks=self.final_k,
            evidence_max_tokens=self.max_evidence_tokens,
            lambda_relevance=self.lambda_relevance,
            enable_auto_search=self.enable_auto_search,
            enable_page_dedup=self.enable_page_dedup,
            enable_diversity_selection=self.enable_diversity_selection,
            conversation_context=conversation_context,
            router_query=router_query,
        )
        self.last_selected_evidence = result.selected_evidence
        self.last_router_call = getattr(self.base, "last_router_call", None)
        base_trace = result.trace["round_traces"][0] or {}
        self.last_trace = {
            **base_trace,
            "b2": result.trace,
            "raw_candidate_count": sum(
                int(trace.get("raw_candidate_count", 0))
                for trace in result.trace["round_traces"]
                if trace
            ),
            "returned_candidate_count": len(result.candidates),
            "retrieval_latency_seconds": round(
                result.trace["total_retrieval_latency_ms"] / 1000, 9
            ),
        }
        if self.log_path:
            payload = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                **result.trace,
            }
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_lock, self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return result.candidates

from __future__ import annotations

import json
import hashlib
import math
import platform
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.chat.chatbot import Chatbot
from src.config import Settings, get_settings
from src.llm.base import TokenUsage
from src.llm.pricing import KNOWN_PRICES, calculate_cost


NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)*(?:%|x)?", re.IGNORECASE)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
FAILURE_CATEGORIES = {
    "none",
    "retrieval_miss",
    "false_refusal",
    "unsafe_answer",
    "citation_error",
    "answer_error",
    "runtime_error",
}


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Question file must contain a JSON list")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(data, 1):
        if isinstance(item, str):
            row: dict[str, Any] = {
                "id": f"q-{index:03d}",
                "question": item,
                "gold_answer": None,
                "gold_printed_pages": [],
                "answerable": None,
                "question_type": "unspecified",
                "requires_calculation": False,
            }
        elif isinstance(item, dict) and item.get("question"):
            row = dict(item)
            row.setdefault("id", f"q-{index:03d}")
            row.setdefault("gold_answer", None)
            row.setdefault("gold_printed_pages", [])
            row.setdefault("answerable", None)
            row.setdefault("question_type", row.get("category", "unspecified"))
            row.setdefault("requires_calculation", False)
        else:
            raise ValueError(f"Invalid question item at position {index}: {item!r}")
        row["id"] = str(row["id"])
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate question id: {row['id']}")
        seen_ids.add(row["id"])
        pages = row["gold_printed_pages"]
        if not isinstance(pages, list) or not all(isinstance(page, int) for page in pages):
            raise ValueError(f"gold_printed_pages must be a list of ints for {row['id']}")
        if row["answerable"] is True and not pages:
            raise ValueError(f"Answerable question {row['id']} requires gold_printed_pages")
        rows.append(row)
    return rows


def load_manual_reviews(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return {str(item["id"]): item for item in data}
    if isinstance(data, dict):
        return {str(key): value for key, value in data.items()}
    raise ValueError("Manual review must be a JSON object or list")


def _normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _numbers(text: str) -> set[str]:
    return {match.group(0).lower().replace(" ", "") for match in NUMBER_RE.finditer(text)}


def _page_metrics(retrieved: list[int], gold: list[int], k: int) -> tuple[float, bool]:
    if not gold:
        return math.nan, False
    found = set(retrieved[:k]) & set(gold)
    return len(found) / len(set(gold)), bool(found)


def _safe_mean(values: list[float]) -> float | None:
    filtered = [value for value in values if not math.isnan(value)]
    return round(statistics.mean(filtered), 6) if filtered else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 6)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_response(
    item: dict[str, Any],
    response: dict[str, Any],
    manual_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold_pages = [int(page) for page in item.get("gold_printed_pages", [])]
    retrieved_pages = [
        int(chunk["printed_page"]) for chunk in response.get("retrieved_chunks", [])
    ]
    citations = [int(page) for page in response.get("citations", [])]
    answerable = item.get("answerable")
    refused = bool(response.get("refused"))
    gold_answer = item.get("gold_answer") or ""

    page_scores: dict[str, float | None] = {}
    page_hits: dict[str, bool] = {}
    for k in (1, 5, 10):
        recall, hit = _page_metrics(retrieved_pages, gold_pages, k)
        page_scores[f"recall_at_{k}"] = None if math.isnan(recall) else recall
        page_hits[f"hit_at_{k}"] = hit

    gold_set = set(gold_pages)
    citation_set = set(citations)
    citation_precision = (
        len(citation_set & gold_set) / len(citation_set) if citation_set else 0.0
    )
    citation_recall = len(citation_set & gold_set) / len(gold_set) if gold_set else math.nan
    token_f1 = _token_f1(response.get("answer", ""), gold_answer) if gold_answer else math.nan
    gold_numbers = _numbers(gold_answer)
    answer_numbers = _numbers(response.get("answer", ""))
    numeric_recall = (
        len(gold_numbers & answer_numbers) / len(gold_numbers) if gold_numbers else math.nan
    )
    if answerable is False:
        automatic_correct: bool | None = refused
    elif answerable is True and gold_answer:
        numbers_ok = math.isnan(numeric_recall) or numeric_recall == 1.0
        automatic_correct = not refused and numbers_ok and token_f1 >= 0.45
    else:
        automatic_correct = None

    manual_correct = None
    manual_notes = None
    manual_failure = None
    if manual_review:
        manual_correct = manual_review.get("correct")
        manual_notes = manual_review.get("notes")
        manual_failure = manual_review.get("failure_category")
        if manual_failure is not None and manual_failure not in FAILURE_CATEGORIES:
            raise ValueError(f"Unknown manual failure category: {manual_failure}")

    effective_correct = manual_correct if manual_correct is not None else automatic_correct
    retrieval_hit = page_hits["hit_at_10"]
    if response.get("error"):
        failure = "runtime_error"
    elif answerable is True and not retrieval_hit:
        failure = "retrieval_miss"
    elif answerable is True and refused:
        failure = "false_refusal"
    elif answerable is False and not refused:
        failure = "unsafe_answer"
    elif answerable is True and (
        citation_precision < 1.0
        or (not math.isnan(citation_recall) and citation_recall < 1.0)
        or not response.get("citation_valid", False)
    ):
        failure = "citation_error"
    elif effective_correct is False:
        failure = "answer_error"
    else:
        failure = "none"
    if manual_failure:
        failure = manual_failure

    return {
        **page_scores,
        **page_hits,
        "citation_precision": round(citation_precision, 6),
        "citation_recall": round(citation_recall, 6) if not math.isnan(citation_recall) else None,
        "citation_exact": citation_set == gold_set if gold_set else not citation_set,
        "token_f1": round(token_f1, 6) if not math.isnan(token_f1) else None,
        "numeric_recall": round(numeric_recall, 6) if not math.isnan(numeric_recall) else None,
        "automatic_correct": automatic_correct,
        "manual_correct": manual_correct,
        "effective_correct": effective_correct,
        "manual_notes": manual_notes,
        "failure_category": failure,
    }


def summarize(
    rows: list[dict[str, Any]],
    settings: Settings,
    run_name: str,
    questions_path: Path,
) -> dict[str, Any]:
    scores = [row["scores"] for row in rows]
    latencies = [float(row["latency_seconds"]) for row in rows if row.get("error") is None]
    answerable_scores = [score for row, score in zip(rows, scores) if row.get("answerable") is True]
    judged = [score["effective_correct"] for score in scores if score["effective_correct"] is not None]
    manual = [score["manual_correct"] for score in scores if score["manual_correct"] is not None]
    usage_rows = [row.get("llm", {}).get("usage", {}) for row in rows]
    costs = [row.get("llm", {}).get("cost_usd") for row in rows]
    known_costs = [float(value) for value in costs if value is not None]
    call_usage_sources = {
        str(call.get("usage", {}).get("source", "unavailable"))
        for row in rows
        for call in row.get("llm", {}).get("calls", [])
    }
    usage_sources = call_usage_sources or {
        str(u.get("source", "unavailable")) for u in usage_rows
    }
    total_usage = TokenUsage(
        uncached_input_tokens=sum(int(u.get("uncached_input_tokens", 0)) for u in usage_rows),
        cached_input_tokens=sum(int(u.get("cached_input_tokens", 0)) for u in usage_rows),
        cache_write_input_tokens=sum(int(u.get("cache_write_input_tokens", 0)) for u in usage_rows),
        output_tokens=sum(int(u.get("output_tokens", 0)) for u in usage_rows),
        reasoning_tokens=sum(int(u.get("reasoning_tokens", 0)) for u in usage_rows),
        source=next(iter(usage_sources)) if len(usage_sources) == 1 else "mixed",
    )
    purpose_calls = Counter(
        call.get("purpose", "unknown")
        for row in rows
        for call in row.get("llm", {}).get("calls", [])
    )
    provider_models = Counter(
        f"{call.get('provider', 'unknown')}::{call.get('model', 'unknown')}"
        for row in rows
        for call in row.get("llm", {}).get("calls", [])
    )
    routing_rows = [row["routing"] for row in rows if row.get("routing")]
    route_decisions = [row for row in routing_rows if row.get("route")]
    index_metadata_path = settings.index_dir / "metadata.json"
    index_metadata = {}
    if index_metadata_path.exists():
        with index_metadata_path.open(encoding="utf-8") as handle:
            index_metadata = json.load(handle)
    try:
        portable_index_dir = settings.index_dir.relative_to(settings.repository_root).as_posix()
    except ValueError:
        portable_index_dir = str(settings.index_dir)
    return {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_method": {
            "retrieval": "printed-page recall and hit rate against manually verified gold pages",
            "answer_automatic": "deterministic token-F1>=0.45 plus exact numeric coverage; refusal exactness for unanswerable questions",
            "answer_primary": "manual review when provided, otherwise automatic heuristic",
            "warning": "Automatic answer correctness is diagnostic, not a substitute for human review.",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "top_k_evidence": settings.top_k,
            "retrieval_k_evaluated": settings.evaluation_retrieval_k,
            "index_dir": portable_index_dir,
        },
        "artifacts": {
            "questions": str(questions_path),
            "questions_sha256": _sha256(questions_path),
            "index_metadata": index_metadata,
        },
        "counts": {
            "questions": len(rows),
            "answerable": sum(row.get("answerable") is True for row in rows),
            "unanswerable": sum(row.get("answerable") is False for row in rows),
            "runtime_errors": sum(row.get("error") is not None for row in rows),
            "manually_reviewed": len(manual),
        },
        "metrics": {
            "retrieval_recall_at_1": _safe_mean([s["recall_at_1"] for s in answerable_scores]),
            "retrieval_recall_at_5": _safe_mean([s["recall_at_5"] for s in answerable_scores]),
            "retrieval_recall_at_10": _safe_mean([s["recall_at_10"] for s in answerable_scores]),
            "retrieval_hit_at_1": _safe_mean([float(s["hit_at_1"]) for s in answerable_scores]),
            "retrieval_hit_at_5": _safe_mean([float(s["hit_at_5"]) for s in answerable_scores]),
            "retrieval_hit_at_10": _safe_mean([float(s["hit_at_10"]) for s in answerable_scores]),
            "answer_accuracy": round(sum(bool(value) for value in judged) / len(judged), 6) if judged else None,
            "manual_answer_accuracy": round(sum(bool(value) for value in manual) / len(manual), 6) if manual else None,
            "citation_precision": _safe_mean([s["citation_precision"] for s in answerable_scores]),
            "citation_recall": _safe_mean([
                float(s["citation_recall"])
                for s in answerable_scores
                if s["citation_recall"] is not None
            ]),
            "citation_validator_pass_rate": _safe_mean([float(bool(row.get("citation_valid"))) for row in rows]),
            "refusal_accuracy": _safe_mean([
                float(bool(row.get("refused")) == (row.get("answerable") is False))
                for row in rows
                if row.get("answerable") is not None
            ]),
            "latency_seconds_mean": _safe_mean(latencies),
            "latency_seconds_p50": _percentile(latencies, 0.50),
            "latency_seconds_p95": _percentile(latencies, 0.95),
            "llm_calls_total": sum(int(row.get("llm", {}).get("call_count", 0)) for row in rows),
            "tokens": total_usage.to_dict(),
            "calls_by_purpose": dict(sorted(purpose_calls.items())),
            "calls_by_provider_model": dict(sorted(provider_models.items())),
            "cost_usd_total": round(sum(known_costs), 8) if len(known_costs) == len(rows) else None,
            "cost_usd_mean": round(statistics.mean(known_costs), 8) if len(known_costs) == len(rows) and known_costs else None,
            "cost_projection_same_tokens_usd": {
                model: calculate_cost(total_usage, rates)
                for model, rates in KNOWN_PRICES.items()
            },
        },
        "routing_metrics": {
            "enabled": bool(routing_rows),
            "route_distribution": dict(
                sorted(Counter(str(row["route"]) for row in route_decisions).items())
            ),
            "confidence_mean": _safe_mean(
                [float(row["confidence"]) for row in route_decisions]
            ),
            "router_latency_seconds_p50": _percentile(
                [float(row["router_latency_seconds"]) for row in route_decisions], 0.50
            ),
            "router_latency_seconds_p95": _percentile(
                [float(row["router_latency_seconds"]) for row in route_decisions], 0.95
            ),
            "raw_candidates_mean": _safe_mean(
                [
                    float(row["raw_candidate_count"])
                    for row in route_decisions
                    if row.get("raw_candidate_count") is not None
                ]
            ),
            "returned_candidates_mean": _safe_mean(
                [
                    float(row["returned_candidate_count"])
                    for row in route_decisions
                    if row.get("returned_candidate_count") is not None
                ]
            ),
            "trace_types": sorted(
                {
                    "route" if row.get("route") else next(iter(row), "unknown")
                    for row in routing_rows
                }
            ),
        },
        "failure_taxonomy": dict(sorted(Counter(s["failure_category"] for s in scores).items())),
    }


def evaluate_questions(
    questions_path: Path,
    output_path: Path,
    chatbot: Any | None = None,
    *,
    summary_path: Path | None = None,
    run_name: str = "evaluation",
    manual_reviews_path: Path | None = None,
) -> list[dict[str, Any]]:
    default_settings = get_settings()
    bot = chatbot or Chatbot(default_settings)
    settings = getattr(bot, "settings", default_settings)
    manual_reviews = load_manual_reviews(manual_reviews_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as output:
        for item in load_questions(questions_path):
            try:
                response = bot.ask(item["question"], conversation_history=[])
                error = None
            except Exception as exception:  # keep batch runs auditable and complete
                response = {
                    "answer": "",
                    "standalone_query": item["question"],
                    "retrieved_chunks": [],
                    "citations": [],
                    "citation_valid": False,
                    "refused": False,
                    "latency_seconds": 0.0,
                    "latency_breakdown": {},
                    "llm": {"call_count": 0, "usage": {}, "cost_usd": None, "calls": []},
                    "routing": None,
                }
                error = f"{type(exception).__name__}: {exception}"
            scored_response = {**response, "error": error}
            scores = score_response(item, scored_response, manual_reviews.get(item["id"]))
            row = {
                **item,
                "standalone_query": response["standalone_query"],
                "answer": response["answer"],
                "retrieved_pages": [chunk["printed_page"] for chunk in response["retrieved_chunks"]],
                "retrieved_chunks": response["retrieved_chunks"],
                "citations": response["citations"],
                "citation_valid": response["citation_valid"],
                "refused": response["refused"],
                "latency_seconds": response["latency_seconds"],
                "latency_breakdown": response.get("latency_breakdown", {}),
                "llm": response.get("llm", {}),
                "routing": response.get("routing"),
                "scores": scores,
                "error": error,
            }
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            results.append(row)
    summary = summarize(results, settings, run_name, questions_path)
    destination = summary_path or output_path.with_suffix(".summary.json")
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return results

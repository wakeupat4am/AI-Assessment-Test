#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.llm.base import GenerationResult, aggregate_calls
from src.routing.llm_xrouter import LLMAgentXRouter
from src.routing.xrouter import RuleBasedXRouter


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def summarize(rows: list[dict[str, Any]], calls: list[GenerationResult]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    per_class: dict[str, dict[str, int | float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["expected_route"]].append(row)
    for route, route_rows in sorted(grouped.items()):
        route_correct = sum(bool(row["correct"]) for row in route_rows)
        per_class[route] = {
            "questions": len(route_rows),
            "correct": route_correct,
            "accuracy": round(route_correct / len(route_rows), 6),
        }
    latencies = [float(row["latency_seconds"]) for row in rows]
    fallbacks = Counter(row["decision_source"] for row in rows)
    return {
        "questions": total,
        "correct": correct,
        "route_accuracy": round(correct / total, 6) if total else None,
        "per_class": per_class,
        "confusion": dict(
            sorted(
                Counter(
                    f"{row['expected_route']}->{row['selected_route']}" for row in rows
                ).items()
            )
        ),
        "route_distribution": dict(
            sorted(Counter(row["selected_route"] for row in rows).items())
        ),
        "latency_seconds_mean": round(statistics.mean(latencies), 6),
        "latency_seconds_p50": percentile(latencies, 0.50),
        "latency_seconds_p95": percentile(latencies, 0.95),
        "decision_source": dict(sorted(fallbacks.items())),
        "llm": aggregate_calls(calls),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare B1a rules with B1b LLM routing")
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data/evaluation/router/b1_route_labels.json",
    )
    parser.add_argument(
        "--b1a-config", type=Path, default=ROOT / "config/xrouter_b1a.json"
    )
    parser.add_argument(
        "--b1b-config", type=Path, default=ROOT / "config/xrouter_b1b.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/evaluation/experiments/b1b/b1b-route-eval.json",
    )
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    settings = get_settings()
    b1a = RuleBasedXRouter(args.b1a_config)
    b1b = LLMAgentXRouter(args.b1b_config, settings=settings)
    arm_rows: dict[str, list[dict[str, Any]]] = {"b1a": [], "b1b": []}
    b1b_calls: list[GenerationResult] = []
    for item in labels:
        for name, router in (("b1a", b1a), ("b1b", b1b)):
            started = __import__("time").perf_counter()
            decision = router.route(item["query"], item.get("conversation_context"))
            elapsed = __import__("time").perf_counter() - started
            metadata = getattr(router, "last_metadata", {})
            arm_rows[name].append(
                {
                    **item,
                    "selected_route": decision["route"],
                    "confidence": decision["confidence"],
                    "correct": decision["route"] == item["expected_route"],
                    "reasons": decision["reasons"],
                    "latency_seconds": round(elapsed, 6),
                    "decision_source": metadata.get("decision_source", "rules"),
                }
            )
            if name == "b1b" and b1b.last_call is not None:
                b1b_calls.append(b1b.last_call)

    payload = {
        "artifact_schema_version": 1,
        "experiment": "B1b route-decision evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels": str(args.labels),
        "b1a": summarize(arm_rows["b1a"], []),
        "b1b": summarize(arm_rows["b1b"], b1b_calls),
        "rows": arm_rows,
    }
    payload["delta_b1b_minus_b1a"] = {
        "route_accuracy": round(
            payload["b1b"]["route_accuracy"] - payload["b1a"]["route_accuracy"], 6
        ),
        "latency_seconds_p50": round(
            payload["b1b"]["latency_seconds_p50"]
            - payload["b1a"]["latency_seconds_p50"],
            6,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ("b1a", "b1b", "delta_b1b_minus_b1a")}, ensure_ascii=False, indent=2))
    print(f"Artifact: {args.output}")


if __name__ == "__main__":
    main()

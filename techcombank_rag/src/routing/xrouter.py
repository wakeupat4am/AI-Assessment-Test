from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROUTES = {"narrative", "structured", "multi_repr", "multi_hop"}
FEATURE_NAMES = (
    "has_number",
    "has_year",
    "has_financial_term",
    "has_comparison",
    "has_ratio_or_growth",
    "has_table_lookup_pattern",
    "is_explanatory",
    "is_followup",
)


def load_router_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported X-Router config schema_version")
    if set(config.get("routes", {})) != ROUTES:
        raise ValueError(f"X-Router config must define exactly {sorted(ROUTES)}")
    fallback = str(config.get("fallback_route", ""))
    if fallback not in ROUTES:
        raise ValueError(f"Invalid fallback route: {fallback}")
    priority = config.get("route_priority", [])
    if set(priority) != ROUTES or len(priority) != len(ROUTES):
        raise ValueError("route_priority must contain every route exactly once")
    threshold = float(config.get("confidence_threshold", -1))
    if not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    return config


class RuleBasedXRouter:
    """Config-driven deterministic B1a router.

    The module only classifies a query. It never calls an LLM and never changes
    the answer-generation prompt.
    """

    def __init__(
        self,
        config_path: Path,
        *,
        confidence_threshold: float | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = load_router_config(self.config_path)
        configured_threshold = float(self.config["confidence_threshold"])
        self.confidence_threshold = (
            configured_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.log_path = Path(log_path) if log_path else None
        self.config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        self._log_lock = threading.Lock()

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.casefold()).strip()

    @staticmethod
    def _term_matches(text: str, terms: list[str]) -> list[str]:
        return [term for term in terms if term.casefold() in text]

    def extract_features(
        self, query: str, conversation_context: str | None = None
    ) -> tuple[dict[str, bool], dict[str, int]]:
        normalized = self._normalize(query)
        context = self._normalize(conversation_context or "")
        feature_config = self.config["features"]
        financial = self._term_matches(normalized, feature_config["financial_terms"])
        entities = self._term_matches(normalized, feature_config["entity_terms"])
        followup_markers = self._term_matches(
            normalized, feature_config["followup_terms"]
        )
        is_short = len(normalized.split()) <= int(
            feature_config.get("followup_max_tokens", 6)
        )
        features = {
            "has_number": bool(re.search(feature_config["number_regex"], normalized)),
            "has_year": bool(re.search(feature_config["year_regex"], normalized)),
            "has_financial_term": bool(financial),
            "has_comparison": bool(
                self._term_matches(normalized, feature_config["comparison_terms"])
            ),
            "has_ratio_or_growth": bool(
                self._term_matches(normalized, feature_config["ratio_or_growth_terms"])
            ),
            "has_table_lookup_pattern": bool(
                self._term_matches(normalized, feature_config["table_lookup_terms"])
            ),
            "is_explanatory": bool(
                self._term_matches(normalized, feature_config["explanatory_terms"])
            ),
            "is_followup": bool(context) and (bool(followup_markers) or is_short),
        }
        counts = {
            "financial_terms": len(financial),
            "entity_terms": len(entities),
        }
        return features, counts

    @staticmethod
    def _rule_matches(
        rule: dict[str, Any], features: dict[str, bool], counts: dict[str, int]
    ) -> bool:
        if any(not features.get(name, False) for name in rule.get("all", [])):
            return False
        any_features = rule.get("any", [])
        if any_features and not any(features.get(name, False) for name in any_features):
            return False
        if any(features.get(name, False) for name in rule.get("none", [])):
            return False
        if any(
            counts.get(name, 0) < int(minimum)
            for name, minimum in rule.get("min_counts", {}).items()
        ):
            return False
        return True

    def _route(self, query: str, conversation_context: str | None) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        features, counts = self.extract_features(query, conversation_context)
        route_scores: dict[str, float] = {}
        route_reasons: dict[str, list[str]] = {}
        for route, route_config in self.config["routes"].items():
            score = float(route_config.get("base_score", 0))
            reasons: list[str] = []
            for rule in route_config.get("rules", []):
                if self._rule_matches(rule, features, counts):
                    score += float(rule.get("add", 0))
                    if rule.get("reason"):
                        reasons.append(str(rule["reason"]))
            route_scores[route] = min(1.0, max(0.0, score))
            route_reasons[route] = reasons

        priority = {route: index for index, route in enumerate(self.config["route_priority"])}
        selected = min(
            route_scores,
            key=lambda route: (-route_scores[route], priority[route]),
        )
        confidence = route_scores[selected]
        reasons = list(route_reasons[selected])
        if confidence < self.confidence_threshold:
            original = selected
            selected = str(self.config["fallback_route"])
            reasons = [
                *reasons,
                (
                    f"confidence {confidence:.3f} below threshold "
                    f"{self.confidence_threshold:.3f}; fallback from {original}"
                ),
            ]
        if not reasons:
            reasons = ["no strong configured intent signal; deterministic fallback"]
        return {
            "route": selected,
            "confidence": round(confidence, 6),
            "reasons": reasons,
            "query_features": {name: bool(features[name]) for name in FEATURE_NAMES},
        }

    def route(
        self, query: str, conversation_context: str | None = None
    ) -> dict[str, Any]:
        started = time.perf_counter()
        decision = self._route(query, conversation_context)
        latency_seconds = time.perf_counter() - started
        if self.log_path:
            payload = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiment": self.config.get("experiment", "B1a"),
                "query": query,
                "conversation_context_present": bool(conversation_context),
                "selected_route": decision["route"],
                "confidence": decision["confidence"],
                "reasons": decision["reasons"],
                "features": decision["query_features"],
                "latency_seconds": round(latency_seconds, 9),
                "config_sha256": self.config_sha256,
            }
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_lock, self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return decision

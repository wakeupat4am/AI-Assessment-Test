from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.llm.base import GenerationResult, coerce_generation_result
from src.llm.provider import create_llm_client
from src.routing.xrouter import FEATURE_NAMES, ROUTES, RuleBasedXRouter, load_router_config


class LLMAgentXRouter:
    """B1b constrained LLM router with a stable B1a-compatible contract.

    The model makes exactly one route decision before retrieval. It returns
    concise decision evidence rather than an unrestricted chain of thought.
    Invalid output, timeout, and low confidence all fail closed to multi_repr.
    """

    def __init__(
        self,
        config_path: Path,
        *,
        settings: Settings | None = None,
        llm_client: Any | None = None,
        confidence_threshold: float | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config_path = Path(config_path)
        self.config = load_router_config(self.config_path)
        if "llm_router" not in self.config:
            raise ValueError("B1b config must define llm_router")
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
        self._feature_extractor = RuleBasedXRouter(self.config_path)
        self.llm = llm_client or create_llm_client(self._router_settings())
        self.last_call: GenerationResult | None = None
        self.last_metadata: dict[str, Any] = {}

    def _router_settings(self) -> Settings:
        if not self.settings.router_llm_model:
            raise ValueError("ROUTER_LLM_MODEL (or LLM_MODEL) is required for B1b")
        return replace(
            self.settings,
            llm_provider=self.settings.router_llm_provider,
            llm_base_url=self.settings.router_llm_base_url,
            llm_model=self.settings.router_llm_model,
            llm_model_fast=self.settings.router_llm_model,
            llm_model_strong=self.settings.router_llm_model,
            llm_api_key=self.settings.router_llm_api_key,
            llm_temperature=self.settings.router_llm_temperature,
            llm_timeout_seconds=self.settings.router_llm_timeout_seconds,
            max_new_tokens=self.settings.router_llm_max_new_tokens,
        )

    def _messages(
        self, query: str, conversation_context: str | None
    ) -> list[dict[str, str]]:
        llm_config = self.config["llm_router"]
        examples = json.dumps(
            llm_config.get("few_shot_examples", []),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        context = conversation_context.strip() if conversation_context else "(none)"
        user_prompt = str(llm_config["user_prompt_template"]).format(
            query=query.strip(),
            conversation_context=context,
            examples=examples,
        )
        return [
            {"role": "system", "content": str(llm_config["system_prompt"])},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise ValueError("router response does not contain a JSON object")
            value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("router response must be a JSON object")
        return value

    @staticmethod
    def _validate_decision(
        value: dict[str, Any], extracted_features: dict[str, bool]
    ) -> dict[str, Any]:
        route = value.get("route")
        if route not in ROUTES:
            raise ValueError(f"invalid route: {route!r}")
        confidence = value.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        reasons = value.get("reasons")
        contract_repairs: list[str] = []
        if isinstance(reasons, str) and reasons.strip():
            reasons = [reasons]
            contract_repairs.append("coerced reasons string to one-item list")
        if not isinstance(reasons, list) or not reasons or not all(
            isinstance(reason, str) and reason.strip() for reason in reasons
        ):
            raise ValueError("reasons must be a non-empty string list")
        # B1b deliberately asks the LLM only for the actual decision fields.
        # Deterministic features are attached afterward to keep the public B1
        # contract stable while cutting output tokens and JSON failure risk.
        features = value.get("query_features", extracted_features)
        if not isinstance(features, dict) or set(features) != set(FEATURE_NAMES):
            raise ValueError("query_features must contain the exact B1 feature set")
        if not all(isinstance(features[name], bool) for name in FEATURE_NAMES):
            raise ValueError("all query_features values must be boolean")
        return {
            "route": str(route),
            "confidence": round(confidence, 6),
            "reasons": [reason.strip() for reason in reasons[:4]],
            "query_features": {name: features[name] for name in FEATURE_NAMES},
            "_contract_repairs": contract_repairs,
        }

    def _fallback(
        self,
        query: str,
        conversation_context: str | None,
        reason: str,
        *,
        confidence: float = 0.0,
    ) -> dict[str, Any]:
        features, _ = self._feature_extractor.extract_features(
            query, conversation_context
        )
        return {
            "route": str(self.config["fallback_route"]),
            "confidence": round(confidence, 6),
            "reasons": [reason],
            "query_features": {name: bool(features[name]) for name in FEATURE_NAMES},
        }

    def route(
        self, query: str, conversation_context: str | None = None
    ) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        started = time.perf_counter()
        self.last_call = None
        decision_source = "llm"
        raw_route: str | None = None
        fallback_reason: str | None = None
        extracted_features, _ = self._feature_extractor.extract_features(
            query, conversation_context
        )
        try:
            generated = self.llm.generate(
                self._messages(query, conversation_context),
                temperature=self.settings.router_llm_temperature,
                purpose="route",
                model=self.settings.router_llm_model or None,
            )
            self.last_call = coerce_generation_result(generated, purpose="route")
            decision = self._validate_decision(
                self._extract_json(self.last_call.text), extracted_features
            )
            contract_repairs = decision.pop("_contract_repairs")
            if contract_repairs:
                decision_source = "llm_normalized"
            raw_route = decision["route"]
            if decision["confidence"] < self.confidence_threshold:
                fallback_reason = (
                    f"confidence {decision['confidence']:.3f} below threshold "
                    f"{self.confidence_threshold:.3f}; fallback from {raw_route}"
                )
                decision = {
                    **decision,
                    "route": str(self.config["fallback_route"]),
                    "reasons": [*decision["reasons"], fallback_reason],
                }
                decision_source = "low_confidence_fallback"
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            decision = self._fallback(
                query,
                conversation_context,
                f"LLM router failure; safe fallback: {fallback_reason}",
            )
            decision_source = "error_fallback"

        latency_seconds = time.perf_counter() - started
        self.last_metadata = {
            "decision_source": decision_source,
            "raw_route": raw_route,
            "fallback_reason": fallback_reason,
            "contract_repairs": contract_repairs if "contract_repairs" in locals() else [],
            "latency_seconds": round(latency_seconds, 9),
        }
        if self.log_path:
            payload = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiment": self.config.get("experiment", "B1b"),
                "query": query,
                "conversation_context_present": bool(conversation_context),
                "selected_route": decision["route"],
                "raw_route": raw_route,
                "confidence": decision["confidence"],
                "reasons": decision["reasons"],
                "features": decision["query_features"],
                "decision_source": decision_source,
                "fallback_reason": fallback_reason,
                "contract_repairs": self.last_metadata["contract_repairs"],
                "latency_seconds": round(latency_seconds, 9),
                "llm_call": self.last_call.to_dict() if self.last_call else None,
                "config_sha256": self.config_sha256,
            }
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_lock, self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return decision

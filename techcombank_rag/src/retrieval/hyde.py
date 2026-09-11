from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.llm.base import GenerationResult, coerce_generation_result
from src.llm.provider import create_llm_client


MODES = {"hyde_only", "fusion"}
PROMPT_VARIANTS = {"paper_faithful", "finance_safe"}
NUMERIC_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:[.,]\d{1,3})+|\d+)(?:\s*%|x)?", re.IGNORECASE)


def _number_key(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def redact_invented_numbers(query: str, hypothetical: str) -> str:
    """Replace numbers absent from the query in finance-safe HyDE text."""
    allowed = {_number_key(value) for value in NUMERIC_RE.findall(query)}

    def replace_number(match: re.Match[str]) -> str:
        value = match.group(0)
        return value if _number_key(value) in allowed else "[VALUE]"

    return NUMERIC_RE.sub(replace_number, hypothetical)


def load_hyde_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    prompts = value.get("prompts")
    if not isinstance(prompts, dict) or not PROMPT_VARIANTS.issubset(prompts):
        raise ValueError("HyDE config must define paper_faithful and finance_safe prompts")
    for name in PROMPT_VARIANTS:
        prompt = prompts[name]
        if not isinstance(prompt, dict) or not {
            "system_prompt",
            "user_prompt_template",
        }.issubset(prompt):
            raise ValueError(f"HyDE prompt {name} is incomplete")
    return value


class HyDERetriever:
    """Generate one hypothetical passage and retrieve real A3.1 children.

    Hypothetical text only creates a search vector. Returned evidence always
    comes from the shipped index, so the answer layer cannot cite generated text.
    """

    def __init__(
        self,
        retriever: Any,
        *,
        settings: Settings | None = None,
        llm_client: Any | None = None,
        config_path: Path | None = None,
        mode: str | None = None,
        prompt_variant: str | None = None,
        candidate_k: int | None = None,
        rrf_k: int | None = None,
        original_weight: float | None = None,
        hypothetical_weight: float | None = None,
        cache_size: int | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.base = retriever
        self.settings = settings or get_settings()
        self.mode = mode or self.settings.hyde_mode
        self.prompt_variant = prompt_variant or self.settings.hyde_prompt_variant
        if self.mode not in MODES:
            raise ValueError(f"HyDE mode must be one of {sorted(MODES)}")
        if self.prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"HyDE prompt variant must be one of {sorted(PROMPT_VARIANTS)}"
            )
        self.config_path = Path(config_path or self.settings.hyde_config_path)
        self.config = load_hyde_config(self.config_path)
        self.config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        self.candidate_k = int(candidate_k or self.settings.hyde_candidate_k)
        self.rrf_k = int(rrf_k or self.settings.hyde_rrf_k)
        self.original_weight = float(
            self.settings.hyde_original_weight
            if original_weight is None
            else original_weight
        )
        self.hypothetical_weight = float(
            self.settings.hyde_hypothetical_weight
            if hypothetical_weight is None
            else hypothetical_weight
        )
        self.cache_size = max(
            0, self.settings.hyde_cache_size if cache_size is None else int(cache_size)
        )
        if self.candidate_k <= 0 or self.rrf_k <= 0:
            raise ValueError("HyDE candidate_k and rrf_k must be positive")
        if self.original_weight < 0 or self.hypothetical_weight <= 0:
            raise ValueError("HyDE fusion weights are invalid")
        if not hasattr(self.base, "retrieve_by_vector") or not hasattr(
            self.base, "embedder"
        ):
            raise TypeError("HyDERetriever requires a DenseRetriever-compatible base")
        self.llm = llm_client or create_llm_client(self._hyde_settings())
        self.log_path = Path(log_path) if log_path else None
        self._cache: OrderedDict[str, tuple[str, Any]] = OrderedDict()
        self._log_lock = threading.Lock()
        self.last_trace: dict[str, Any] | None = None
        self.last_retrieval_calls: list[GenerationResult] = []
        self.last_router_call = None

    def _hyde_settings(self) -> Settings:
        if not self.settings.hyde_llm_model:
            raise ValueError("HYDE_MODEL (or LLM_MODEL) is required")
        return replace(
            self.settings,
            llm_provider=self.settings.hyde_llm_provider,
            llm_base_url=self.settings.hyde_llm_base_url,
            llm_model=self.settings.hyde_llm_model,
            llm_model_fast=self.settings.hyde_llm_model,
            llm_model_strong=self.settings.hyde_llm_model,
            llm_api_key=self.settings.hyde_llm_api_key,
            llm_temperature=self.settings.hyde_llm_temperature,
            llm_timeout_seconds=self.settings.hyde_llm_timeout_seconds,
            max_new_tokens=self.settings.hyde_llm_max_new_tokens,
        )

    def _messages(self, query: str) -> list[dict[str, str]]:
        prompt = self.config["prompts"][self.prompt_variant]
        return [
            {"role": "system", "content": str(prompt["system_prompt"])},
            {
                "role": "user",
                "content": str(prompt["user_prompt_template"]).format(query=query),
            },
        ]

    def _cache_get(self, key: str) -> tuple[str, Any] | None:
        if not self.cache_size or key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def _cache_put(self, key: str, value: tuple[str, Any]) -> None:
        if not self.cache_size:
            return
        self._cache[key] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _hypothetical_vector(self, query: str) -> tuple[str, Any, bool]:
        key = f"{self.prompt_variant}:{' '.join(query.casefold().split())}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached[0], cached[1], True
        generated = self.llm.generate(
            self._messages(query),
            temperature=self.settings.hyde_llm_temperature,
            purpose="hyde_generation",
            model=self.settings.hyde_llm_model or None,
        )
        call = coerce_generation_result(generated, purpose="hyde_generation")
        self.last_retrieval_calls.append(call)
        hypothetical = call.text.strip().strip("`").strip()
        if self.prompt_variant == "finance_safe":
            hypothetical = redact_invented_numbers(query, hypothetical)
        if not hypothetical:
            raise ValueError("HyDE generation is empty")
        vector = self.base.embedder.encode_documents([hypothetical])
        self._cache_put(key, (hypothetical, vector))
        return hypothetical, vector, False

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        requested_k = int(top_k or self.settings.top_k)
        if not query.strip() or requested_k <= 0:
            return []
        started = time.perf_counter()
        self.last_retrieval_calls = []
        hypothetical = ""
        cache_hit = False
        error: str | None = None
        original_rows: list[dict[str, Any]] = []
        hyde_rows: list[dict[str, Any]] = []
        try:
            if self.mode == "fusion":
                original_rows = self.base.retrieve(query, self.candidate_k)
            hypothetical, vector, cache_hit = self._hypothetical_vector(query)
            hyde_rows = self.base.retrieve_by_vector(vector, self.candidate_k)
            if self.mode == "hyde_only":
                results = [
                    {**row, "retrieval_source": "hyde", "hyde_rank": rank}
                    for rank, row in enumerate(hyde_rows[:requested_k], 1)
                ]
            else:
                results = self._fuse(original_rows, hyde_rows, requested_k)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            original_rows = original_rows or self.base.retrieve(query, self.candidate_k)
            results = [
                {**row, "retrieval_source": "original_fallback"}
                for row in original_rows[:requested_k]
            ]
        latency = time.perf_counter() - started
        trace = {
            "mode": self.mode,
            "prompt_variant": self.prompt_variant,
            "query": query,
            "hypothetical_document": hypothetical,
            "cache_hit": cache_hit,
            "fallback": error is not None,
            "error": error,
            "original_candidates": len(original_rows),
            "hyde_candidates": len(hyde_rows),
            "returned_candidates": len(results),
            "final_pages": [int(row["printed_page"]) for row in results],
            "latency_seconds": round(latency, 6),
            "config_sha256": self.config_sha256,
        }
        self.last_trace = {"hyde": trace}
        if self.log_path:
            payload = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                **trace,
                "llm_call": (
                    self.last_retrieval_calls[0].to_dict()
                    if self.last_retrieval_calls
                    else None
                ),
            }
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_lock, self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return results

    def _fuse(
        self,
        original_rows: list[dict[str, Any]],
        hyde_rows: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        rankings = (
            ("original", original_rows, self.original_weight),
            ("hyde", hyde_rows, self.hypothetical_weight),
        )
        for source, rows, weight in rankings:
            for rank, row in enumerate(rows, 1):
                key = str(row["chunk_id"])
                entry = fused.setdefault(
                    key,
                    {
                        "row": dict(row),
                        "fusion_score": 0.0,
                        "dense_score": float(row.get("score", 0.0)),
                        "sources": [],
                    },
                )
                entry["fusion_score"] += weight / (self.rrf_k + rank)
                entry["dense_score"] = max(
                    float(entry["dense_score"]), float(row.get("score", 0.0))
                )
                entry["sources"].append({"source": source, "rank": rank})
        ordered = sorted(
            fused.values(),
            key=lambda item: (item["fusion_score"], item["dense_score"]),
            reverse=True,
        )
        output: list[dict[str, Any]] = []
        for entry in ordered[:top_k]:
            output.append(
                {
                    **entry["row"],
                    "score": float(entry["dense_score"]),
                    "fusion_score": round(float(entry["fusion_score"]), 9),
                    "retrieval_source": "original+hyde",
                    "retrieval_sources": entry["sources"],
                }
            )
        return output

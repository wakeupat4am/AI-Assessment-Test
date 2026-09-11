from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import Settings
from src.llm.base import GenerationResult, TokenUsage
from src.routing.llm_xrouter import LLMAgentXRouter
from src.routing.xrouter import FEATURE_NAMES


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/xrouter_b1b.json"


def decision_json(route: str, confidence: float = 0.91) -> str:
    features = {name: False for name in FEATURE_NAMES}
    return json.dumps(
        {
            "route": route,
            "confidence": confidence,
            "reasons": [f"test evidence for {route}"],
            "query_features": features,
        }
    )


def compact_decision_json(route: str, confidence: float = 0.91) -> str:
    return json.dumps(
        {"route": route, "confidence": confidence, "reasons": ["short reason"]}
    )


class FakeLLM:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate(self, messages, temperature=0, purpose="answer", model=None):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "purpose": purpose,
                "model": model,
            }
        )
        if isinstance(self.response, Exception):
            raise self.response
        return GenerationResult(
            text=self.response,
            provider="test",
            model="qwen-test",
            purpose=purpose,
            latency_seconds=0.01,
            usage=TokenUsage(uncached_input_tokens=20, output_tokens=12),
            cost_usd=0.0,
        )


@pytest.mark.parametrize(
    "route",
    [
        "structured", "narrative", "multi_hop", "multi_repr",
        "structured", "narrative", "multi_hop", "multi_repr",
        "structured", "narrative", "multi_hop", "multi_repr",
        "structured", "narrative", "multi_hop", "multi_repr",
        "structured", "narrative", "multi_hop", "multi_repr",
    ],
)
def test_twenty_constrained_route_decisions(route: str) -> None:
    fake = FakeLLM(decision_json(route))
    router = LLMAgentXRouter(CONFIG_PATH, settings=Settings(), llm_client=fake)
    result = router.route("Một câu hỏi tiếng Việt đại diện")
    assert result["route"] == route
    assert set(result) == {"route", "confidence", "reasons", "query_features"}
    assert fake.calls[0]["purpose"] == "route"
    assert fake.calls[0]["temperature"] == 0
    assert router.last_call is not None


def test_prompt_includes_query_and_conversation_context() -> None:
    fake = FakeLLM(decision_json("multi_repr"))
    router = LLMAgentXRouter(CONFIG_PATH, settings=Settings(), llm_client=fake)
    router.route("Còn năm trước?", "Trước đó người dùng hỏi CASA năm 2025")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "Còn năm trước?" in prompt
    assert "CASA năm 2025" in prompt


def test_compact_llm_output_gets_full_public_feature_contract() -> None:
    router = LLMAgentXRouter(
        CONFIG_PATH,
        settings=Settings(),
        llm_client=FakeLLM(compact_decision_json("structured")),
    )
    result = router.route("Lợi nhuận năm 2025 là bao nhiêu?")
    assert result["route"] == "structured"
    assert set(result["query_features"]) == set(FEATURE_NAMES)
    assert result["query_features"]["has_year"] is True


def test_low_confidence_falls_back_to_multi_repr() -> None:
    fake = FakeLLM(decision_json("structured", confidence=0.4))
    router = LLMAgentXRouter(CONFIG_PATH, settings=Settings(), llm_client=fake)
    result = router.route("Lợi nhuận là bao nhiêu?")
    assert result["route"] == "multi_repr"
    assert router.last_metadata["decision_source"] == "low_confidence_fallback"


@pytest.mark.parametrize(
    "bad_response",
    [
        "not json",
        "{}",
        '{"route":"unknown"}',
        decision_json("structured").replace('"confidence": 0.91', '"confidence": 2'),
    ],
)
def test_invalid_output_fails_closed(bad_response: str) -> None:
    router = LLMAgentXRouter(
        CONFIG_PATH, settings=Settings(), llm_client=FakeLLM(bad_response)
    )
    result = router.route("Thông tin về CASA")
    assert result["route"] == "multi_repr"
    assert result["confidence"] == 0
    assert router.last_metadata["decision_source"] == "error_fallback"


def test_timeout_fails_closed() -> None:
    router = LLMAgentXRouter(
        CONFIG_PATH,
        settings=Settings(),
        llm_client=FakeLLM(TimeoutError("router timeout")),
    )
    result = router.route("Chiến lược ngân hàng số là gì?")
    assert result["route"] == "multi_repr"
    assert "safe fallback" in result["reasons"][0]


def test_router_log_captures_usage_and_fallback(tmp_path: Path) -> None:
    log_path = tmp_path / "b1b-router.jsonl"
    router = LLMAgentXRouter(
        CONFIG_PATH,
        settings=Settings(),
        llm_client=FakeLLM(decision_json("narrative")),
        log_path=log_path,
    )
    router.route("Tại sao CASA quan trọng?")
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["experiment"] == "B1b"
    assert payload["selected_route"] == "narrative"
    assert payload["llm_call"]["purpose"] == "route"
    assert payload["llm_call"]["usage"]["total_tokens"] == 32

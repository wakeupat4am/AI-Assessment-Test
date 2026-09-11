from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TokenUsage:
    uncached_input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    source: str = "provider"

    @property
    def total_input_tokens(self) -> int:
        return (
            self.uncached_input_tokens
            + self.cached_input_tokens
            + self.cache_write_input_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_input_tokens"] = self.total_input_tokens
        payload["total_tokens"] = self.total_tokens
        return payload


@dataclass(frozen=True)
class GenerationResult:
    text: str
    provider: str = "unknown"
    model: str = "unknown"
    purpose: str = "answer"
    latency_seconds: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float | None = None
    pricing_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["usage"] = self.usage.to_dict()
        return payload


class LLMClient(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0,
        purpose: str = "answer",
        model: str | None = None,
    ) -> GenerationResult: ...


def coerce_generation_result(
    value: GenerationResult | str,
    *,
    purpose: str,
    provider: str = "test-double",
    model: str = "test-double",
) -> GenerationResult:
    """Keep lightweight test doubles and downstream adapters easy to use."""
    if isinstance(value, GenerationResult):
        return value
    return GenerationResult(
        text=str(value).strip(), provider=provider, model=model, purpose=purpose
    )


def aggregate_calls(calls: list[GenerationResult]) -> dict[str, Any]:
    sources = {call.usage.source for call in calls}
    usage = TokenUsage(
        uncached_input_tokens=sum(c.usage.uncached_input_tokens for c in calls),
        cached_input_tokens=sum(c.usage.cached_input_tokens for c in calls),
        cache_write_input_tokens=sum(c.usage.cache_write_input_tokens for c in calls),
        output_tokens=sum(c.usage.output_tokens for c in calls),
        reasoning_tokens=sum(c.usage.reasoning_tokens for c in calls),
        source=(
            "unavailable"
            if not sources
            else next(iter(sources))
            if len(sources) == 1
            else "mixed"
        ),
    )
    known_costs = [c.cost_usd for c in calls if c.cost_usd is not None]
    return {
        "call_count": len(calls),
        "usage": usage.to_dict(),
        "cost_usd": round(sum(known_costs), 8) if len(known_costs) == len(calls) else None,
        "calls": [call.to_dict() for call in calls],
    }

from __future__ import annotations

from dataclasses import dataclass

from src.llm.base import TokenUsage


PRICING_SNAPSHOT_DATE = "2026-09-09"


@dataclass(frozen=True)
class PriceRates:
    input_per_million: float
    cached_input_per_million: float
    cache_write_per_million: float
    output_per_million: float
    source: str


# Public list prices as of the snapshot date. Environment overrides always win.
# Local open-weight inference has no API charge and is reported with hardware
# plus wall-clock time separately.
KNOWN_PRICES: dict[str, PriceRates] = {
    "gpt-5.6-sol": PriceRates(4.0, 0.4, 5.0, 20.0, "OpenAI pricing 2026-09-09"),
    "gpt-5.6": PriceRates(4.0, 0.4, 5.0, 20.0, "OpenAI pricing 2026-09-09"),
    "gpt-5.6-terra": PriceRates(2.0, 0.2, 2.5, 12.0, "OpenAI pricing 2026-09-09"),
    "gpt-5.6-luna": PriceRates(0.2, 0.02, 0.25, 1.2, "OpenAI pricing 2026-09-09"),
    "claude-sonnet-5": PriceRates(3.0, 0.3, 3.75, 15.0, "Anthropic pricing 2026-09-09"),
    "claude-opus-5": PriceRates(5.0, 0.5, 6.25, 25.0, "Anthropic pricing 2026-09-09"),
    "claude-fable-5": PriceRates(10.0, 1.0, 12.5, 50.0, "Anthropic pricing 2026-09-09"),
}


def resolve_rates(
    model: str,
    *,
    input_override: float | None = None,
    cached_input_override: float | None = None,
    cache_write_override: float | None = None,
    output_override: float | None = None,
    local: bool = False,
) -> PriceRates | None:
    if local:
        return PriceRates(0.0, 0.0, 0.0, 0.0, "local inference; API cost only")
    known = KNOWN_PRICES.get(model.lower())
    values = (
        input_override,
        cached_input_override,
        cache_write_override,
        output_override,
    )
    if any(value is not None for value in values):
        if input_override is None or output_override is None:
            raise ValueError(
                "Pricing overrides require both LLM_INPUT_PRICE_PER_MILLION and "
                "LLM_OUTPUT_PRICE_PER_MILLION."
            )
        return PriceRates(
            input_override,
            cached_input_override if cached_input_override is not None else input_override,
            cache_write_override if cache_write_override is not None else input_override,
            output_override,
            f"environment override {PRICING_SNAPSHOT_DATE}",
        )
    return known


def calculate_cost(usage: TokenUsage, rates: PriceRates | None) -> float | None:
    if rates is None or usage.source == "unavailable":
        return None
    cost = (
        usage.uncached_input_tokens * rates.input_per_million
        + usage.cached_input_tokens * rates.cached_input_per_million
        + usage.cache_write_input_tokens * rates.cache_write_per_million
        + usage.output_tokens * rates.output_per_million
    ) / 1_000_000
    return round(cost, 8)

from __future__ import annotations

from decimal import Decimal

from src.llm.base import GenerationResult, TokenUsage
from src.retrieval.tool_agent import ToolCallingRetriever, format_vietnamese_decimal, parse_vietnamese_decimal


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None):
        self.calls.append((query, top_k))
        if len(self.calls) > 1 and "2024" in query:
            return [{"chunk_id": "old", "text": "Năm 2024 là 27.538 tỷ đồng", "printed_page": 7, "pdf_page": 7, "score": 0.8, "granularity": "row"}]
        return [
            {"chunk_id": "new", "text": "Năm 2025 là 32.538 tỷ đồng; năm 2024 là 27.538 tỷ đồng", "printed_page": 7, "pdf_page": 7, "score": 0.9, "granularity": "row"},
        ]


class FakeLLM:
    def __init__(self, output: str) -> None:
        self.output = output

    def generate(self, messages, temperature=0, purpose="answer", model=None):
        del messages, temperature, model
        return GenerationResult(
            text=self.output,
            provider="test",
            model="test",
            purpose=purpose,
            usage=TokenUsage(uncached_input_tokens=10, output_tokens=5),
        )


CONFIG = {
    "agent_protocol": "test",
    "initial_top_k": 20,
    "followup_top_k": 12,
    "max_tool_actions_after_initial_retrieval": 1,
    "evidence_preview_chunks": 2,
    "evidence_preview_characters_per_chunk": 300,
    "trigger_patterns": ["tăng thêm", "so với"],
    "calculation_patterns": ["tăng thêm"],
    "allowed_operations": ["subtract", "percentage_change", "cagr"],
    "tool_names": ["hybrid_retrieve", "calculator", "finish"],
}


def test_vietnamese_number_parsing_and_formatting() -> None:
    assert parse_vietnamese_decimal("27.538") == 27538
    assert parse_vietnamese_decimal("1,13%") == Decimal("1.13")
    assert format_vietnamese_decimal(parse_vietnamese_decimal("32.538") - parse_vietnamese_decimal("27.538")) == "5.000"


def test_calculator_tool_only_uses_verbatim_evidence_operands() -> None:
    retriever = ToolCallingRetriever(
        FakeRetriever(),
        FakeLLM('{"tool":"calculator","operation":"subtract","left":{"value":"32.538","page":7},"right":{"value":"27.538","page":7},"unit":"tỷ đồng"}'),
        config=CONFIG,
    )
    rows = retriever.retrieve("Lợi nhuận tăng thêm bao nhiêu?", 3)
    assert rows[0]["derived"] is True
    assert "5.000 tỷ đồng" in rows[0]["text"]
    assert retriever.last_trace["b5"]["actions"][1]["outcome"] == "ok"
    assert retriever.last_retrieval_calls[0].purpose == "agent_planning"


def test_invalid_tool_arguments_do_not_create_derived_evidence() -> None:
    retriever = ToolCallingRetriever(
        FakeRetriever(),
        FakeLLM('{"tool":"calculator","operation":"subtract","left":{"value":"999","page":7},"right":{"value":"27.538","page":7}}'),
        config=CONFIG,
    )
    rows = retriever.retrieve("Lợi nhuận tăng thêm bao nhiêu?", 2)
    assert all(not row.get("derived") for row in rows)
    assert "rejected" in retriever.last_trace["b5"]["actions"][1]["outcome"]


def test_agent_can_call_a_second_focused_retrieval() -> None:
    base = FakeRetriever()
    retriever = ToolCallingRetriever(
        base,
        FakeLLM('{"tool":"hybrid_retrieve","query":"lợi nhuận trước thuế năm 2024"}'),
        config=CONFIG,
    )
    rows = retriever.retrieve("Lợi nhuận 2025 so với 2024 thế nào?", 3)
    assert len(base.calls) == 2
    assert base.calls[1][0].endswith("2024")
    assert {row["chunk_id"] for row in rows} == {"new", "old"}


def test_agent_accepts_nested_tool_json_from_openai_compatible_model() -> None:
    base = FakeRetriever()
    retriever = ToolCallingRetriever(
        base,
        FakeLLM('{"hybrid_retrieve":{"query":"lợi nhuận trước thuế năm 2024"}}'),
        config=CONFIG,
    )
    retriever.retrieve("Lợi nhuận 2025 so với 2024 thế nào?", 3)
    assert len(base.calls) == 2
    assert retriever.last_trace["b5"]["planner_source"] == "planner_nested_tool_json"


def test_non_trigger_query_avoids_planner_call() -> None:
    retriever = ToolCallingRetriever(FakeRetriever(), FakeLLM('{"tool":"finish"}'), config=CONFIG)
    retriever.retrieve("CASA năm 2025 là bao nhiêu?", 1)
    assert retriever.last_retrieval_calls == []
    assert retriever.last_trace["b5"]["triggered"] is False

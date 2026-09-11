from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import Settings
from src.llm.base import GenerationResult, TokenUsage
from src.retrieval.hyde import HyDERetriever, redact_invented_numbers
from src.retrieval.parent_expansion import ParentExpandingRetriever


CONFIG = Path(__file__).resolve().parents[1] / "config/hyde.json"


class FakeEmbedder:
    def __init__(self) -> None:
        self.documents: list[str] = []

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.documents.extend(texts)
        return np.asarray([[0.0, 1.0]], dtype="float32")


class FakeDense:
    def __init__(self) -> None:
        self.embedder = FakeEmbedder()
        self.original_calls = 0
        self.vector_calls = 0
        self.original = [
            {"chunk_id": "a", "score": 0.90, "printed_page": 1, "pdf_page": 1, "text": "A"},
            {"chunk_id": "b", "score": 0.80, "printed_page": 2, "pdf_page": 2, "text": "B"},
        ]
        self.hyde = [
            {"chunk_id": "c", "score": 0.88, "printed_page": 3, "pdf_page": 3, "text": "C"},
            {"chunk_id": "a", "score": 0.86, "printed_page": 1, "pdf_page": 1, "text": "A"},
        ]

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        del query
        self.original_calls += 1
        return self.original[:top_k]

    def retrieve_by_vector(self, vector: np.ndarray, top_k: int | None = None) -> list[dict]:
        assert vector.shape == (1, 2)
        self.vector_calls += 1
        return self.hyde[:top_k]


class FakeLLM:
    def __init__(self, text: str = "Đoạn giả định 999 tỷ đồng") -> None:
        self.text = text
        self.calls = 0

    def generate(self, messages, temperature=0, purpose="answer", model=None):
        del messages, temperature, model
        self.calls += 1
        return GenerationResult(
            text=self.text,
            provider="test",
            model="test",
            purpose=purpose,
            usage=TokenUsage(uncached_input_tokens=10, output_tokens=5),
            latency_seconds=0.01,
            cost_usd=0.0,
        )


class ExplodingLLM:
    def generate(self, messages, temperature=0, purpose="answer", model=None):
        del messages, temperature, purpose, model
        raise RuntimeError("offline")


def make_hyde(base, llm, **kwargs) -> HyDERetriever:
    return HyDERetriever(
        base,
        settings=Settings(hyde_config_path=CONFIG, hyde_llm_model="test"),
        llm_client=llm,
        config_path=CONFIG,
        candidate_k=10,
        cache_size=8,
        **kwargs,
    )


def test_finance_safe_redacts_only_numbers_missing_from_query() -> None:
    result = redact_invented_numbers(
        "CASA năm 2025 là bao nhiêu?",
        "CASA năm 2025 đạt 40,4% và tăng 17%.",
    )
    assert "2025" in result
    assert "40,4%" not in result and "17%" not in result
    assert result.count("[VALUE]") == 2


def test_hyde_only_embeds_passage_and_returns_only_real_chunks() -> None:
    base = FakeDense()
    retriever = make_hyde(
        base, FakeLLM("Tài liệu giả định"), mode="hyde_only", prompt_variant="paper_faithful"
    )
    results = retriever.retrieve("câu hỏi", 2)
    assert [row["chunk_id"] for row in results] == ["c", "a"]
    assert base.original_calls == 0 and base.vector_calls == 1
    assert base.embedder.documents == ["Tài liệu giả định"]
    assert all(row["text"] in {"A", "C"} for row in results)


def test_fusion_rewards_chunk_found_by_both_queries() -> None:
    retriever = make_hyde(
        FakeDense(), FakeLLM(), mode="fusion", prompt_variant="paper_faithful"
    )
    results = retriever.retrieve("câu hỏi", 3)
    assert results[0]["chunk_id"] == "a"
    assert {item["source"] for item in results[0]["retrieval_sources"]} == {
        "original",
        "hyde",
    }
    assert results[0]["score"] == 0.90


def test_generation_failure_falls_back_to_original_retrieval() -> None:
    retriever = make_hyde(
        FakeDense(), ExplodingLLM(), mode="hyde_only", prompt_variant="paper_faithful"
    )
    results = retriever.retrieve("câu hỏi", 1)
    assert results[0]["chunk_id"] == "a"
    assert results[0]["retrieval_source"] == "original_fallback"
    assert retriever.last_trace["hyde"]["fallback"] is True


def test_cache_avoids_repeated_generation_and_embedding() -> None:
    base = FakeDense()
    llm = FakeLLM("Tài liệu giả định")
    retriever = make_hyde(
        base, llm, mode="fusion", prompt_variant="paper_faithful"
    )
    retriever.retrieve("Cùng câu hỏi", 2)
    retriever.retrieve("  cùng   CÂU HỎI ", 2)
    assert llm.calls == 1
    assert len(base.embedder.documents) == 1
    assert retriever.last_trace["hyde"]["cache_hit"] is True
    assert retriever.last_retrieval_calls == []


def test_parent_expansion_preserves_hyde_trace_and_usage_call() -> None:
    child = make_hyde(
        FakeDense(), FakeLLM("Tài liệu giả định"), mode="hyde_only", prompt_variant="paper_faithful"
    )
    parent = ParentExpandingRetriever(child)
    parent.retrieve("câu hỏi", 1)
    assert parent.last_trace["hyde"]["mode"] == "hyde_only"
    assert parent.last_retrieval_calls[0].purpose == "hyde_generation"

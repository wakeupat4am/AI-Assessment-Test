from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.chat.chatbot import (
    Chatbot,
    build_answer_prompt,
    extract_citations,
    format_evidence,
    validate_citations,
    validate_numeric_support,
)
from src.chat.query_rewriter import rewrite_query
from src.chat.text_safety import normalize_history, normalize_utf8_text
from src.config import Settings
from src.evaluation.evaluator import evaluate_questions
from src.ingestion.chunker import chunk_pages
from src.llm.base import GenerationResult, TokenUsage, aggregate_calls
from src.llm.pricing import KNOWN_PRICES, calculate_cost
from src.llm.provider import _token_count
from src.retrieval.retriever import DenseRetriever


class DummyEmbedder:
    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype="float32")


class EmptyRetriever:
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        return []


class ExplodingLLM:
    def generate(self, messages: list[dict[str, str]], temperature: float = 0) -> str:
        raise AssertionError("LLM should not run on low-confidence refusal")


def test_chunks_preserve_page_metadata() -> None:
    pages = [
        {
            "text": "A" * 200,
            "pdf_page_index": 2,
            "pdf_page": 3,
            "page_part": "left",
            "printed_page": 4,
            "mapping_method": "detected",
            "source": "report.pdf",
        }
    ]
    chunks = chunk_pages(pages, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(chunk["pdf_page"] == 3 for chunk in chunks)
    assert all(chunk["printed_page"] == 4 for chunk in chunks)


def test_citation_parser() -> None:
    assert extract_citations("Số liệu [tr. 45, 46] và [tr. 45].") == [45, 46]


def test_citation_outside_evidence_is_rejected() -> None:
    citations, valid = validate_citations("Kết quả [tr. 99].", {45, 46})
    assert citations == [99]
    assert valid is False


def test_numeric_value_must_exist_on_cited_page() -> None:
    chunks = [
        {"printed_page": 5, "text": "Tỷ lệ CASA 40,4%"},
        {"printed_page": 117, "text": "Tăng trưởng CASA"},
    ]
    assert validate_numeric_support("CASA là 40,4% [tr. 5].", chunks, [5])
    assert not validate_numeric_support("CASA là 40,4% [tr. 117].", chunks, [117])


def test_answer_prompt_keeps_question_after_evidence() -> None:
    prompt = build_answer_prompt("CASA năm 2024?", "Bằng chứng dài")
    assert prompt.index("Bằng chứng dài") < prompt.index("CASA năm 2024?")
    assert "đúng chỉ tiêu và đúng kỳ/năm" in prompt


def test_best_evidence_is_closest_to_question() -> None:
    chunks = [
        {"chunk_id": "best", "printed_page": 1, "pdf_page": 1, "score": 0.9, "text": "BEST"},
        {"chunk_id": "second", "printed_page": 2, "pdf_page": 2, "score": 0.8, "text": "SECOND"},
    ]
    evidence = format_evidence(chunks)
    assert evidence.index("SECOND") < evidence.index("BEST")


def test_standalone_question_returns_text_without_llm() -> None:
    assert rewrite_query("CASA năm 2025 là bao nhiêu?", []) == "CASA năm 2025 là bao nhiêu?"


def test_terminal_surrogates_are_utf8_safe() -> None:
    damaged = "tóm tắt \udcc6 báo cáo"
    normalized = normalize_utf8_text(damaged)
    assert "\udcc6" not in normalized
    normalized.encode("utf-8")


def test_second_turn_rewriter_sanitizes_question_and_history() -> None:
    class EncodingCheckingLLM:
        def generate(self, messages, temperature=0, purpose="answer", model=None):
            for message in messages:
                message["content"].encode("utf-8")
            return GenerationResult(text="tóm tắt báo cáo lưu chuyển tiền tệ", purpose=purpose)

    history = [
        {"role": "user", "content": "Câu đầu \udcc6"},
        {"role": "assistant", "content": "Không đủ thông tin"},
    ]
    rewritten = rewrite_query(
        "câu thứ hai \udced",
        normalize_history(history),
        client=EncodingCheckingLLM(),
    )
    assert rewritten == "tóm tắt báo cáo lưu chuyển tiền tệ"


def test_retriever_returns_top_k(tmp_path: Path) -> None:
    chunks = [
        {"chunk_id": "a", "text": "closest", "printed_page": 1, "pdf_page": 1},
        {"chunk_id": "b", "text": "other", "printed_page": 2, "pdf_page": 2},
    ]
    with (tmp_path / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    settings = Settings(index_dir=tmp_path)
    retriever = DenseRetriever(settings, embedder=DummyEmbedder(), index_dir=tmp_path)
    results = retriever.retrieve("query", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "a"


def test_refusal_path_does_not_call_llm(tmp_path: Path) -> None:
    settings = Settings(index_dir=tmp_path, processed_dir=tmp_path)
    chatbot = Chatbot(settings, retriever=EmptyRetriever(), llm_client=ExplodingLLM())
    result = chatbot.ask("Câu hỏi không có trong báo cáo")
    assert result["refused"] is True
    assert result["citations"] == []


def test_batch_runner_needs_no_keyboard_input(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"question": "Một câu hỏi"}]), encoding="utf-8")

    class FakeBot:
        def ask(self, question: str, conversation_history: list[dict]) -> dict:
            return {
                "standalone_query": question,
                "answer": "Không đủ thông tin",
                "retrieved_chunks": [],
                "citations": [],
                "citation_valid": True,
                "refused": True,
                "latency_seconds": 0.01,
                "routing": {
                    "hyde": {
                        "mode": "fusion",
                        "fallback": False,
                    }
                },
            }

    output = tmp_path / "results.jsonl"
    rows = evaluate_questions(questions, output, chatbot=FakeBot())
    assert len(rows) == 1
    assert output.exists()


def test_single_model_fallback_covers_fast_and_strong(tmp_path: Path) -> None:
    settings = Settings(index_dir=tmp_path, llm_model="one-model")
    assert settings.model_for_purpose("rewrite") == "one-model"
    assert settings.model_for_purpose("answer") == "one-model"
    assert settings.model_for_purpose("repair") == "one-model"


def test_usage_log_preserves_purpose_tokens_latency_and_cost() -> None:
    call = GenerationResult(
        text="answer",
        provider="openai",
        model="gpt-5.6-terra",
        purpose="answer",
        latency_seconds=1.25,
        usage=TokenUsage(uncached_input_tokens=1000, output_tokens=100),
        cost_usd=0.0032,
        pricing_source="test",
    )
    result = aggregate_calls([call])
    assert result["calls"][0]["purpose"] == "answer"
    assert result["calls"][0]["latency_seconds"] == 1.25
    assert result["usage"]["uncached_input_tokens"] == 1000
    assert result["usage"]["output_tokens"] == 100
    assert result["cost_usd"] == 0.0032


def test_current_cost_projection_accounts_for_cache_and_output() -> None:
    usage = TokenUsage(
        uncached_input_tokens=1000,
        cached_input_tokens=1000,
        output_tokens=100,
    )
    assert calculate_cost(usage, KNOWN_PRICES["gpt-5.6-terra"]) == 0.0034


def test_token_count_handles_batch_encoding_shape() -> None:
    assert _token_count({"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}) == 3


def test_evaluation_splits_are_valid_and_disjoint() -> None:
    from src.evaluation.evaluator import load_questions

    evaluation_dir = Path(__file__).resolve().parents[1] / "data/evaluation"
    splits = {
        name: load_questions(evaluation_dir / f"{name}.json")
        for name in ("public", "dev", "holdout")
    }
    assert {name: len(rows) for name, rows in splits.items()} == {
        "public": 10,
        "dev": 20,
        "holdout": 10,
    }
    id_sets = [{row["id"] for row in rows} for rows in splits.values()]
    assert not (id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2])

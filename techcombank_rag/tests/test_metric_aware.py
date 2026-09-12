from __future__ import annotations

from src.chat.query_rewriter import rewrite_query
from src.chat.metric_answerer import answer_metric_query
from src.retrieval.metric_aware import MetricAwareRetriever, SelectiveMetricAwareRetriever
from src.retrieval.metric_identity import (
    candidate_matches_constraints,
    enrich_metric_identity,
    normalize_search_text,
)


def row(chunk_id: str, page: int, text: str, score: float = 1.0):
    return {
        "chunk_id": chunk_id,
        "printed_page": page,
        "pdf_page": page,
        "granularity": "row",
        "text": text,
        "score": score,
        "fusion_score_normalized": score,
    }


class FakeRetriever:
    def __init__(self, rows):
        self.rows = rows
        self.last_trace = None

    def retrieve(self, query, top_k=None):
        del query
        return [dict(item) for item in self.rows[:top_k]]


def test_ocr_normalization_preserves_meaning_for_search() -> None:
    assert normalize_search_text("Thu nhập tử hoạt động - triệu đông") == (
        "Thu nhập từ hoạt động - triệu đồng"
    )


def test_a32_identity_distinguishes_cash_receipt_from_nfi() -> None:
    cash = enrich_metric_identity(
        row(
            "a31_cash",
            301,
            "Ngữ cảnh tài liệu: Báo cáo lưu chuyển tiền tệ hợp nhất | "
            "Cột 1: Thu nhập tử hoạt động dịch vụ nhận được | "
            "2025\\nTriệu đông: 8.396.946 | 2024\\nTriệu đồng: 7.679.933",
        )
    )
    nfi = enrich_metric_identity(
        row(
            "a31_nfi",
            50,
            "Ngữ cảnh tài liệu: Kết quả hoạt động | "
            "Cột 1: Thu nhập thuần từ hoạt động dịch vụ | 2025: 11,5 nghìn tỷ đồng",
        )
    )
    assert cash["metric_key"] != nfi["metric_key"]
    assert cash["metric_qualifiers"] == ["nhận được"]
    assert cash["statement_type"] == "cash_flow"
    assert "Thu nhập từ hoạt động dịch vụ nhận được" in cash["search_text"]


def test_metric_reranker_promotes_required_qualifier() -> None:
    nfi = enrich_metric_identity(row("a31_nfi", 50, "Cột 1: Thu nhập từ hoạt động dịch vụ | 2025: 11,5", 1.0))
    cash = enrich_metric_identity(row("a31_cash", 301, "Cột 1: Thu nhập tử hoạt động dịch vụ nhận được | 2025: 8.396.946", 0.8))
    retriever = MetricAwareRetriever(
        FakeRetriever([nfi, cash]),
        {
            "base_weight": 0.48,
            "label_weight": 0.22,
            "exact_label_weight": 0.10,
            "qualifier_weight": 0.14,
            "year_weight": 0.03,
            "statement_weight": 0.03,
            "missing_qualifier_penalty": 0.35,
        },
        candidate_k=2,
    )
    results = retriever.retrieve(
        "Thu nhập từ hoạt động dịch vụ nhận được năm 2025 là bao nhiêu?", 2
    )
    assert results[0]["printed_page"] == 301
    assert results[0]["metric_constraint_pass"] is True
    assert results[1]["metric_constraint_pass"] is False


def test_metric_constraint_rejects_near_named_wrong_metric() -> None:
    wrong = enrich_metric_identity(row("a31_nfi", 50, "Thu nhập từ hoạt động dịch vụ đạt 11,5"))
    assert not candidate_matches_constraints(
        "Thu nhập từ hoạt động dịch vụ nhận được năm 2025", wrong
    )


def test_metric_followup_preserves_exact_anchor_without_llm() -> None:
    history = [
        {
            "role": "user",
            "content": "Thu nhập từ hoạt động dịch vụ nhận được năm 2025 là bao nhiêu?",
        },
        {"role": "assistant", "content": "Một câu trả lời có thể sai"},
    ]
    assert rewrite_query(
        "Còn năm 2024 thì sao?",
        history,
        preserve_metric_identity=True,
    ) == "Thu nhập từ hoạt động dịch vụ nhận được năm 2024 là bao nhiêu?"


def test_current_b4b_still_uses_original_llm_rewriter() -> None:
    class FakeLLM:
        def generate(self, messages, temperature=0, purpose="answer", model=None):
            del messages, temperature, purpose, model
            return "LLM rewrite unchanged"

    rewritten = rewrite_query(
        "Còn năm 2024 thì sao?",
        [{"role": "user", "content": "CASA năm 2025?"}],
        client=FakeLLM(),
        preserve_metric_identity=False,
    )
    assert rewritten == "LLM rewrite unchanged"


def test_selective_metric_path_preserves_frozen_results_for_generic_queries() -> None:
    frozen = FakeRetriever([row("frozen", 5, "CASA năm 2025")])
    metric = FakeRetriever([row("metric", 301, "Thu nhập dịch vụ nhận được")])
    retriever = SelectiveMetricAwareRetriever(
        frozen, metric, ["nhận được", "đã trả"]
    )
    assert retriever.retrieve("CASA năm 2025", 1)[0]["chunk_id"] == "frozen"
    assert retriever.last_trace["b4"]["selected_path"] == "A3.1+B4b_frozen"
    assert retriever.retrieve("Thu nhập dịch vụ nhận được", 1)[0]["chunk_id"] == "metric"
    assert retriever.last_trace["b4"]["selected_path"] == "A3.2+B4e"


def test_nfi_query_can_remain_on_frozen_path() -> None:
    frozen = FakeRetriever([row("frozen_nfi", 50, "NFI 11,5")])
    metric = FakeRetriever([row("wrong_note", 356, "Dịch vụ 12.526.963")])
    retriever = SelectiveMetricAwareRetriever(
        frozen, metric, ["nhận được", "báo cáo kết quả hoạt động"]
    )
    result = retriever.retrieve(
        "Thu nhập thuần từ hoạt động dịch vụ NFI năm 2025 đạt bao nhiêu?", 1
    )
    assert result[0]["printed_page"] == 50
    assert retriever.last_trace["b4"]["selected_path"] == "A3.1+B4b_frozen"


def test_deterministic_metric_answer_reads_exact_row() -> None:
    chunk = enrich_metric_identity(
        row(
            "a31_cash",
            301,
            "Cột 1: Thu nhập tử hoạt động dịch vụ nhận được | "
            "2025\\nTriệu đông: 8.396.946 | 2024\\nTriệu đồng: 7.679.933",
        )
    )
    chunk["metric_constraint_pass"] = True
    result = answer_metric_query(
        "Thu nhập từ hoạt động dịch vụ nhận được năm 2025 là bao nhiêu?", [chunk]
    )
    assert result is not None
    assert result.answer == (
        "Thu nhập từ hoạt động dịch vụ nhận được năm 2025 là "
        "8.396.946 triệu đồng [tr. 301]."
    )


def test_deterministic_metric_answer_calculates_same_row_comparison() -> None:
    chunk = enrich_metric_identity(
        row(
            "a31_cash",
            301,
            "Cột 1: Thu nhập tử hoạt động dịch vụ nhận được | "
            "2025\\nTriệu đông: 8.396.946 | 2024\\nTriệu đồng: 7.679.933",
        )
    )
    chunk["metric_constraint_pass"] = True
    result = answer_metric_query(
        "Thu nhập từ hoạt động dịch vụ nhận được năm 2025 tăng bao nhiêu so với năm 2024?",
        [chunk],
    )
    assert result is not None
    assert "717.013 triệu đồng" in result.answer
    assert "9,34%" in result.answer
    assert result.calculation is not None


def test_deterministic_metric_answer_refuses_cross_row_or_missing_year() -> None:
    chunk = enrich_metric_identity(
        row(
            "a31_cash",
            301,
            "Cột 1: Thu nhập tử hoạt động dịch vụ nhận được | "
            "2025\\nTriệu đông: 8.396.946",
        )
    )
    chunk["metric_constraint_pass"] = True
    assert answer_metric_query(
        "Thu nhập từ hoạt động dịch vụ nhận được năm 2025 so với năm 2024?", [chunk]
    ) is None

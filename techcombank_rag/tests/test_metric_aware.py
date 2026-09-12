from __future__ import annotations

from pathlib import Path

from src.chat.chatbot import Chatbot
from src.chat.citation_grounding import canonicalize_citation
from src.chat.conversation_grounding import (
    history_queries_for_multi_reference,
    merge_round_robin,
)
from src.chat.query_rewriter import rewrite_query
from src.chat.metric_answerer import answer_metric_query
from src.config import Settings
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


def test_explicit_question_is_not_rewritten_from_metric_history() -> None:
    class ExplodingLLM:
        def generate(self, *args, **kwargs):
            raise AssertionError("A self-contained question must bypass rewrite")

    question = (
        "Thu nhập từ hoạt động dịch vụ năm 2025 trong thuyết minh "
        "báo cáo tài chính hợp nhất là bao nhiêu?"
    )
    assert rewrite_query(
        question,
        [{"role": "user", "content": "Thu nhập dịch vụ nhận được năm 2025?"}],
        client=ExplodingLLM(),
        preserve_metric_identity=True,
    ) == question


def test_plural_reference_decomposes_only_previous_user_questions() -> None:
    history = [
        {"role": "user", "content": "Dòng tiền dịch vụ nhận được năm 2025 là bao nhiêu?"},
        {"role": "assistant", "content": "Một câu trả lời không được dùng làm evidence"},
        {"role": "user", "content": "Thu nhập dịch vụ trong thuyết minh năm 2025?"},
        {"role": "assistant", "content": "Một câu trả lời khác"},
        {"role": "user", "content": "NFI năm 2025 đạt bao nhiêu?"},
        {"role": "assistant", "content": "Một câu trả lời thứ ba"},
    ]
    queries = history_queries_for_multi_reference(
        "Ba số liệu trên có phải cùng một chỉ tiêu không?", history
    )
    assert queries == [history[0]["content"], history[2]["content"], history[4]["content"]]
    assert all("câu trả lời" not in query for query in queries)


def test_round_robin_merge_preserves_each_subquery_top_result() -> None:
    groups = [
        [row("a", 301, "A"), row("a2", 301, "A2")],
        [row("b", 356, "B"), row("b2", 356, "B2")],
        [row("c", 50, "C"), row("c2", 50, "C2")],
    ]
    assert [item["chunk_id"] for item in merge_round_robin(groups)[:3]] == [
        "a", "b", "c"
    ]


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


def test_semantic_metric_answer_includes_directly_attached_context() -> None:
    chunk = row("npl", 5, "Tỷ lệ nợ xấu 1,13% Chất lượng tài sản thuộc nhóm tốt nhất thị trường")
    chunk.update(
        {
            "representation": "a3.1",
            "reporting_year": 2025,
            "semantic_fields": {
                "metric": "Tỷ lệ nợ xấu",
                "value": "1,13%",
                "unit": "%",
                "details": ["Chất lượng tài sản thuộc nhóm tốt nhất thị trường"],
            },
        }
    )
    result = answer_metric_query("Tỷ lệ nợ xấu năm 2025 là bao nhiêu?", [chunk])
    assert result is not None
    assert result.answer == (
        "Tỷ lệ nợ xấu năm 2025 là 1,13%, chất lượng tài sản thuộc nhóm "
        "tốt nhất thị trường [tr. 5]."
    )


def test_citation_canonicalizer_requires_all_numeric_claims() -> None:
    chunks = [
        row("partial", 27, "302 chi nhánh và phòng giao dịch"),
        row("complete", 4, "302 chi nhánh và phòng giao dịch tại 29 trên 34 tỉnh thành"),
    ]
    result = canonicalize_citation(
        "Techcombank có 302 chi nhánh và phòng giao dịch tại 29 trên 34 tỉnh thành [tr. 27].",
        chunks,
    )
    assert result is not None
    assert result[1] == [4]
    assert result[0].endswith("[tr. 4].")


def test_citation_canonicalizer_uses_earliest_equivalent_source() -> None:
    answer = "S&P Global Ratings xếp hạng BB và Fitch Ratings xếp hạng BB- [tr. 43]."
    chunks = [
        row("later", 43, "S&P Global Ratings xếp hạng BB và Fitch Ratings xếp hạng BB-"),
        row("summary", 4, "S&P Global Ratings xếp hạng BB và Fitch Ratings xếp hạng BB-"),
    ]
    result = canonicalize_citation(answer, chunks)
    assert result is not None
    assert result[1] == [4]


def test_metric_pipeline_bypasses_rewrite_and_returns_contextual_metric(tmp_path: Path) -> None:
    chunk = row("npl", 5, "Tỷ lệ nợ xấu 1,13% Chất lượng tài sản thuộc nhóm tốt nhất thị trường")
    chunk.update(
        {
            "representation": "a3.1",
            "reporting_year": 2025,
            "semantic_fields": {
                "metric": "Tỷ lệ nợ xấu",
                "value": "1,13%",
                "unit": "%",
                "details": ["Chất lượng tài sản thuộc nhóm tốt nhất thị trường"],
            },
        }
    )

    class NoLLM:
        def generate(self, *args, **kwargs):
            raise AssertionError("Neither rewrite nor answer LLM should be needed")

    settings = Settings(
        index_dir=tmp_path,
        processed_dir=tmp_path,
        b4_retrieval_mode="metric_aware",
        min_retrieval_score=-1.0,
    )
    bot = Chatbot(settings, retriever=FakeRetriever([chunk]), llm_client=NoLLM())
    result = bot.ask(
        "Tỷ lệ nợ xấu của Techcombank năm 2025 là bao nhiêu?",
        [{"role": "user", "content": "CASA năm 2025 là bao nhiêu?"}],
    )
    assert result["answer_mode"] == "deterministic_semantic_metric"
    assert "chất lượng tài sản thuộc nhóm tốt nhất thị trường" in result["answer"]
    assert result["citations"] == [5]


def test_metric_pipeline_retrieves_each_plural_history_reference(tmp_path: Path) -> None:
    groups = {
        "nhận được": [row("cash", 301, "Dòng tiền dịch vụ nhận được 8.396.946")],
        "thuyết minh": [row("note", 356, "Thu nhập dịch vụ thuyết minh 12.526.963")],
        "NFI": [row("nfi", 50, "NFI 11,5 nghìn tỷ đồng")],
    }

    class QueryRetriever:
        last_trace = None
        last_retrieval_calls = []
        last_router_call = None
        last_selected_evidence = None

        def __init__(self):
            self.queries = []

        def retrieve(self, query, top_k=None):
            self.queries.append(query)
            for marker, rows in groups.items():
                if marker.casefold() in query.casefold():
                    return rows[:top_k]
            return []

    class AnswerLLM:
        def generate(self, messages, temperature=0, purpose="answer", model=None):
            assert purpose == "answer"
            return (
                "Đây là ba chỉ tiêu khác nhau: dòng tiền nhận được 8.396.946 "
                "[tr. 301], thu nhập trong thuyết minh 12.526.963 [tr. 356], "
                "và NFI 11,5 nghìn tỷ đồng [tr. 50]."
            )

    history = [
        {"role": "user", "content": "Thu nhập dịch vụ nhận được năm 2025 là bao nhiêu?"},
        {"role": "assistant", "content": "8.396.946"},
        {"role": "user", "content": "Thu nhập dịch vụ trong thuyết minh năm 2025?"},
        {"role": "assistant", "content": "12.526.963"},
        {"role": "user", "content": "NFI năm 2025 đạt bao nhiêu?"},
        {"role": "assistant", "content": "11,5"},
    ]
    retriever = QueryRetriever()
    settings = Settings(
        index_dir=tmp_path,
        processed_dir=tmp_path,
        b4_retrieval_mode="metric_aware",
        min_retrieval_score=-1.0,
    )
    result = Chatbot(settings, retriever=retriever, llm_client=AnswerLLM()).ask(
        "Ba số liệu trên có phải cùng một chỉ tiêu không?", history
    )
    assert len(retriever.queries) == 3
    assert result["citations"] == [301, 356, 50]
    assert result["citation_valid"] is True
    assert result["routing"]["multi_turn_decomposition"]["query_count"] == 3

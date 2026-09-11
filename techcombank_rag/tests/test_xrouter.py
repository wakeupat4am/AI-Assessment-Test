from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import Settings
from src.retrieval.routed_retriever import RoutedRetriever
from src.routing.xrouter import FEATURE_NAMES, RuleBasedXRouter


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/xrouter_b1a.json"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Lợi nhuận trước thuế năm 2025 là bao nhiêu?", "structured"),
        ("Tỷ lệ nợ xấu của Techcombank là bao nhiêu?", "structured"),
        ("Tổng tài sản tại ngày 31/12/2025 là bao nhiêu?", "structured"),
        ("RBG là viết tắt của khối nào?", "structured"),
        ("Fitch xếp hạng tín nhiệm Techcombank ở mức nào?", "structured"),
        ("Tính đến năm 2025 ngân hàng có bao nhiêu chi nhánh?", "structured"),
        ("Tại sao CASA quan trọng với ngân hàng?", "narrative"),
        ("Chiến lược ngân hàng số được mô tả như thế nào?", "narrative"),
        ("Chính sách quản trị rủi ro được mô tả như thế nào?", "narrative"),
        ("Vì sao chất lượng tài sản quan trọng?", "narrative"),
        ("Vai trò của Khối Dữ liệu và Phân tích là gì?", "narrative"),
        ("Định hướng phát triển bền vững của ngân hàng là gì?", "narrative"),
        ("CASA năm 2025 so với 2024 thay đổi thế nào?", "multi_hop"),
        ("Doanh thu từ phí tăng bao nhiêu phần trăm so với năm trước?", "multi_hop"),
        ("S&P và Fitch xếp hạng Techcombank như thế nào?", "multi_hop"),
        ("So sánh ROA và ROE của Techcombank năm 2025.", "multi_hop"),
        ("Tổng thu nhập hoạt động và CAGR giai đoạn 2018–2025 là bao nhiêu?", "multi_hop"),
        ("Tính tốc độ tăng trưởng tổng tài sản từ năm 2023 đến năm 2025.", "multi_hop"),
        ("Cho tôi số liệu CASA và giải thích nguyên nhân thay đổi.", "multi_hop"),
        ("Nợ xấu tăng hay giảm so với cùng kỳ?", "multi_hop"),
        ("Techcombank có những điểm nổi bật gì?", "multi_repr"),
        ("Thông tin về Khối Ngân hàng Bán lẻ.", "multi_repr"),
        ("Cho biết tình hình hoạt động của Techcombank.", "multi_repr"),
        ("Thông tin liên quan đến CASA.", "multi_repr"),
    ],
)
def test_vietnamese_routing_examples(query: str, expected: str) -> None:
    decision = RuleBasedXRouter(CONFIG_PATH).route(query)
    assert decision["route"] == expected
    assert 0 <= decision["confidence"] <= 1
    assert decision["reasons"]
    assert set(decision["query_features"]) == set(FEATURE_NAMES)


def test_followup_uses_conversation_context() -> None:
    router = RuleBasedXRouter(CONFIG_PATH)
    no_context = router.route("Còn chỉ tiêu này?")
    with_context = router.route(
        "Còn chỉ tiêu này?", "Người dùng vừa hỏi về tỷ lệ CASA năm 2025."
    )
    assert no_context["query_features"]["is_followup"] is False
    assert with_context["query_features"]["is_followup"] is True
    assert with_context["route"] == "multi_repr"


def test_low_confidence_falls_back_to_multi_repr() -> None:
    decision = RuleBasedXRouter(CONFIG_PATH).route("Techcombank")
    assert decision["route"] == "multi_repr"
    assert decision["confidence"] < 0.62
    assert any("fallback" in reason for reason in decision["reasons"])


def test_router_jsonl_log_contains_required_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "router.jsonl"
    router = RuleBasedXRouter(CONFIG_PATH, log_path=log_path)
    router.route("Lợi nhuận năm 2025 là bao nhiêu?")
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["query"] == "Lợi nhuận năm 2025 là bao nhiêu?"
    assert payload["selected_route"] == "structured"
    assert isinstance(payload["confidence"], float)
    assert set(payload["features"]) == set(FEATURE_NAMES)
    assert payload["latency_seconds"] >= 0


class FakeRetriever:
    def __init__(self, source: str) -> None:
        self.source = source

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        del query
        return [
            {
                "chunk_id": f"{self.source}-{rank}",
                "text": f"{self.source} evidence {rank}",
                "printed_page": rank,
                "pdf_page": rank,
                "score": 0.9 - rank / 100,
            }
            for rank in range(1, top_k + 1)
        ]


def test_routed_retriever_dispatches_and_counts_candidates(tmp_path: Path) -> None:
    router = RuleBasedXRouter(CONFIG_PATH, log_path=tmp_path / "router.jsonl")
    retrievers = {
        name: FakeRetriever(name) for name in ("narrative", "structured", "multi_repr")
    }
    routed = RoutedRetriever(Settings(), router=router, retrievers=retrievers)
    rows = routed.retrieve_routed("Tỷ lệ nợ xấu là bao nhiêu?", top_k=3)
    assert all(row["retrieval_source"] == "structured" for row in rows)
    assert routed.last_trace is not None
    assert routed.last_trace["route"] == "structured"
    assert routed.last_trace["raw_candidate_count"] == 3
    assert routed.last_trace["returned_candidate_count"] == 3


def test_multi_hop_fuses_two_indexes_without_changing_dense_score() -> None:
    router = RuleBasedXRouter(CONFIG_PATH)
    retrievers = {
        name: FakeRetriever(name) for name in ("narrative", "structured", "multi_repr")
    }
    routed = RoutedRetriever(Settings(), router=router, retrievers=retrievers)
    rows = routed.retrieve_routed(
        "CASA năm 2025 so với 2024 thay đổi thế nào?", top_k=4
    )
    assert routed.last_trace is not None
    assert routed.last_trace["route"] == "multi_hop"
    assert routed.last_trace["selected_indexes"] == ["structured", "narrative"]
    assert routed.last_trace["raw_candidate_count"] == 8
    assert routed.last_trace["returned_candidate_count"] == 4
    assert all(row["score"] > 0.5 for row in rows)
    assert all("fusion_score" in row for row in rows)


def test_controlled_a2_override_excludes_a3_multi_repr_index() -> None:
    router = RuleBasedXRouter(CONFIG_PATH)
    retrievers = {
        name: FakeRetriever(name) for name in ("narrative", "structured")
    }
    controlled_a2 = {
        "narrative": ["narrative"],
        "structured": ["structured"],
        "multi_repr": ["structured", "narrative"],
        "multi_hop": ["structured", "narrative"],
    }
    routed = RoutedRetriever(
        Settings(),
        router=router,
        retrievers=retrievers,
        route_indexes=controlled_a2,
    )
    routed.retrieve_routed("Thông tin liên quan đến CASA.", top_k=3)
    assert routed.last_trace is not None
    assert routed.last_trace["route"] == "multi_repr"
    assert routed.last_trace["selected_indexes"] == ["structured", "narrative"]
    assert routed.last_trace["raw_candidate_count"] == 6

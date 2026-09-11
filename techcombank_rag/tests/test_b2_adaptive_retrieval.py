from __future__ import annotations

from src.retrieval.auto_search import adaptive_search, check_evidence_sufficiency
from src.retrieval.diversity import select_diverse_evidence, select_with_token_budget
from src.retrieval.page_dedup import deduplicate_by_page


def candidate(
    chunk_id: str,
    page: int,
    text: str,
    score: float,
    representation: str = "block",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "printed_page": page,
        "pdf_page": page,
        "text": text,
        "score": score,
        "granularity": representation,
    }


class FakeRetriever:
    def __init__(self, first: list[dict], second: list[dict] | None = None) -> None:
        self.first = first
        self.second = second if second is not None else first
        self.queries: list[str] = []
        self.last_trace = None

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        self.queries.append(query)
        rows = self.first if len(self.queries) == 1 else self.second
        return [dict(row) for row in rows[:top_k]]


def test_one_hop_numeric_query_stops_after_one_search() -> None:
    retriever = FakeRetriever(
        [candidate("roe-2025", 5, "ROE năm 2025 đạt 16,4%.", 0.95, "row")]
    )
    result = adaptive_search("ROE năm 2025 là bao nhiêu?", retriever)
    assert result.state.sufficient is True
    assert result.state.search_round == 1
    assert result.state.stop_reason == "sufficient_after_round_1"


def test_cross_year_comparison_triggers_targeted_second_search() -> None:
    retriever = FakeRetriever(
        [candidate("roe-2025", 5, "ROE năm 2025 đạt 16,4%.", 0.95, "row")],
        [candidate("roe-2024", 6, "ROE năm 2024 đạt 15,2%.", 0.93, "row")],
    )
    result = adaptive_search(
        "ROE năm 2025 thay đổi thế nào so với năm 2024?", retriever
    )
    assert result.state.search_round == 2
    assert "2024" in result.state.queries_used[1]
    assert result.state.queries_used[1] != result.state.original_query
    assert result.state.sufficient is True


def test_question_requiring_two_pages_is_sufficient_when_both_entities_exist() -> None:
    evidence = [
        candidate("sp", 4, "S&P xếp hạng Techcombank ở mức BB.", 0.9),
        candidate("fitch", 43, "Fitch xếp hạng Techcombank ở mức BB-.", 0.88),
    ]
    result = check_evidence_sufficiency(
        "S&P và Fitch xếp hạng tín nhiệm Techcombank như thế nào?", evidence
    )
    assert result.sufficient is True


def test_duplicate_row_block_page_does_not_crowd_one_page() -> None:
    rows = [
        candidate("row", 120, "CASA 2025 40,9%", 0.99, "row"),
        candidate("block", 120, "Bảng CASA 2025 40,9%", 0.98, "block"),
        candidate("page", 120, "Toàn trang Bảng CASA 2025 40,9%", 0.97, "page"),
        candidate("prior", 118, "CASA 2024 39,3%", 0.90, "row"),
    ]
    deduped = deduplicate_by_page(rows, max_per_page=2, similarity_threshold=0.88)
    assert [row["chunk_id"] for row in deduped] == ["row", "prior"]
    assert deduped[0]["printed_page"] == 120


def test_diversity_prefers_new_page_and_information() -> None:
    rows = [
        candidate("a", 120, "CASA 2025 đạt 40,9 phần trăm", 1.0, "row"),
        candidate("b", 120, "CASA 2025 đạt 40,9 phần trăm", 0.99, "page"),
        candidate("c", 118, "CASA 2024 đạt 39,3 phần trăm", 0.90, "row"),
    ]
    selected = select_diverse_evidence(
        "CASA 2025 so với 2024", rows, final_k=2, lambda_relevance=0.7
    )
    assert [row["chunk_id"] for row in selected] == ["a", "c"]


def test_diversity_does_not_displace_direct_metric_evidence() -> None:
    rows = [
        candidate("generic-1", 89, "Techcombank xử lý nợ năm 2025.", 0.904),
        candidate("generic-2", 35, "Techcombank đạt nhiều cột mốc năm 2025.", 0.903),
        candidate("exact", 15, "Tỷ lệ nợ xấu NPL năm 2025 là 1,13%.", 0.893),
        candidate("novel-1", 84, "Tổng tài sản công ty quản lý quỹ.", 0.882),
        candidate("novel-2", 51, "Thu nhập khác đạt 3,8 nghìn tỷ đồng.", 0.887),
        candidate("novel-3", 257, "Techcombank đóng góp thuế cho nền kinh tế.", 0.890),
    ]
    selected = select_diverse_evidence(
        "Tỷ lệ nợ xấu của Techcombank năm 2025 là bao nhiêu?",
        rows,
        final_k=5,
        lambda_relevance=0.7,
    )
    assert selected[4]["chunk_id"] == "exact"


def test_npl_alias_satisfies_bad_debt_term() -> None:
    evidence = [candidate("npl", 15, "NPL năm 2025 đạt 1,13%.", 0.89)]
    result = check_evidence_sufficiency(
        "Tỷ lệ nợ xấu của Techcombank năm 2025 là bao nhiêu?", evidence
    )
    assert result.sufficient is True


def test_narrative_explanation_missing_generates_explanation_query() -> None:
    evidence = [candidate("casa", 5, "CASA 2025 đạt 40,4%.", 0.95, "row")]
    result = check_evidence_sufficiency(
        "CASA năm 2025 thay đổi thế nào và nguyên nhân là gì?", evidence
    )
    assert result.sufficient is False
    assert "explanation" in result.missing_information
    assert result.followup_query is not None
    assert "nguyên nhân" in result.followup_query.casefold()


def test_second_search_with_no_new_evidence_stops_insufficient() -> None:
    first = [candidate("roe-2025", 5, "ROE năm 2025 đạt 16,4%.", 0.95)]
    retriever = FakeRetriever(first, first)
    result = adaptive_search(
        "ROE năm 2025 thay đổi thế nào so với năm 2024?", retriever
    )
    assert result.state.search_round == 2
    assert result.state.sufficient is False
    assert result.state.stop_reason == "max_rounds_reached_insufficient"
    assert len(retriever.queries) == 2


def test_auto_search_can_be_disabled_independently() -> None:
    retriever = FakeRetriever(
        [candidate("roe-2025", 5, "ROE năm 2025 đạt 16,4%.", 0.95)]
    )
    result = adaptive_search(
        "ROE năm 2025 so với năm 2024?",
        retriever,
        enable_auto_search=False,
        enable_page_dedup=True,
        enable_diversity_selection=True,
    )
    assert result.state.search_round == 1
    assert result.state.stop_reason == "auto_search_disabled"


def test_token_budget_limits_final_evidence() -> None:
    rows = [
        candidate("a", 1, "a" * 40, 1.0),
        candidate("b", 2, "b" * 40, 0.9),
        candidate("c", 3, "c" * 40, 0.8),
    ]
    selected, used = select_with_token_budget(rows, max_chunks=3, max_tokens=20)
    assert len(selected) == 2
    assert used == 20

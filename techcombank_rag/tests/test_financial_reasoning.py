from __future__ import annotations

from pathlib import Path

import pytest

from src.chat.chatbot import Chatbot
from src.chat.financial_reasoning import (
    FinancialReasoningController,
    load_financial_reasoning_config,
)
from src.config import APP_ROOT, Settings


CONFIG = APP_ROOT / "config/financial_reasoning.json"


def chunk(chunk_id: str, page: int, text: str, score: float = 1.0) -> dict:
    return {
        "chunk_id": chunk_id,
        "printed_page": page,
        "pdf_page": page,
        "text": text,
        "score": score,
    }


class QueryRetriever:
    last_retrieval_calls = []
    last_trace = None
    last_selected_evidence = None

    def __init__(self, routes: dict[str, list[dict]]) -> None:
        self.routes = routes
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        self.queries.append(query)
        for marker, candidates in self.routes.items():
            if marker.casefold() in query.casefold():
                return [dict(item) for item in candidates[:top_k]]
        return []


def controller(retriever=None, **kwargs) -> FinancialReasoningController:
    return FinancialReasoningController(
        retriever or QueryRetriever({}),
        load_financial_reasoning_config(CONFIG),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("query", "intent", "query_type", "operation"),
    [
        (
            "Techcombank đã đạt kết quả lợi nhuận trước thuế như thế nào so với kế hoạch năm 2025?",
            "plan_vs_actual",
            "calculation",
            "plan_variance",
        ),
        (
            "Các số liệu năm 2025 cho thấy Techcombank đang kết hợp tăng trưởng và kiểm soát rủi ro như thế nào?",
            "growth_risk_balance",
            "multi_hop",
            None,
        ),
        (
            "Vì sao Data Brain quan trọng?",
            "generic",
            "narrative",
            None,
        ),
    ],
)
def test_query_analysis_is_deterministic(query, intent, query_type, operation) -> None:
    plan = controller().analyze(query)
    assert (plan.intent, plan.query_type, plan.operation) == (
        intent,
        query_type,
        operation,
    )


def test_entity_scope_penalizes_subsidiary_for_bank_query() -> None:
    bank = chunk(
        "bank",
        14,
        "Thực thể: Techcombank\nMục: Kế hoạch | 2024: Thực hiện\n"
        "Mục: Lợi nhuận trước thuế | 2025: 31.500 | So sánh: 32.538",
        0.7,
    )
    subsidiary = chunk(
        "tcc",
        84,
        "Công Ty Cổ Phần Quản Lý Quỹ Kỹ Thương (Techcom Capital - TCC)\n"
        "Lợi nhuận trước thuế 183 tỷ đồng, thực hiện vượt kế hoạch",
        1.2,
    )
    ranked = controller()._rank(
        controller().analyze(
            "Techcombank đã đạt lợi nhuận trước thuế như thế nào so với kế hoạch 2025?"
        ),
        [subsidiary, bank],
    )
    assert ranked[0]["chunk_id"] == "bank"
    assert ranked[1]["financial_entity_mismatch"] is True


def test_missing_fact_triggers_only_one_bounded_second_round() -> None:
    first = chunk("credit", 15, "Tăng trưởng tín dụng năm 2025 đạt 18,4%")
    npl = chunk("npl", 5, "Tỷ lệ nợ xấu NPL 1,13%; tỷ lệ bao phủ nợ xấu 127,9%")
    car = chunk("car", 5, "Tỷ lệ an toàn vốn CAR năm 2025 đạt 14,6%")
    retriever = QueryRetriever(
        {
            "kết hợp tăng trưởng": [first],
            "tỷ lệ nợ xấu": [npl],
            "tỷ lệ bao phủ": [npl],
            "tỷ lệ an toàn vốn": [car],
        }
    )
    state = controller(retriever).retrieve(
        "Các số liệu năm 2025 cho thấy Techcombank đang kết hợp tăng trưởng và kiểm soát rủi ro như thế nào?",
        10,
    )
    assert state.search_rounds == 2
    assert len(state.queries_used) <= 5
    assert state.coverage.sufficient is True
    assert set(state.coverage.covered_facts) == {
        "credit_growth",
        "npl",
        "npl_coverage",
        "car",
    }


def test_complete_evidence_stops_after_first_round() -> None:
    evidence = chunk(
        "loan",
        15,
        "Cho vay khách hàng cá nhân tăng trưởng 30,8% N/N; "
        "cho vay khách hàng doanh nghiệp tăng trưởng 13,4% N/N.",
    )
    retriever = QueryRetriever({"cho vay": [evidence]})
    state = controller(retriever).retrieve(
        "Trong năm 2025, tăng trưởng cho vay khách hàng cá nhân và khách hàng doanh nghiệp khác nhau như thế nào?",
        10,
    )
    assert state.search_rounds == 1
    assert retriever.queries == [state.plan.original_query]
    assert state.coverage.sufficient is True


def test_generic_query_preserves_frozen_retriever_order_and_depth() -> None:
    rows = [chunk(f"c{index}", index, f"generic evidence {index}") for index in range(20)]
    retriever = QueryRetriever({"bao nhiêu": rows})
    state = controller(retriever).retrieve(
        "Thu nhập bình quân của nhân viên năm 2025 là bao nhiêu?", 10
    )
    assert state.plan.intent == "generic"
    assert [item["chunk_id"] for item in state.candidates] == [f"c{i}" for i in range(10)]
    assert state.candidates[0]["score"] == rows[0]["score"]
    assert not any("financial_rank_components" in item for item in state.candidates)


@pytest.mark.parametrize(
    ("query", "text", "page", "expected"),
    [
        (
            "Techcombank đã đạt kết quả lợi nhuận trước thuế như thế nào so với kế hoạch năm 2025?",
            "Mục: Kế hoạch | 2024: Thực hiện | 2025: 2024 | % So sánh với: Kế hoạch\n"
            "Mục: Lợi nhuận trước thuế | 2024: 27.538 | 2025: 31.500 | "
            "% So sánh với: 32.538 | Cột 5: +18,16% | Cột 6: +3,30%",
            14,
            "1.038 tỷ đồng",
        ),
        (
            "Trong năm 2025, tăng trưởng cho vay khách hàng cá nhân và khách hàng doanh nghiệp khác nhau như thế nào?",
            "Cho vay khách hàng cá nhân tăng trưởng 30,8% N/N, trong khi cho vay khách hàng doanh nghiệp tăng trưởng 13,4% N/N.",
            15,
            "17,4 điểm phần trăm",
        ),
        (
            "Techcombank đã thay đổi mức độ tập trung tín dụng vào bất động sản như thế nào trong năm 2025?",
            "Tỷ trọng tín dụng bất động sản giảm về 31% tổng dư nợ tại 31/12/2025, so với 33% cuối năm 2024.",
            54,
            "2 điểm phần trăm",
        ),
        (
            "Nếu tín dụng xanh năm 2025 đạt 18,7 nghìn tỷ đồng và tăng 15% so với 2024, có thể ước tính tín dụng xanh năm 2024 là bao nhiêu?",
            "Tín dụng xanh tại Techcombank đạt 18,7 Nghìn Tỷ đồng, tăng 15% so với năm 2024",
            244,
            "16,3 nghìn tỷ đồng",
        ),
    ],
)
def test_calculators_are_grounded_and_carry_provenance(query, text, page, expected) -> None:
    item = chunk("source", page, text)
    ctrl = controller()
    plan = ctrl.analyze(query)
    coverage = ctrl.coverage(plan, [item])
    from src.chat.financial_reasoning import FinancialRetrievalState

    state = FinancialRetrievalState(plan, [item], coverage)
    result = ctrl.calculate(state, [item])
    assert result is not None
    assert expected in result.answer
    assert result.pages == (page,)
    assert all(operand["page"] == page for operand in result.operands)


def test_chatbot_calculator_path_uses_no_answer_llm(tmp_path: Path) -> None:
    evidence = chunk(
        "green",
        244,
        "Tín dụng xanh tại Techcombank đạt 18,7 Nghìn Tỷ đồng, tăng 15% so với năm 2024",
    )

    class NoLLM:
        def generate(self, *args, **kwargs):
            raise AssertionError("A self-contained calculation must not call the LLM")

    settings = Settings(
        index_dir=tmp_path,
        processed_dir=tmp_path,
        b4_retrieval_mode="metric_aware",
        financial_reasoning_enabled=True,
        financial_reasoning_config_path=CONFIG,
        min_retrieval_score=-1.0,
    )
    result = Chatbot(
        settings,
        retriever=QueryRetriever({"tín dụng xanh": [evidence]}),
        llm_client=NoLLM(),
    ).ask(
        "Nếu tín dụng xanh năm 2025 đạt 18,7 nghìn tỷ đồng và tăng 15% so với 2024, có thể ước tính tín dụng xanh năm 2024 là bao nhiêu?"
    )
    assert result["answer_mode"] == "deterministic_financial_calculator"
    assert result["citations"] == [244]
    assert result["financial_grounding"]["operation"] == "reverse_percentage_growth"
    assert result["llm"]["call_count"] == 0


def test_customer_multiplier_is_two_not_four() -> None:
    query = (
        "Nếu Techcombank đạt mục tiêu nhân đôi số lượng khách hàng vào năm 2030, "
        "dựa trên quy mô cuối năm 2025 thì cần phục vụ bao nhiêu khách hàng?"
    )
    items = [
        chunk("base", 106, "Techcombank tự hào phục vụ hơn 18 triệu khách hàng"),
        chunk("target", 21, "Mục tiêu nhân đôi số lượng khách hàng vào năm 2030"),
    ]
    ctrl = controller()
    plan = ctrl.analyze(query)
    from src.chat.financial_reasoning import FinancialRetrievalState

    state = FinancialRetrievalState(plan, items, ctrl.coverage(plan, items))
    result = ctrl.calculate(state, items)
    assert result is not None
    assert "hơn 36 triệu khách hàng" in result.answer


def test_sustainability_summary_can_use_one_complete_summary_page() -> None:
    query = "Phát triển bền vững của Techcombank năm 2025 được thể hiện qua những kết quả định lượng nào?"
    evidence = chunk(
        "esg",
        20,
        "Danh mục tính dung xanh lên hơn 18,7 nghìn tỷ đồng. Đã phát hành hơn "
        "1 triệu thẻ Techcombank Visa Eco Debit và Credit. Techcombank đã đóng góp "
        "231 tỷ đồng để hỗ trợ các hoạt động xã hội. VNSI, nằm trong số 20 công ty "
        "niêm yết có thực hành bền vững tốt nhất.",
    )
    ctrl = controller()
    plan = ctrl.analyze(query)
    from src.chat.financial_reasoning import FinancialRetrievalState

    state = FinancialRetrievalState(plan, [evidence], ctrl.coverage(plan, [evidence]))
    result = ctrl.calculate(state, [evidence])
    assert result is not None
    assert result.pages == (20,)
    assert "18,7 nghìn tỷ" in result.answer
    assert "231 tỷ đồng" in result.answer


def test_risk_summary_requires_and_extracts_numeric_car() -> None:
    query = "Các số liệu năm 2025 cho thấy Techcombank đang kết hợp tăng trưởng và kiểm soát rủi ro như thế nào?"
    risk = chunk(
        "risk",
        15,
        "Tăng trưởng tính dụng đạt 18,4% trong năm 2025; tỷ lệ nợ xấu (NPL) "
        "1,13%; tỷ lệ bao phủ nợ xấu tăng lên 127,9%.",
    )
    qualitative_car = chunk("weak-car", 174, "Khẳng định vị trí dẫn đầu về tỷ lệ an toàn vốn")
    numeric_car = chunk(
        "car",
        256,
        "table: ; 2023 ; 2024 ; 2025 ;\n"
        "Tỷ lệ an toàn vốn (%) ; 14,35% ; 15,40% ; 14,60% ;",
    )
    ctrl = controller()
    plan = ctrl.analyze(query)
    weak = ctrl.coverage(plan, [risk, qualitative_car])
    assert "car" in weak.missing_facts

    from src.chat.financial_reasoning import FinancialRetrievalState

    complete = ctrl.coverage(plan, [risk, numeric_car])
    result = ctrl.calculate(
        FinancialRetrievalState(plan, [risk, numeric_car], complete),
        [risk, numeric_car],
    )
    assert result is not None
    assert "CAR 14,60%" in result.answer
    assert result.pages == (15, 256)


def test_financial_reasoning_requires_frozen_metric_pipeline(tmp_path: Path) -> None:
    settings = Settings(
        index_dir=tmp_path,
        financial_reasoning_enabled=True,
        b4_retrieval_mode="hybrid",
    )
    with pytest.raises(ValueError, match="requires B4_RETRIEVAL_MODE=metric_aware"):
        Chatbot(settings, retriever=QueryRetriever({}), llm_client=object())


def test_chatbot_generic_query_uses_unchanged_metric_aware_path(tmp_path: Path) -> None:
    valid = chunk(
        "valid",
        301,
        "Cột 1: Thu nhập từ hoạt động dịch vụ nhận được | "
        "2025\\nTriệu đồng: 8.396.946",
        0.8,
    )
    valid.update(
        {
            "representation": "a3.2",
            "metric_constraint_pass": True,
            "metric_qualifiers": ["nhận được"],
            "statement_type": "cash_flow",
        }
    )
    invalid = dict(valid, chunk_id="invalid", printed_page=50, score=1.0)
    invalid["metric_qualifiers"] = []
    invalid["metric_constraint_pass"] = False

    class NoLLM:
        def generate(self, *args, **kwargs):
            raise AssertionError("The frozen row answerer should answer this query")

    settings = Settings(
        index_dir=tmp_path,
        processed_dir=tmp_path,
        b4_retrieval_mode="metric_aware",
        financial_reasoning_enabled=True,
        financial_reasoning_config_path=CONFIG,
        min_retrieval_score=-1.0,
    )
    result = Chatbot(
        settings,
        retriever=QueryRetriever({"thu nhập": [invalid, valid]}),
        llm_client=NoLLM(),
    ).ask("Thu nhập từ hoạt động dịch vụ nhận được năm 2025 là bao nhiêu?")
    assert "8.396.946" in result["answer"]
    assert result["citations"] == [301]
    assert "financial_reasoning" not in result.get("routing", {})

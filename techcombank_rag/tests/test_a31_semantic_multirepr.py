from __future__ import annotations

from src.ingestion.semantic_multirepr import (
    extract_metric_values,
    infer_section_paths,
    route_subset,
    semantic_multirepresentation_chunks,
)
from src.retrieval.parent_expansion import ParentExpandingRetriever


def record(
    raw_id: str,
    page: int,
    markdown: str,
    blocks: list[dict] | None = None,
) -> dict:
    return {
        "raw_id": raw_id,
        "logical_index": page,
        "pdf_page_index": page,
        "pdf_page": page + 1,
        "page_part": "right",
        "printed_page": page + 1,
        "mapping_method": "test",
        "source": "report.pdf",
        "markdown": markdown,
        "structured_result": {"res": {"parsing_res_list": blocks or []}},
    }


def block(label: str, text: str, bbox: list[int]) -> dict:
    return {"block_label": label, "block_content": text, "block_bbox": bbox}


def a3_chunk(raw_id: str, page: int, granularity: str = "block") -> dict:
    return {
        "chunk_id": f"a3_{raw_id}_{granularity}_0001",
        "text": "Tiêu đề: Điểm nhấn\nNội dung nguồn.",
        "logical_index": page,
        "pdf_page_index": page,
        "pdf_page": page + 1,
        "page_part": "right",
        "printed_page": page + 1,
        "mapping_method": "test",
        "source": "report.pdf",
        "ocr_raw_id": raw_id,
        "parser": "PaddleOCR-VL-1.6",
        "representation": "a3",
        "granularity": granularity,
        "block_type": "table_row" if granularity == "row" else "narrative",
        "heading": "Điểm nhấn",
    }


def test_inherits_specific_section_into_generic_next_page() -> None:
    rows = [
        record("p1", 1, "# Khối khách hàng doanh nghiệp\nNội dung"),
        record("p2", 2, "# Điểm nhấn 2025\nNội dung"),
    ]
    paths = infer_section_paths(rows)
    assert paths["p2"] == ["Khối khách hàng doanh nghiệp", "Điểm nhấn 2025"]


def test_new_specific_heading_resets_inherited_section() -> None:
    rows = [
        record("p1", 1, "# Khối khách hàng doanh nghiệp"),
        record("p2", 2, "# Báo cáo quản trị rủi ro"),
    ]
    assert infer_section_paths(rows)["p2"] == ["Báo cáo quản trị rủi ro"]


def test_layout_pairs_metric_value_unit_and_growth() -> None:
    row = record(
        "p1",
        1,
        "# Điểm nhấn",
        [
            block("text", "Chỉ tiêu hoạt động", [100, 100, 300, 140]),
            block("text", "53,4", [130, 160, 260, 220]),
            block("text", "Nghìn tỷ đồng", [120, 225, 280, 255]),
            block("text", "Tăng trưởng 2018–2025: 16,5%", [90, 270, 310, 310]),
        ],
    )
    metric = extract_metric_values(row)[0]
    assert metric["metric"] == "Chỉ tiêu hoạt động"
    assert metric["value"] == "53,4"
    assert metric["unit"].casefold() == "nghìn tỷ đồng"
    assert metric["years"] == ["2018", "2025"]


def test_layout_uses_columns_instead_of_ocr_reading_order() -> None:
    row = record(
        "p1",
        1,
        "# Điểm nhấn",
        [
            block("text", "Chỉ tiêu A", [100, 100, 260, 140]),
            block("text", "Chỉ tiêu B", [500, 100, 660, 140]),
            block("text", "10,0", [130, 170, 230, 220]),
            block("text", "20,0", [530, 170, 630, 220]),
        ],
    )
    pairs = {item["metric"]: item["value"] for item in extract_metric_values(row)}
    assert pairs == {"Chỉ tiêu A": "10,0", "Chỉ tiêu B": "20,0"}


def test_markdown_fallback_recovers_missing_structured_blocks() -> None:
    row = record(
        "p1",
        1,
        "# Điểm nhấn\n\nDư nợ vay\n\n328,1 Nghìn tỷ đồng\n\n▲ 26,9% N/N",
    )
    metric = extract_metric_values(row)[0]
    assert metric["metric"] == "Dư nợ vay"
    assert metric["numbers"] == ["328,1", "26,9%"]
    assert metric["extraction_method"] == "markdown_sequence_fallback"


def test_numeric_text_is_not_used_as_a_metric_label() -> None:
    row = record(
        "p1",
        1,
        "# Điểm nhấn\n\nSố dư 475,8 tỷ đồng\n\n15,7% N/N",
    )
    assert extract_metric_values(row) == []


def test_a31_enriches_rows_and_creates_metric_children() -> None:
    raw = record(
        "p1",
        1,
        "# Khối bán lẻ\n\nDư nợ vay\n\n328,1 Nghìn tỷ đồng",
    )
    chunks = semantic_multirepresentation_chunks(
        [a3_chunk("p1", 1), a3_chunk("p1", 1, "row")], [raw]
    )
    assert len({item["chunk_id"] for item in chunks}) == len(chunks)
    assert all(item["representation"] == "a3.1" for item in chunks)
    row = next(item for item in chunks if item["granularity"] == "row")
    metric = next(item for item in chunks if item["granularity"] == "metric")
    assert row["section_path"] == ["Khối bán lẻ"]
    assert row["parent_text"]
    assert metric["semantic_fields"]["metric"] == "Dư nợ vay"
    assert metric["parent_id"] == "a31_parent_p1"


def test_a31_adds_document_scope_but_preserves_business_unit_entity() -> None:
    raw = record(
        "p1",
        1,
        "# Khối bán lẻ\n\nDư nợ vay\n\n328,1 Nghìn tỷ đồng",
    )
    chunks = semantic_multirepresentation_chunks(
        [a3_chunk("p1", 1)],
        [raw],
        document_entity="Ngân hàng mẫu",
        reporting_year=2025,
    )
    metric = next(item for item in chunks if item["granularity"] == "metric")
    assert metric["resolved_entity"] == "Ngân hàng mẫu — Khối bán lẻ"
    assert metric["reporting_year"] == 2025
    assert "Thực thể: Ngân hàng mẫu — Khối bán lẻ" in metric["text"]
    assert "Kỳ báo cáo: 2025" in metric["text"]


def test_route_subsets_are_disjoint_where_expected() -> None:
    metric = {"granularity": "metric", "block_type": "metric_value"}
    narrative = {"granularity": "block", "block_type": "narrative"}
    assert route_subset(metric, "structured") is True
    assert route_subset(metric, "narrative") is False
    assert route_subset(narrative, "narrative") is True
    assert route_subset(narrative, "structured") is False
    assert route_subset(metric, "all") is True


class FakeRetriever:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        del query
        return self.rows[:top_k]


def test_parent_expansion_happens_after_child_retrieval() -> None:
    retriever = ParentExpandingRetriever(
        FakeRetriever(
            [
                {
                    "chunk_id": "child",
                    "text": "Chỉ tiêu: Dư nợ vay\nGiá trị: 328,1",
                    "granularity": "metric",
                    "parent_text": "Khối bán lẻ — Điểm nhấn 2025",
                }
            ]
        )
    )
    result = retriever.retrieve("dư nợ", 1)[0]
    assert result["retrieval_text"].startswith("Chỉ tiêu")
    assert "Ngữ cảnh nguồn" in result["text"]
    assert result["parent_expanded"] is True


def test_parent_expansion_respects_character_limit() -> None:
    retriever = ParentExpandingRetriever(
        FakeRetriever(
            [
                {
                    "chunk_id": "child",
                    "text": "child",
                    "granularity": "row",
                    "parent_text": "p" * 100,
                }
            ]
        ),
        max_parent_characters=10,
    )
    assert retriever.retrieve("q", 1)[0]["text"].endswith("p" * 10)

from __future__ import annotations

from src.ingestion.layout_chunker import (
    layout_aware_chunks,
    multi_granularity_chunks,
    selective_cascade_chunks,
    serialized_table_rows,
)


TABLE = """<table>
<tr><th>Chỉ tiêu</th><th>2025</th></tr>
<tr><td>CASA</td><td>40,4%</td></tr>
<tr><td>Nợ xấu</td><td>1,13%</td></tr>
</table>"""


def record() -> dict:
    return {
        "raw_id": "fragment_0004_pdf003_right_tr005",
        "logical_index": 4,
        "pdf_page_index": 2,
        "pdf_page": 3,
        "page_part": "right",
        "printed_page": 5,
        "mapping_method": "detected",
        "source": "report.pdf",
        "markdown": "## Điểm nhấn\n\nTổng tài sản 1.192\n\n" + TABLE,
        "structured_result": {
            "res": {
                "parsing_res_list": [
                    {
                        "block_label": "paragraph_title",
                        "block_content": "Điểm nhấn 2025",
                        "block_order": 1,
                    },
                    {
                        "block_label": "text",
                        "block_content": "Tổng tài sản 1.192 nghìn tỷ đồng",
                        "block_order": 2,
                    },
                    {
                        "block_label": "table",
                        "block_content": TABLE,
                        "block_order": 3,
                    },
                    {
                        "block_label": "header",
                        "block_content": "Báo cáo thường niên",
                        "block_order": None,
                    },
                ]
            }
        },
    }


def baseline_page() -> dict:
    return {
        "logical_index": 4,
        "pdf_page_index": 2,
        "pdf_page": 3,
        "page_part": "right",
        "printed_page": 5,
        "mapping_method": "detected",
        "source": "report.pdf",
        "text": "Embedded narrative text.",
    }


def test_table_rows_bind_headers_to_values() -> None:
    assert serialized_table_rows(TABLE) == [
        "Chỉ tiêu: CASA | 2025: 40,4%",
        "Chỉ tiêu: Nợ xấu | 2025: 1,13%",
    ]


def test_a2_keeps_heading_and_table_atomic() -> None:
    chunks = layout_aware_chunks([record()])
    assert {chunk["block_type"] for chunk in chunks} == {"narrative", "table"}
    assert all("Điểm nhấn 2025" in chunk["text"] for chunk in chunks)
    table = next(chunk for chunk in chunks if chunk["block_type"] == "table")
    assert "Chỉ tiêu: CASA | 2025: 40,4%" in table["text"]
    assert table["printed_page"] == 5


def test_a3_contains_row_block_and_page_views_with_unique_ids() -> None:
    chunks = multi_granularity_chunks([record()])
    assert {chunk["granularity"] for chunk in chunks} == {"row", "block", "page"}
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    row = next(chunk for chunk in chunks if chunk["granularity"] == "row")
    assert "2025: 40,4%" in row["text"]


def test_a4_preserves_pymupdf_narrative_and_adds_only_structured_region() -> None:
    chunks, stats = selective_cascade_chunks([baseline_page()], [record()])
    narrative = next(chunk for chunk in chunks if chunk["granularity"] == "narrative")
    assert narrative["text"] == "Embedded narrative text."
    assert narrative["parser"] == "PyMuPDF"
    assert not any(
        chunk["parser"] == "PaddleOCR-VL-1.6-selective"
        and "Tổng tài sản" in chunk["text"]
        for chunk in chunks
    )
    assert stats["structured_chunk_count"] == 2
    assert stats["selected_label_counts"] == {"table": 2}

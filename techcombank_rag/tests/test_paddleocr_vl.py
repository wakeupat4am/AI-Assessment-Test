from __future__ import annotations

from src.ingestion.chunker import chunk_pages
from src.ingestion.paddleocr_vl import (
    extract_markdown,
    markdown_to_plain_text,
    representation_pages,
)


SAMPLE_MARKDOWN = """# Chỉ tiêu chính

| Chỉ tiêu | 2025 |
| --- | ---: |
| CASA | **40,4%** |

- [Nguồn](https://example.test)
"""


def _record() -> dict:
    return {
        "raw_id": "fragment_0001_pdf001_left_tr001",
        "logical_index": 1,
        "pdf_page_index": 0,
        "pdf_page": 1,
        "page_part": "left",
        "printed_page": 1,
        "mapping_method": "detected",
        "source": "report.pdf",
        "markdown": SAMPLE_MARKDOWN,
    }


def test_plain_conversion_retains_table_values_without_markdown_syntax() -> None:
    plain = markdown_to_plain_text(SAMPLE_MARKDOWN)
    assert "Chỉ tiêu ; 2025" in plain
    assert "CASA ; 40,4%" in plain
    assert "**" not in plain
    assert "https://" not in plain
    assert "|" not in plain


def test_a1a_and_a1b_share_raw_source_but_differ_in_representation() -> None:
    plain = representation_pages([_record()], "plain")[0]
    markdown = representation_pages([_record()], "markdown")[0]
    assert plain["ocr_raw_id"] == markdown["ocr_raw_id"]
    assert plain["printed_page"] == markdown["printed_page"] == 1
    assert plain["representation"] == "plain"
    assert markdown["representation"] == "markdown"
    assert "| Chỉ tiêu |" not in plain["text"]
    assert "| Chỉ tiêu |" in markdown["text"]


def test_chunks_retain_a1_provenance() -> None:
    page = representation_pages([_record()], "markdown")[0]
    chunk = chunk_pages([page], chunk_size=3200, overlap=480)[0]
    assert chunk["parser"] == "PaddleOCR-VL-1.6"
    assert chunk["representation"] == "markdown"
    assert chunk["ocr_raw_id"] == _record()["raw_id"]


def test_blank_paddle_page_is_valid() -> None:
    class BlankResult:
        markdown = {"markdown_texts": "", "markdown_images": {}}

    assert extract_markdown(BlankResult()) == ""

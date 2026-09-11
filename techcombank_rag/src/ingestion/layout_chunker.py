from __future__ import annotations

import re
from collections import Counter
from html.parser import HTMLParser
from typing import Any, Iterable

from src.ingestion.chunker import _windows, chunk_pages
from src.ingestion.paddleocr_vl import markdown_to_plain_text


HEADING_LABELS = {"doc_title", "paragraph_title"}
IGNORED_LABELS = {"header", "footer", "number", "image"}
SELECTIVE_LABELS = {
    "table",
    "chart",
    "figure_title",
    "vision_footnote",
    "display_formula",
}


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(_clean(" ".join(self._cell)))
            self._cell = None
        elif lowered == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def table_rows(content: str) -> list[list[str]]:
    if "<table" in content.lower():
        parser = _TableParser()
        parser.feed(content)
        return parser.rows
    rows: list[list[str]] = []
    for line in content.splitlines():
        if "|" not in line or re.match(r"^\s*\|?\s*:?-{3,}", line):
            continue
        cells = [_clean(cell) for cell in line.strip().strip("|").split("|")]
        if any(cells):
            rows.append(cells)
    return rows


def _row_text(header: list[str], row: list[str]) -> str:
    width = max(len(header), len(row))
    pairs: list[str] = []
    for index in range(width):
        name = header[index] if index < len(header) and header[index] else f"Cột {index + 1}"
        value = row[index] if index < len(row) else ""
        if value:
            pairs.append(f"{name}: {value}")
    return " | ".join(pairs)


def serialized_table_rows(content: str) -> list[str]:
    rows = table_rows(content)
    if not rows:
        plain = markdown_to_plain_text(content)
        return [plain] if plain else []
    header = rows[0]
    if len(rows) == 1:
        return [" | ".join(cell for cell in header if cell)]
    return [text for row in rows[1:] if (text := _row_text(header, row))]


def _blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = (
        record.get("structured_result", {})
        .get("res", {})
        .get("parsing_res_list", [])
    )
    return value if isinstance(value, list) else []


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "logical_index": int(record["logical_index"]),
        "pdf_page_index": int(record["pdf_page_index"]),
        "pdf_page": int(record["pdf_page"]),
        "page_part": record["page_part"],
        "printed_page": int(record["printed_page"]),
        "mapping_method": record.get("mapping_method"),
        "source": record["source"],
        "ocr_raw_id": record["raw_id"],
    }


def _heading_text(headings: dict[str, str]) -> str:
    values = [headings.get("doc_title", ""), headings.get("paragraph_title", "")]
    return " > ".join(value for value in values if value)


def _with_heading(text: str, heading: str) -> str:
    return f"Tiêu đề: {heading}\n{text}" if heading else text


def _make_chunk(
    record: dict[str, Any],
    *,
    experiment: str,
    sequence: int,
    text: str,
    block_type: str,
    granularity: str,
    heading: str = "",
    source_parser: str = "PaddleOCR-VL-1.6",
) -> dict[str, Any]:
    return {
        "chunk_id": (
            f"{experiment.lower()}_{record['raw_id']}_{granularity}_{sequence:04d}"
        ),
        "text": text.strip(),
        **_metadata(record),
        "parser": source_parser,
        "representation": experiment.lower(),
        "granularity": granularity,
        "block_type": block_type,
        "heading": heading,
    }


def layout_aware_chunks(
    records: Iterable[dict[str, Any]],
    *,
    experiment: str = "A2",
    chunk_size: int = 3200,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for record in records:
        sequence = 0
        headings = {"doc_title": "", "paragraph_title": ""}
        narrative_parts: list[str] = []
        narrative_heading = ""

        def emit(text: str, block_type: str, heading: str) -> None:
            nonlocal sequence
            for window in _windows(_with_heading(text, heading), chunk_size, overlap):
                sequence += 1
                chunks.append(
                    _make_chunk(
                        record,
                        experiment=experiment,
                        sequence=sequence,
                        text=window,
                        block_type=block_type,
                        granularity="block",
                        heading=heading,
                    )
                )

        def flush_narrative() -> None:
            nonlocal narrative_parts, narrative_heading
            if narrative_parts:
                emit("\n\n".join(narrative_parts), "narrative", narrative_heading)
            narrative_parts = []
            narrative_heading = ""

        for block in _blocks(record):
            label = str(block.get("block_label", "unknown"))
            content = str(block.get("block_content", "")).strip()
            if label in HEADING_LABELS and content:
                flush_narrative()
                headings[label] = _clean(markdown_to_plain_text(content))
                if label == "doc_title":
                    headings["paragraph_title"] = ""
                continue
            if label in IGNORED_LABELS or not content:
                continue
            heading = _heading_text(headings)
            if label == "table":
                flush_narrative()
                rows = serialized_table_rows(content)
                current: list[str] = []
                for row in rows:
                    candidate = "\n".join([*current, row])
                    if current and len(_with_heading(candidate, heading)) > chunk_size:
                        emit("\n".join(current), "table", heading)
                        current = [row]
                    else:
                        current.append(row)
                if current:
                    emit("\n".join(current), "table", heading)
                continue
            plain = markdown_to_plain_text(content)
            if not plain:
                continue
            if narrative_parts and heading != narrative_heading:
                flush_narrative()
            narrative_heading = heading
            candidate = "\n\n".join([*narrative_parts, plain])
            if narrative_parts and len(_with_heading(candidate, heading)) > chunk_size:
                flush_narrative()
                narrative_heading = heading
            narrative_parts.append(plain)
        flush_narrative()

        # Version-skew fallback: retain a valid record even if structured blocks
        # are unavailable, but keep its provenance explicit.
        if sequence == 0:
            plain = markdown_to_plain_text(str(record.get("markdown", "")))
            if plain:
                emit(plain, "fallback_page", "")
    return chunks


def _page_signature(record: dict[str, Any], limit: int = 3200) -> str:
    parts: list[str] = []
    for block in _blocks(record):
        label = str(block.get("block_label", "unknown"))
        if label in IGNORED_LABELS:
            continue
        content = markdown_to_plain_text(str(block.get("block_content", "")))
        if not content:
            continue
        preview = content if label in HEADING_LABELS else content[:240]
        parts.append(f"{label}: {preview}")
    signature = "\n".join(parts)
    return signature[:limit].rstrip()


def multi_granularity_chunks(
    records: Iterable[dict[str, Any]], chunk_size: int = 3200
) -> list[dict[str, Any]]:
    record_list = list(records)
    output: list[dict[str, Any]] = []
    for chunk in layout_aware_chunks(
        record_list, experiment="A3", chunk_size=chunk_size
    ):
        chunk["granularity"] = "block"
        chunk["chunk_id"] = chunk["chunk_id"].replace("_block_", "_block_")
        output.append(chunk)

    for record in record_list:
        sequence = 0
        headings = {"doc_title": "", "paragraph_title": ""}
        for block in _blocks(record):
            label = str(block.get("block_label", "unknown"))
            content = str(block.get("block_content", "")).strip()
            if label in HEADING_LABELS and content:
                headings[label] = _clean(markdown_to_plain_text(content))
                if label == "doc_title":
                    headings["paragraph_title"] = ""
                continue
            if label != "table" or not content:
                continue
            heading = _heading_text(headings)
            for row in serialized_table_rows(content):
                sequence += 1
                output.append(
                    _make_chunk(
                        record,
                        experiment="A3",
                        sequence=sequence,
                        text=_with_heading(row, heading),
                        block_type="table_row",
                        granularity="row",
                        heading=heading,
                    )
                )
        signature = _page_signature(record, limit=chunk_size)
        if signature:
            output.append(
                _make_chunk(
                    record,
                    experiment="A3",
                    sequence=1,
                    text=signature,
                    block_type="page_signature",
                    granularity="page",
                    source_parser="PaddleOCR-VL-1.6",
                )
            )
    return output


def selective_cascade_chunks(
    baseline_pages: list[dict[str, Any]],
    records: Iterable[dict[str, Any]],
    *,
    chunk_size: int = 3200,
    overlap: int = 480,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for chunk in chunk_pages(baseline_pages, chunk_size=chunk_size, overlap=overlap):
        chunk["chunk_id"] = f"a4_narrative_{chunk['chunk_id']}"
        chunk["parser"] = "PyMuPDF"
        chunk["representation"] = "a4"
        chunk["granularity"] = "narrative"
        chunk["block_type"] = "narrative"
        output.append(chunk)

    label_counts: Counter[str] = Counter()
    selected_pages: set[int] = set()
    structured_count = 0
    for record in records:
        headings = {"doc_title": "", "paragraph_title": ""}
        blocks = _blocks(record)
        page_has_complex_region = any(
            str(block.get("block_label")) in {"table", "chart"} for block in blocks
        )
        sequence = 0
        for block in blocks:
            label = str(block.get("block_label", "unknown"))
            content = str(block.get("block_content", "")).strip()
            if label in HEADING_LABELS and content:
                headings[label] = _clean(markdown_to_plain_text(content))
                if label == "doc_title":
                    headings["paragraph_title"] = ""
                continue
            if label not in SELECTIVE_LABELS or not content:
                continue
            if label in {"figure_title", "vision_footnote"} and not page_has_complex_region:
                continue
            heading = _heading_text(headings)
            values = serialized_table_rows(content) if label == "table" else [
                markdown_to_plain_text(content)
            ]
            values = [value for value in values if value]
            for value in values:
                for window in _windows(_with_heading(value, heading), chunk_size, 0):
                    sequence += 1
                    structured_count += 1
                    label_counts[label] += 1
                    selected_pages.add(int(record["printed_page"]))
                    output.append(
                        _make_chunk(
                            record,
                            experiment="A4",
                            sequence=sequence,
                            text=window,
                            block_type=label,
                            granularity="structured_region",
                            heading=heading,
                            source_parser="PaddleOCR-VL-1.6-selective",
                        )
                    )
    stats = {
        "narrative_chunk_count": sum(
            chunk.get("granularity") == "narrative" for chunk in output
        ),
        "structured_chunk_count": structured_count,
        "selected_printed_page_count": len(selected_pages),
        "selected_label_counts": dict(sorted(label_counts.items())),
    }
    return output, stats

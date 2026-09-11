from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Iterable


IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


def _string_values(value: Any) -> list[str]:
    """Return textual leaves without depending on a PaddleX result version."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for nested in value.values():
            values.extend(_string_values(nested))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for nested in value:
            values.extend(_string_values(nested))
        return values
    return []


def extract_markdown(result: Any) -> str:
    """Extract PaddleX's canonical Markdown text from a prediction result."""
    markdown = getattr(result, "markdown", None)
    if callable(markdown):
        markdown = markdown()
    if isinstance(markdown, dict):
        markdown = markdown.get("markdown_texts", markdown)
    values = [value.strip() for value in _string_values(markdown) if value.strip()]
    # Blank/illustration-only pages are valid pipeline results. They remain in
    # the page mapping but intentionally produce no retrieval chunk.
    if not values:
        return ""
    return "\n\n".join(values).strip()


def markdown_to_plain_text(markdown: str) -> str:
    """Flatten Markdown while retaining human-readable table cell contents.

    This conversion is intentionally deterministic. It does not summarize, call
    an LLM, or add information, so A1a and A1b differ only in representation.
    """
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```[^\n]*\n?", "", text)
    text = IMAGE_RE.sub(lambda match: match.group(1), text)
    text = LINK_RE.sub(lambda match: match.group(1), text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|div|tr|table|h[1-6])\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:td|th)\s*>", " ; ", text, flags=re.IGNORECASE)
    text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)

    output: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        if TABLE_SEPARATOR_RE.match(line):
            continue
        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            line = " ; ".join(cell for cell in cells if cell)
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", line)
        line = re.sub(r"(?:\*\*|__|~~|`)", "", line)
        line = re.sub(r"(?<!\\)[*_]", "", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            output.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def json_safe(value: Any) -> Any:
    """Convert PaddleX/numpy values to an auditable JSON-compatible record."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(nested) for nested in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    # PIL images and other binary visualization objects are deliberately not
    # embedded in JSON; their type remains visible for provenance.
    return {"unserialized_type": f"{type(value).__module__}.{type(value).__name__}"}


def load_raw_records(raw_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("fragment_*.json")):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if not isinstance(record.get("markdown"), str):
            raise ValueError(f"Missing Markdown text in {path}")
        records.append(record)
    return sorted(records, key=lambda row: int(row["logical_index"]))


def representation_pages(
    records: Iterable[dict[str, Any]], representation: str
) -> list[dict[str, Any]]:
    if representation not in {"plain", "markdown"}:
        raise ValueError("representation must be 'plain' or 'markdown'")
    pages: list[dict[str, Any]] = []
    for record in records:
        markdown = str(record["markdown"]).strip()
        text = markdown_to_plain_text(markdown) if representation == "plain" else markdown
        pages.append(
            {
                "text": text,
                "logical_index": int(record["logical_index"]),
                "pdf_page_index": int(record["pdf_page_index"]),
                "pdf_page": int(record["pdf_page"]),
                "page_part": record["page_part"],
                "printed_page": int(record["printed_page"]),
                "mapping_method": record.get("mapping_method"),
                "source": record["source"],
                "representation": representation,
                "parser": "PaddleOCR-VL-1.6",
                "ocr_raw_id": record["raw_id"],
            }
        )
    return pages

from __future__ import annotations

from typing import Any


def _windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        target = min(start + chunk_size, len(text))
        end = target
        if target < len(text):
            search_from = start + int(chunk_size * 0.70)
            newline = text.rfind("\n", search_from, target)
            space = text.rfind(" ", search_from, target)
            end = max(newline, space)
            if end <= start:
                end = target
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(
    pages: list[dict[str, Any]], chunk_size: int = 3200, overlap: int = 480
) -> list[dict[str, Any]]:
    """Create character-window chunks without crossing printed-page regions."""
    chunks: list[dict[str, Any]] = []
    optional_metadata = ("representation", "parser", "ocr_raw_id", "logical_index")
    for page in pages:
        for number, text in enumerate(_windows(page.get("text", ""), chunk_size, overlap), 1):
            printed = int(page["printed_page"])
            part = page.get("page_part", "full")
            chunk = {
                    "chunk_id": (
                        f"pdf{int(page['pdf_page']):03d}_{part}_"
                        f"tr{printed:03d}_chunk{number:02d}"
                    ),
                    "text": text,
                    "pdf_page_index": int(page["pdf_page_index"]),
                    "pdf_page": int(page["pdf_page"]),
                    "page_part": part,
                    "printed_page": printed,
                    "mapping_method": page.get("mapping_method"),
                    "source": page["source"],
                }
            for key in optional_metadata:
                if key in page:
                    chunk[key] = page[key]
            chunks.append(chunk)
    return chunks

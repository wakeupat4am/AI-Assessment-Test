from __future__ import annotations

from typing import Any


class ParentExpandingRetriever:
    """Retrieve compact children, then attach bounded source context."""

    def __init__(self, retriever: Any, *, max_parent_characters: int = 1800) -> None:
        self.base = retriever
        self.max_parent_characters = max(0, max_parent_characters)
        self.last_trace = None
        self.last_router_call = None
        self.last_retrieval_calls: list[Any] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        rows = self.base.retrieve(query, top_k)
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            parent = str(item.get("parent_text", "")).strip()
            if parent and self.max_parent_characters and item.get("granularity") in {
                "metric",
                "row",
            }:
                child_text = str(item.get("text", "")).strip()
                bounded_parent = parent[: self.max_parent_characters].rstrip()
                if bounded_parent and bounded_parent not in child_text:
                    item["retrieval_text"] = child_text
                    item["text"] = f"{child_text}\n\nNgữ cảnh nguồn:\n{bounded_parent}"
                    item["parent_expanded"] = True
            output.append(item)
        self.last_trace = getattr(self.base, "last_trace", None)
        self.last_router_call = getattr(self.base, "last_router_call", None)
        self.last_retrieval_calls = list(
            getattr(self.base, "last_retrieval_calls", []) or []
        )
        return output

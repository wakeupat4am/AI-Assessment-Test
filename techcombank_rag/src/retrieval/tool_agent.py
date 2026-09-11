from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.llm.base import GenerationResult, coerce_generation_result


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)*(?:%|x)?")


def normalize_number(value: str) -> str:
    return re.sub(r"\s+", "", value).replace("₫", "")


def parse_vietnamese_decimal(value: str) -> Decimal:
    raw = normalize_number(value).rstrip("%x")
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", raw):
        raw = raw.replace(".", "")
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+", raw):
        raw = raw.replace(",", "")
    elif "," in raw and "." not in raw:
        raw = raw.replace(",", ".")
    elif "," in raw and "." in raw:
        raw = raw.replace(".", "").replace(",", ".")
    try:
        return Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(f"Invalid numeric operand: {value!r}") from error


def format_vietnamese_decimal(value: Decimal, *, decimal_places: int = 2) -> str:
    quantizer = Decimal(1).scaleb(-decimal_places)
    rounded = value.quantize(quantizer)
    rendered = f"{rounded:,.{decimal_places}f}"
    rendered = rendered.replace(",", "_").replace(".", ",").replace("_", ".")
    return rendered.rstrip("0").rstrip(",")


@dataclass(frozen=True)
class ToolAction:
    tool: str
    arguments: dict[str, Any]
    source: str


def _strip_json(text: str) -> dict[str, Any] | None:
    candidate = text.strip().strip("`")
    if candidate.startswith("json"):
        candidate = candidate[4:].strip()
    match = JSON_RE.search(candidate)
    if match:
        candidate = match.group(0)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def load_b5_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "agent_protocol", "initial_top_k", "followup_top_k",
        "max_tool_actions_after_initial_retrieval", "evidence_preview_chunks",
        "evidence_preview_characters_per_chunk", "trigger_patterns",
        "calculation_patterns", "allowed_operations", "tool_names",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"B5 configuration is missing: {', '.join(missing)}")
    return config


class ToolCallingRetriever:
    """A bounded agent that executes audited host tools over real A3.1 evidence.

    The transport is a strict JSON tool-call protocol rather than a provider-specific
    function-calling API, so the same behaviour works for OpenAI-compatible Qwen,
    OpenAI, and Anthropic. The model only chooses an action; Python validates every
    argument and performs the actual retrieval/calculation.
    """

    def __init__(
        self,
        retriever: Any,
        llm_client: Any,
        *,
        config: dict[str, Any],
        log_path: Path | None = None,
    ) -> None:
        self.base = retriever
        self.llm = llm_client
        self.config = config
        self.log_path = Path(log_path) if log_path else None
        self.last_trace: dict[str, Any] | None = None
        self.last_retrieval_calls: list[GenerationResult] = []
        self._last_planner_response: str | None = None
        self.last_router_call = None
        self._log_lock = threading.Lock()

    def _is_triggered(self, query: str) -> bool:
        lowered = query.casefold()
        return any(pattern in lowered for pattern in self.config["trigger_patterns"])

    def _is_calculation_query(self, query: str) -> bool:
        lowered = query.casefold()
        return any(pattern in lowered for pattern in self.config["calculation_patterns"])

    def _preview(self, rows: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        limit = int(self.config["evidence_preview_characters_per_chunk"])
        for index, row in enumerate(rows[: int(self.config["evidence_preview_chunks"])], 1):
            blocks.append(
                f"E{index} | page={int(row['printed_page'])} | id={row['chunk_id']}\n"
                + str(row.get("text", ""))[:limit]
            )
        return "\n\n".join(blocks)

    def _planner_messages(self, query: str, evidence: list[dict[str, Any]]) -> list[dict[str, str]]:
        tool_schema = {
            "hybrid_retrieve": {
                "query": "short focused Vietnamese retrieval query for one missing fact"
            },
            "calculator": {
                "operation": "subtract | percentage_change | cagr",
                "left": {"value": "literal number from evidence", "page": 1},
                "right": {"value": "literal number from evidence", "page": 1},
                "unit": "unit appearing with the values"
            },
            "finish": {},
        }
        system = (
            "Bạn là bộ lập kế hoạch gọi công cụ cho Financial RAG. Không trả lời câu hỏi. "
            "Chỉ xuất một JSON hợp lệ, không markdown. Chỉ chọn đúng một tool trong schema. "
            "calculator chỉ được dùng khi cả hai toán hạng xuất hiện nguyên văn trong evidence, "
            "cùng đơn vị và phép tính trả lời trực tiếp câu hỏi. Nếu thiếu fact, dùng hybrid_retrieve "
            "với một query ngắn, khác query gốc. Với câu hỏi có 'tăng thêm', 'chênh lệch', hoặc "
            "'bao nhiêu phần trăm' và evidence có hai số phù hợp: BẮT BUỘC chọn calculator, không finish. "
            "Nếu evidence đã đủ và không cần tính toán, hoặc không an toàn, dùng finish."
        )
        user = (
            f"Câu hỏi: {query}\n\n"
            f"Tools JSON schema: {json.dumps(tool_schema, ensure_ascii=False)}\n\n"
            f"Evidence thật:\n{self._preview(evidence)}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _plan(self, query: str, evidence: list[dict[str, Any]]) -> ToolAction:
        generated = self.llm.generate(
            self._planner_messages(query, evidence), temperature=0, purpose="agent_planning"
        )
        result = coerce_generation_result(generated, purpose="agent_planning")
        self.last_retrieval_calls.append(result)
        self._last_planner_response = result.text[:1000]
        payload = _strip_json(result.text)
        if not payload:
            return ToolAction("finish", {}, "planner_invalid_json")
        # Some OpenAI-compatible models emit a native-tool-shaped object such
        # as {"hybrid_retrieve": {"query": "..."}} even when explicitly
        # asked for {"tool": "hybrid_retrieve", ...}. Accept that equivalent
        # form, but keep validation and host execution unchanged.
        if "tool" not in payload and len(payload) == 1:
            candidate_tool, candidate_arguments = next(iter(payload.items()))
            if candidate_tool in self.config["tool_names"] and isinstance(candidate_arguments, dict):
                return ToolAction(str(candidate_tool), dict(candidate_arguments), "planner_nested_tool_json")
        tool = str(payload.get("tool", "")).strip()
        if tool not in self.config["tool_names"]:
            return ToolAction("finish", {}, "planner_invalid_tool")
        arguments = {key: value for key, value in payload.items() if key != "tool"}
        return ToolAction(tool, arguments, "planner")

    @staticmethod
    def _find_operand(rows: list[dict[str, Any]], value: str, page: int) -> dict[str, Any] | None:
        normalized = normalize_number(value)
        for row in rows:
            if int(row["printed_page"]) != int(page):
                continue
            text = normalize_number(str(row.get("text", "")))
            if normalized in text:
                return row
        return None

    def _calculate(self, action: ToolAction, rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
        args = action.arguments
        operation = str(args.get("operation", ""))
        if operation not in self.config["allowed_operations"]:
            return None, "operation_not_allowed"
        left = args.get("left")
        right = args.get("right")
        if not isinstance(left, dict) or not isinstance(right, dict):
            return None, "invalid_operands"
        try:
            left_value = str(left["value"])
            right_value = str(right["value"])
            left_page = int(left["page"])
            right_page = int(right["page"])
        except (KeyError, TypeError, ValueError):
            return None, "invalid_operand_fields"
        left_source = self._find_operand(rows, left_value, left_page)
        right_source = self._find_operand(rows, right_value, right_page)
        if left_source is None or right_source is None:
            return None, "operands_not_verbatim_in_evidence"
        try:
            first = parse_vietnamese_decimal(left_value)
            second = parse_vietnamese_decimal(right_value)
        except ValueError as error:
            return None, str(error)
        if operation == "subtract":
            result = first - second
            formula = f"{left_value} − {right_value}"
        elif operation == "percentage_change":
            if second == 0:
                return None, "division_by_zero"
            result = (first - second) / second * Decimal(100)
            formula = f"({left_value} − {right_value}) / {right_value} × 100"
        else:
            years = args.get("years")
            if not isinstance(years, (int, float)) or years <= 0 or second <= 0:
                return None, "invalid_cagr_inputs"
            # Decimal does not support a fractional exponent consistently across
            # Python versions. This is still deterministic; inputs were checked
            # against evidence above and the rendered result is rounded below.
            result = Decimal(str((((float(first) / float(second)) ** (1 / float(years))) - 1) * 100))
            formula = f"({left_value} / {right_value})^(1/{years}) − 1"
        unit = str(args.get("unit", "")).strip()
        suffix = "%" if operation in {"percentage_change", "cagr"} else f" {unit}".rstrip()
        output = format_vietnamese_decimal(result) + suffix
        source_pages = sorted({left_page, right_page})
        source_text = (
            f"Kết quả công cụ calculator (đã kiểm chứng toán hạng): {formula} = {output}. "
            f"Toán hạng {left_value} xuất hiện ở trang {left_page}; "
            f"toán hạng {right_value} xuất hiện ở trang {right_page}."
        )
        digest = hashlib.sha256(
            f"{operation}|{left_value}|{right_value}|{source_pages}".encode()
        ).hexdigest()[:12]
        derived = {
            "chunk_id": f"b5_calculation_{digest}",
            "text": source_text,
            "retrieval_text": source_text,
            "printed_page": left_page,
            "pdf_page": int(left_source.get("pdf_page", left_page)),
            "page_part": left_source.get("page_part", "derived"),
            "score": 1.0,
            "granularity": "metric",
            "block_type": "derived_calculation",
            "representation": "b5",
            "derived": True,
            "tool_name": "calculator",
            "operation": operation,
            "source_pages": source_pages,
            "source_chunk_ids": [left_source["chunk_id"], right_source["chunk_id"]],
        }
        return derived, None

    @staticmethod
    def _merge(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for row in [*first, *second]:
            key = str(row["chunk_id"])
            if key not in merged or float(row.get("score", 0)) > float(merged[key].get("score", 0)):
                merged[key] = dict(row)
        return sorted(merged.values(), key=lambda row: (-float(row.get("score", 0)), str(row["chunk_id"])))

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        started = time.perf_counter()
        requested_k = int(top_k or 10)
        initial_k = int(self.config["initial_top_k"])
        initial = self.base.retrieve(query, initial_k)
        self.last_retrieval_calls = []
        actions: list[dict[str, Any]] = [{
            "tool": "hybrid_retrieve", "arguments": {"query": query, "top_k": initial_k},
            "outcome": "ok", "returned_candidates": len(initial),
        }]
        final = list(initial)
        triggered = self._is_triggered(query)
        planner_source = "not_triggered"
        if triggered and int(self.config["max_tool_actions_after_initial_retrieval"]) >= 1:
            action = self._plan(query, initial)
            planner_source = action.source
            if action.tool == "hybrid_retrieve":
                followup = str(action.arguments.get("query", "")).strip()
                if followup and followup.casefold() != query.casefold():
                    second = self.base.retrieve(followup, int(self.config["followup_top_k"]))
                    final = self._merge(initial, second)
                    actions.append({
                        "tool": "hybrid_retrieve", "arguments": {"query": followup, "top_k": int(self.config["followup_top_k"])},
                        "outcome": "ok", "returned_candidates": len(second),
                    })
                else:
                    actions.append({"tool": "hybrid_retrieve", "arguments": action.arguments, "outcome": "rejected_invalid_followup"})
            elif action.tool == "calculator":
                derived, error = self._calculate(action, initial)
                if derived is not None:
                    final = [derived, *initial]
                    actions.append({
                        "tool": "calculator", "arguments": action.arguments, "outcome": "ok",
                        "result": derived["text"], "source_pages": derived["source_pages"],
                    })
                else:
                    actions.append({"tool": "calculator", "arguments": action.arguments, "outcome": f"rejected:{error}"})
            else:
                actions.append({"tool": "finish", "arguments": {}, "outcome": "ok"})
        output = final[:requested_k]
        trace = {
            "protocol": self.config["agent_protocol"],
            "query": query,
            "triggered": triggered,
            "planner_source": planner_source,
            "actions": actions,
            "tool_calls_executed": len(actions),
            "planner_calls": len(self.last_retrieval_calls),
            "planner_response": self._last_planner_response,
            "final_evidence_count": len(output),
            "final_pages": [int(row["printed_page"]) for row in output],
            "latency_seconds": round(time.perf_counter() - started, 6),
        }
        self.last_trace = {"b5": trace}
        if self.log_path:
            payload = {"created_at_utc": datetime.now(timezone.utc).isoformat(), **trace}
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_lock, self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return output

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from src.chat.query_rewriter import rewrite_query
from src.config import Settings, get_settings
from src.llm.base import GenerationResult, aggregate_calls, coerce_generation_result
from src.llm.provider import create_llm_client
from src.retrieval.retriever import DenseRetriever


REFUSAL_TEXT = "Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này."
CITATION_RE = re.compile(r"\[tr\.\s*(\d+(?:\s*,\s*\d+)*)\]", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![\w])\d+(?:\.\d{3})*(?:,\d+)?%?")

SYSTEM_PROMPT = """Bạn là trợ lý cho chuyên viên phân tích quan hệ nhà đầu tư.
Chỉ trả lời từ bằng chứng trong Báo cáo thường niên được cung cấp.

Quy tắc:
1. Trả lời bằng tiếng Việt.
2. Không dùng kiến thức được huấn luyện trước làm bằng chứng cho dữ kiện của báo cáo.
3. Mỗi khẳng định thực tế phải có trích dẫn trang như [tr. 45].
4. Chỉ dùng số trang in được ghi trong metadata bằng chứng.
5. Không bao giờ tự tạo số trang.
6. Giữ nguyên số liệu và đơn vị.
7. Nếu bằng chứng không đủ, trả lời đúng câu: "Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này."
8. Không đoán.
9. Nếu nguồn mâu thuẫn, nêu rõ mâu thuẫn.
10. Chỉ trả lời trực tiếp điều được hỏi trong tối đa 2 câu; nếu câu hỏi chỉ yêu cầu một giá trị, trả lời đúng một câu chứa tên chỉ tiêu, giá trị và citation.
11. Trước khi trả lời, phải xác nhận bằng chứng nêu đúng chỉ tiêu và đúng kỳ/năm được hỏi.
12. Với bảng bị trích xuất thành dòng chữ, chỉ ghép giá trị với đúng nhãn/cột tương ứng; giữ nguyên tên nhãn trong câu trả lời và không chọn một con số chỉ vì nó đứng gần trong cùng chunk.
13. Không gán mức tăng trưởng, số dư, đơn vị hoặc kỳ/năm của một chỉ tiêu cho chỉ tiêu khác đứng gần nó trong bảng/văn bản.
14. Nếu thiếu đúng chỉ tiêu hoặc đúng kỳ/năm, dùng đúng câu từ chối ở quy tắc 7; không thay bằng chỉ tiêu hay kỳ khác.
15. EVIDENCE 1 là kết quả retrieval mạnh nhất. Khi bảng phẳng gây mơ hồ, ưu tiên cặp nhãn-giá trị rõ ràng ở evidence có rank cao hơn và chỉ dùng evidence thấp hơn để đối chiếu.
16. Đặt citation ngay trong cùng câu chứa khẳng định; không liệt kê citation thành bullet riêng.
17. Không thêm số dư tương ứng, so sánh, tăng trưởng hoặc phép quy đổi nếu câu hỏi chỉ hỏi một giá trị.
18. Trả lời ngắn gọn, phù hợp với chuyên viên quan hệ nhà đầu tư."""


def extract_citations(answer: str) -> list[int]:
    pages: list[int] = []
    for match in CITATION_RE.finditer(answer):
        for value in match.group(1).split(","):
            page = int(value.strip())
            if page not in pages:
                pages.append(page)
    return pages


def validate_citations(answer: str, evidence_pages: set[int]) -> tuple[list[int], bool]:
    citations = extract_citations(answer)
    return citations, bool(citations) and set(citations).issubset(evidence_pages)


def validate_numeric_support(
    answer: str, chunks: list[dict[str, Any]], citations: list[int]
) -> bool:
    """Require generated numeric values to occur verbatim on a cited printed page."""
    if not citations:
        return False
    answer_without_citations = CITATION_RE.sub("", answer)
    values = [
        value.replace(" ", "")
        for value in NUMBER_RE.findall(answer_without_citations)
        if not (len(value) == 4 and value.isdigit() and 1900 <= int(value) <= 2100)
    ]
    cited_text = " ".join(
        chunk["text"] for chunk in chunks if int(chunk["printed_page"]) in citations
    ).replace(" ", "")
    return all(value in cited_text for value in values)


def format_evidence(chunks: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    # Put rank 1 nearest to the final question. This preserves all top-k evidence while
    # reducing recency bias toward the least relevant chunk in a long prompt.
    for index in range(len(chunks), 0, -1):
        chunk = chunks[index - 1]
        rank_label = " — HIGHEST RETRIEVAL RANK" if index == 1 else ""
        blocks.append(
            f"[EVIDENCE {index}{rank_label}]\n"
            f"Printed page: {chunk['printed_page']}\n"
            f"PDF page: {chunk['pdf_page']} ({chunk.get('page_part', 'full')})\n"
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Retrieval score: {chunk['score']:.4f}\n"
            f"Text:\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return REFUSAL_TEXT.lower() in lowered or "không đủ thông tin" in lowered


def build_answer_prompt(question: str, evidence: str) -> str:
    """Keep the final instruction next to the question so long evidence cannot bury it."""
    return (
        f"Bằng chứng:\n{evidence}\n\n"
        f"CÂU HỎI CẦN TRẢ LỜI: {question}\n\n"
        "Chỉ trả lời câu hỏi này trong tối đa 2 câu. Trước khi trả lời, hãy kiểm tra "
        "bằng chứng có nêu đúng chỉ tiêu và đúng kỳ/năm được hỏi. Nếu thiếu một trong "
        "hai, dùng đúng câu từ chối; không thay bằng chỉ tiêu hoặc kỳ/năm khác. Bắt đầu "
        "từ EVIDENCE 1 (rank cao nhất), giữ nguyên tên nhãn và ghép đúng nhãn với đúng "
        "giá trị. Nếu câu hỏi chỉ hỏi một giá trị, trả lời đúng một câu và không thêm "
        "giá trị liên quan khác."
    )


class Chatbot:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: Any | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm_client or create_llm_client(self.settings)
        if retriever is not None:
            base_retriever = retriever
        elif self.settings.xrouter_enabled:
            from src.retrieval.routed_retriever import RoutedRetriever

            base_retriever = RoutedRetriever(self.settings)
        else:
            base_retriever = DenseRetriever(self.settings)
        if self.settings.b4_retrieval_mode != "off":
            if self.settings.xrouter_enabled or self.settings.hyde_enabled:
                raise ValueError(
                    "B4_RETRIEVAL_MODE cannot be combined implicitly with X-Router "
                    "or HyDE; use an explicit experimental composition."
                )
            from src.retrieval.b4_factory import create_b4_retriever
            from src.retrieval.parent_expansion import ParentExpandingRetriever

            base_retriever = ParentExpandingRetriever(
                create_b4_retriever(base_retriever, self.settings),
                max_parent_characters=self.settings.a31_parent_characters,
            )
        if self.settings.hyde_enabled:
            if self.settings.xrouter_enabled:
                raise ValueError(
                    "ENABLE_HYDE and XROUTER_ENABLED cannot both be enabled in the "
                    "default pipeline; use an explicit fusion retriever for experiments."
                )
            from src.retrieval.hyde import HyDERetriever

            base_retriever = HyDERetriever(
                base_retriever,
                settings=self.settings,
                mode=self.settings.hyde_mode,
                prompt_variant=self.settings.hyde_prompt_variant,
                log_path=self.settings.hyde_log_path,
            )
        if self.settings.b5_agent_enabled:
            if self.settings.xrouter_enabled or self.settings.hyde_enabled:
                raise ValueError(
                    "B5_AGENT_ENABLED cannot be combined implicitly with X-Router or HyDE."
                )
            if self.settings.b4_retrieval_mode != "hybrid":
                raise ValueError("B5_AGENT_ENABLED requires B4_RETRIEVAL_MODE=hybrid")
            from src.retrieval.tool_agent import ToolCallingRetriever, load_b5_config

            base_retriever = ToolCallingRetriever(
                base_retriever,
                self.llm,
                config=load_b5_config(self.settings.b5_agent_config_path),
                log_path=self.settings.b5_agent_log_path,
            )
        b2_requested = self.settings.b2_enabled or any(
            (
                self.settings.enable_auto_search,
                self.settings.enable_page_dedup,
                self.settings.enable_diversity_selection,
            )
        )
        if b2_requested:
            from src.retrieval.auto_search import AdaptiveRetriever

            if isinstance(base_retriever, AdaptiveRetriever):
                self.retriever = base_retriever
            else:
                self.retriever = AdaptiveRetriever(
                    base_retriever,
                    enable_auto_search=self.settings.enable_auto_search,
                    enable_page_dedup=self.settings.enable_page_dedup,
                    enable_diversity_selection=self.settings.enable_diversity_selection,
                    initial_top_k=self.settings.b2_initial_top_k,
                    second_round_top_k=self.settings.b2_second_round_top_k,
                    max_search_rounds=self.settings.b2_max_search_rounds,
                    max_per_page=self.settings.b2_max_per_page,
                    similarity_threshold=self.settings.b2_page_similarity_threshold,
                    final_k=self.settings.b2_final_k,
                    max_evidence_tokens=self.settings.b2_max_evidence_tokens,
                    lambda_relevance=self.settings.b2_diversity_lambda,
                    log_path=self.settings.b2_log_path,
                )
        else:
            self.retriever = base_retriever

    def ask(
        self, question: str, conversation_history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        started = time.perf_counter()
        history = conversation_history or []
        llm_calls: list[GenerationResult] = []
        standalone = rewrite_query(
            question,
            history,
            client=self.llm,
            max_history_turns=self.settings.max_history_turns,
            trace_sink=llm_calls,
        )
        retrieval_started = time.perf_counter()
        candidate_k = max(self.settings.top_k, self.settings.evaluation_retrieval_k)
        if hasattr(self.retriever, "retrieve_routed"):
            conversation_context = (
                json.dumps(history, ensure_ascii=False) if history else None
            )
            candidates = self.retriever.retrieve_routed(
                standalone,
                candidate_k,
                conversation_context=conversation_context,
                router_query=question,
            )
        else:
            candidates = self.retriever.retrieve(standalone, candidate_k)
        routing = getattr(self.retriever, "last_trace", None)
        retrieval_calls = list(
            getattr(self.retriever, "last_retrieval_calls", []) or []
        )
        llm_calls.extend(retrieval_calls)
        router_call = getattr(self.retriever, "last_router_call", None)
        if router_call is not None and all(
            router_call is not call for call in retrieval_calls
        ):
            llm_calls.append(router_call)
        selected_evidence = getattr(self.retriever, "last_selected_evidence", None)
        chunks = (
            list(selected_evidence)
            if selected_evidence is not None
            else candidates[: self.settings.top_k]
        )
        retrieval_seconds = time.perf_counter() - retrieval_started
        top_score = chunks[0]["score"] if chunks else float("-inf")
        b2_trace = routing.get("b2", {}) if routing else {}
        b2_insufficient = (
            bool(b2_trace)
            and bool(getattr(self.retriever, "enable_auto_search", False))
            and not bool(b2_trace.get("sufficient"))
        )
        if not chunks or top_score < self.settings.min_retrieval_score or b2_insufficient:
            result = self._result(
                question,
                standalone,
                REFUSAL_TEXT,
                [],
                candidates,
                started,
                True,
                True,
                llm_calls,
                retrieval_seconds,
                routing,
            )
            self._debug(result, "")
            return result

        evidence = format_evidence(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_answer_prompt(question, evidence),
            },
        ]
        generated = self.llm.generate(
            messages,
            temperature=self.settings.llm_temperature,
            purpose="answer",
        )
        answer_result = coerce_generation_result(generated, purpose="answer")
        llm_calls.append(answer_result)
        answer = answer_result.text
        evidence_pages = {int(chunk["printed_page"]) for chunk in chunks}
        refused = _looks_like_refusal(answer)
        citations, citation_valid = validate_citations(answer, evidence_pages)
        if citation_valid:
            citation_valid = validate_numeric_support(answer, chunks, citations)
        if refused:
            citation_valid = not citations or set(citations).issubset(evidence_pages)
        elif not citation_valid:
            answer, repair_result = self._repair_citations(
                question, evidence, answer, evidence_pages
            )
            llm_calls.append(repair_result)
            refused = _looks_like_refusal(answer)
            citations, citation_valid = validate_citations(answer, evidence_pages)
            if citation_valid:
                citation_valid = validate_numeric_support(answer, chunks, citations)
            if refused:
                citation_valid = not citations or set(citations).issubset(evidence_pages)

        if not refused and not citation_valid:
            answer = (
                "Không thể tạo câu trả lời với trích dẫn hợp lệ từ bằng chứng được truy xuất."
            )
            citations = []
            refused = True

        result = self._result(
            question,
            standalone,
            answer,
            citations,
            candidates,
            started,
            refused,
            citation_valid,
            llm_calls,
            retrieval_seconds,
            routing,
        )
        self._debug(result, evidence)
        return result

    def _repair_citations(
        self, question: str, evidence: str, previous: str, evidence_pages: set[int]
    ) -> tuple[str, GenerationResult]:
        allowed = ", ".join(str(page) for page in sorted(evidence_pages))
        prompt = (
            f"Câu trả lời trước có trích dẫn thiếu hoặc không hợp lệ:\n{previous}\n\n"
            f"Chỉ các trang sau được phép trích dẫn: {allowed}.\n"
            "Hãy trả lời lại câu hỏi chỉ từ bằng chứng, với trích dẫn hợp lệ cho mỗi "
            "khẳng định thực tế. Mỗi số liệu phải xuất hiện nguyên văn trong text của "
            "đúng trang được trích dẫn. Nếu không đủ bằng chứng, dùng đúng câu từ chối.\n\n"
            f"Bằng chứng:\n{evidence}\n\n"
            f"CÂU HỎI CẦN TRẢ LỜI: {question}\n\n"
            "Chỉ trả lời câu hỏi này trong tối đa 2 câu. Kiểm tra đúng chỉ tiêu và "
            "đúng kỳ/năm; nếu thiếu, dùng đúng câu từ chối."
        )
        generated = self.llm.generate(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0,
            purpose="repair",
        )
        result = coerce_generation_result(generated, purpose="repair")
        return result.text, result

    @staticmethod
    def _result(
        question: str,
        standalone: str,
        answer: str,
        citations: list[int],
        chunks: list[dict[str, Any]],
        started: float,
        refused: bool,
        citation_valid: bool,
        llm_calls: list[GenerationResult],
        retrieval_seconds: float,
        routing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        llm_summary = aggregate_calls(llm_calls)
        total_seconds = time.perf_counter() - started
        result = {
            "question": question,
            "standalone_query": standalone,
            "answer": answer,
            "citations": citations,
            "citation_valid": citation_valid,
            "retrieved_chunks": chunks,
            "latency_seconds": round(total_seconds, 3),
            "latency_breakdown": {
                "retrieval_seconds": round(retrieval_seconds, 6),
                "llm_seconds": round(sum(call.latency_seconds for call in llm_calls), 6),
                "other_seconds": round(
                    max(
                        0.0,
                        total_seconds
                        - retrieval_seconds
                        - sum(call.latency_seconds for call in llm_calls),
                    ),
                    6,
                ),
            },
            "llm": llm_summary,
            "refused": refused,
        }
        if routing is not None:
            result["routing"] = routing
        return result

    def _debug(self, result: dict[str, Any], evidence: str) -> None:
        if not self.settings.debug:
            return
        payload = dict(result)
        payload["evidence_input"] = evidence
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        path = Path(self.settings.processed_dir) / "debug_queries.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

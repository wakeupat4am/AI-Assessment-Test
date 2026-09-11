# B1a — deterministic rule-based X-Router

## Hypothesis

Chọn representation theo intent trước retrieval có thể cải thiện recall so với
luôn dùng A2, mà không thêm LLM call và không thay đổi answer generation.

## Frozen downstream controls

- Query embedding: `intfloat/multilingual-e5-small`.
- Retrieval: normalized dense embedding + FAISS `IndexFlatIP`, top-10 candidates,
  top-5 evidence cho generator.
- Generator: Qwen3.5-9B, temperature 0; prompt, refusal threshold, citation
  validator và repair flow giữ nguyên.
- Baseline arm: mọi query luôn dùng full A2 structured index.
- Router arm: chỉ thay retrieval path. Router không gọi LLM.

## Route contract và index mapping

| Route | Index/path | Ý nghĩa |
|---|---|---|
| `narrative` | B1a narrative index, 837 A2 paragraph/section chunks | Câu mô tả, giải thích, chính sách, chiến lược |
| `structured` | A2, 1.111 heading/table-aware chunks | Tra số, năm, financial metric, row/column/table lookup |
| `multi_repr` | A3, 3.462 row/block/page chunks | Query mơ hồ cần precise fact và surrounding context |
| `multi_hop` | A2 structured + B1a narrative; deterministic RRF | So sánh, nhiều thực thể, nhiều kỳ hoặc fact + explanation |

B1a narrative index không OCR hoặc embed lại. Script lọc 837
`narrative/fallback_page` chunks từ A2 và reconstruct đúng vectors đã freeze.
Chunk ID giống nhau được deduplicate khi fusion để không chiếm hai vị trí top-k.

## Router

Input:

```json
{"query": "...", "conversation_context": "optional"}
```

Output giữ đúng contract `route`, `confidence`, `reasons` và tám boolean
`query_features`. Feature patterns, route weights, priority, threshold 0,62,
fallback và index mapping nằm duy nhất trong `config/xrouter_b1a.json`.

Nếu winning confidence thấp hơn threshold, route bắt buộc chuyển sang
`multi_repr`. Mỗi decision được append JSONL với query, route, confidence,
features, reasons, config SHA-256 và router latency. Router deterministic, không
có model/token cost.

## Verification

- 24 representative Vietnamese routing cases, gồm toàn bộ ví dụ của đề xuất.
- Follow-up/context, low-confidence fallback, JSONL logging, route dispatch,
  candidate counting và multi-index RRF đều có tests.
- Toàn repository: **52 tests passed** trên local và server.
- Narrative index: 837 chunks, 384 dimensions, checksum verified.
- Public/dev ablation: không có runtime error.

## Public ablation — 10 organizer questions

Public answer accuracy đã được human-review bằng cùng labels của A2 vì answer và
citation của hai arm giống nhau; chỉ thứ tự hai tail candidates của SQ-07 đổi.

| Metric | Always A2 | B1a X-Router | Delta |
|---|---:|---:|---:|
| Hit@1 | 0,333333 | 0,333333 | 0 |
| Hit@5 | 0,444444 | 0,444444 | 0 |
| Hit@10 | 0,666667 | 0,666667 | 0 |
| Answer accuracy (manual) | 0,700000 | 0,700000 | 0 |
| Retrieval p50 | 58,96 ms | 55,86 ms | -3,10 ms |
| Retrieval p95 | 108,97 ms | 115,87 ms | +6,90 ms |
| Router p50 / p95 | — | 0,61 / 1,21 ms | — |
| Raw candidates/query | 10 | 13 | +3 |
| Returned candidates/query | 10 | 10 | 0 |
| Input tokens | 33.653 | 33.653 | 0 |

Route distribution: 7 `structured`, 3 `multi_hop`. Public không có câu
qualitative thuần, nên `narrative` không được kích hoạt.

## Dev ablation — 20 diagnostic questions

Dev answer score dưới đây là automatic diagnostic, chưa human-review.

| Metric | Always A2 | B1a X-Router | Delta |
|---|---:|---:|---:|
| Hit@1 | 0,166667 | 0,166667 | 0 |
| Hit@5 | 0,611111 | **0,722222** | **+0,111111** |
| Hit@10 | 0,611111 | **0,722222** | **+0,111111** |
| Answer accuracy (automatic) | 0,150000 | 0,150000 | 0 |
| Retrieval p50 | 45,58 ms | 46,35 ms | +0,77 ms |
| Retrieval p95 | 77,01 ms | 75,96 ms | -1,04 ms |
| Router p50 / p95 | — | 0,47 / 0,67 ms | — |
| Raw candidates/query | 10 | 11 | +1 |
| Returned candidates/query | 10 | 10 | 0 |
| Input tokens | 61.484 | 59.736 | -1.748 |

Route distribution: 13 `structured`, 5 `multi_repr`, 2 `multi_hop`, 0
`narrative`. B1a cứu retrieval cho `dev-09` và `dev-10`, tương đương thêm 2/18
answerable questions có gold page trong top-10. Với `dev-10`, gold page 151 đã
được retrieve nhưng generator vẫn trả sai 3 thay vì 9 thành viên HĐQT. Vì vậy
retrieval tăng nhưng answer accuracy chưa tăng.

## Decision

B1a đạt mục tiêu của một router baseline: deterministic, explainable, dưới 1 ms,
không tốn token và tăng dev Hit@5/10 11,11 điểm phần trăm. Public không đổi nên
không có bằng chứng overclaim rằng router luôn tốt hơn A2.

Giữ B1a làm control cho B1b. B1b chỉ thay `RuleBasedXRouter` bằng LLM router theo
cùng input/output contract; routed retriever, index mapping, answer generation và
evaluation harness phải giữ nguyên để đo đúng giá trị của agentic decision.

## Reproduce

```bash
make build-b1a-index
make ablate-b1a
make ablate-b1a-dev
```

Primary artifacts nằm trong `data/evaluation/experiments/b1a/`, gồm row files,
summaries, router JSONL logs và `b1a-ablation-{public,dev}.json`.

# A3.2 + B4e: metric-aware error repair

## Why this exists

The deployable control remains **A3.1 + B4b**. It confused three similarly named
facts: cash receipts from services on printed page 301, financial-note service
income on page 356, and NFI on page 50. In multi-turn chat the generic rewriter
also shortened `thu nhập từ hoạt động dịch vụ nhận được` to `thu nhập từ hoạt
động dịch vụ`, so later turns drifted to a different metric.

A3.2+B4e is a separate candidate, derived from the frozen A3.1 chunks without
rerunning OCR. The frozen control is recorded in `config/a31_b4b_frozen.json`;
its chunk, FAISS, and BM25 SHA-256 hashes remained unchanged after the A3.2 build.

## Exact changes

1. A conservative search-only OCR normalization fixes isolated tokens such as
   `tử -> từ` and `đông -> đồng`; original evidence text is never overwritten.
2. Each A3.2 chunk receives a metric label, statement type, qualifiers, and years.
3. B4e reranks a bounded dense+BM25 pool using metric-label overlap, exact
   qualifier, requested year, and statement type.
4. A selective gate sends only ambiguity-risk queries to A3.2+B4e. All other
   questions use the exact frozen A3.1+B4b retriever.
5. The multi-turn rewriter anchors follow-ups to the previous user metric, never
   to possibly incorrect assistant text.
6. When one source row unambiguously contains the requested label, years, values,
   and a consistent unit, a deterministic answerer reads it directly. A two-year
   difference and percentage are computed with `Decimal`. Any failed constraint
   falls back to the existing grounded LLM/refusal path.

## Commands

The shipped A3.2 index is already built. Rebuild only for research:

```bash
make build-a32
make benchmark-metric-aware
make benchmark-metric-multiturn
```

Run the fixed candidate on the server:

```bash
INDEX_DIR=techcombank_rag/data/index/a32_metric_aware/all \
B4_RETRIEVAL_MODE=metric_aware \
B4_CONFIG=techcombank_rag/config/b4e_metric_aware.json \
B4_BM25_ARTIFACT=techcombank_rag/data/index/a32_metric_aware/all/bm25.json.gz \
B4_FROZEN_INDEX_DIR=techcombank_rag/data/index/a31_semantic_multirepr/all \
B4_FROZEN_CONFIG=techcombank_rag/config/b4_retrieval.json \
techcombank_rag/.venv/bin/python techcombank_rag/scripts/chat.py
```

## Final server results

All comparisons use the same Qwen/Qwen3.5-9B endpoint, multilingual-E5-small,
questions, gold printed pages, and evidence top-k. Public/dev/holdout below are
retrieval-only regression gates.

| Set / metric | Frozen A3.1+B4b | Fixed A3.2+B4e |
|---|---:|---:|
| Public Hit@1 / @5 / @10 | 66.67 / 100 / 100% | 66.67 / 100 / 100% |
| Dev Hit@1 / @5 / @10 | 55.56 / 77.78 / 88.89% | 55.56 / 77.78 / 88.89% |
| Holdout Hit@1 / @5 / @10 | 77.78 / 77.78 / 88.89% | 77.78 / 77.78 / 88.89% |
| Metric-confusion Hit@1 / @5 / @10 | 50 / 80 / 100% | 90 / 100 / 100% |
| Metric-confusion manual answer accuracy | 60% | 100% |
| Metric-confusion citation precision / recall | 60 / 60% | 100 / 100% |
| Metric-confusion mean end-to-end latency | 5.439 s | 2.409 s |
| Metric-confusion LLM calls / 10 questions | 10 | 4 |
| Metric-confusion total tokens | 46,266 | 18,760 |

The automatic token-F1 answer score is 0% versus 20%. It is not the primary
metric here because its number tokenizer splits Vietnamese thousands separators;
the 10/10 fixed score is a per-item human semantic review stored in
`manual-review-final.json`. This targeted diagnostic does not predict hidden-test
accuracy.

On the seven-turn multi-turn diagnostic, exact numeric coverage and correct-page
citation overlap improved from **14.29% to 71.43%**. Mean latency fell from
**6.067 s to 1.620 s**, mainly because five unambiguous table lookups/calculations
did not require a generation call. The two remaining failures concern the more
generic `tiền gửi của khách hàng tại 31/12` wording and are intentionally retained
as evidence that the selective gate is conservative.

The originally reported conversation is now:

```text
2025 -> 8.396.946 triệu đồng [tr. 301]
2024 follow-up -> 7.679.933 triệu đồng [tr. 301]
change -> +717.013 triệu đồng, khoảng +9,34% [tr. 301]
```

## Trade-offs

- A second shipped index adds about 22 MiB and took 561.28 seconds to derive and
  embed on the recorded server run; it adds no OCR/API charge.
- Metric-aware retrieval checks up to 40 candidates only for gated queries.
- Deterministic row answering is safer and cheaper for exact tables, but its
  parser intentionally declines irregular or cross-row layouts.
- Because the regression sets tie rather than improve, A3.1+B4b stays preserved
  as the clean control. A3.2+B4e is an independently selectable repair candidate.

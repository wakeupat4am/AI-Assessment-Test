# A3.1 — Semantic multi-representation

## Outcome

A3.1 is complete and shipped with its FAISS indexes. It reuses the frozen PaddleOCR-VL-1.6 output and A3 row/block/page chunks; OCR is not rerun. It adds generic metric-value extraction, inherited document headings, document/reporting scope, child-first retrieval, and bounded parent expansion.

The controlled comparison fixes A3.1 and the answer generator across all arms. Only the retrieval controller changes:

```text
PaddleOCR-VL raw + frozen A3 row/block/page
  -> infer section path across pages
  -> extract metric/value/year/unit/detail children
  -> add document entity + reporting period
  -> embed once
  -> materialize all / narrative / structured FAISS views
  -> retrieve compact child
  -> expand parent context after retrieval
  -> B0 one-shot | B1a rule router | B1b LLM router | B2 adaptive
  -> unchanged Qwen answer + printed-page citations
```

`B0` below means the old one-shot controller running on A3.1, not the original A0/PyMuPDF document representation. This keeps the controller ablation fair.

## What A3.1 changes

1. It infers a conservative section path in document order. Generic headings such as “Điểm nhấn 2025” inherit the preceding specific section, which binds page 59 to “Khối Ngân hàng Bán lẻ (RBG)”.
2. It extracts label/value groups from layout coordinates first, structured multiline blocks second, and Markdown sequence fallback third. No public-question metric allowlist is used.
3. Each semantic child stores metric, value, details, numbers, years, unit, source fragment, printed page, parent ID, confidence, and extraction method.
4. `document_entity=Techcombank` and `reporting_year=2025` are build metadata, not query-specific rules. Explicit company or business-unit headings override/refine document scope.
5. Dense retrieval indexes the concise child. Parent context is appended only after retrieval and is capped at 1,800 characters in the benchmark.
6. One embedding pass creates three views from the same vectors: `all`, `narrative`, and `structured`. The route views therefore do not introduce embedding drift.

## Built artifacts

| Property | Value |
|---|---:|
| PDF sheets | 197 |
| OCR fragments reused | 393 |
| A3 source chunks | 3,462 |
| A3.1 total chunks | 3,658 |
| New semantic metric children | 196 |
| Chunks with inherited headings | 82 |
| Layout-coordinate extractions | 94 |
| Structured multiline extractions | 47 |
| Markdown fallback extractions | 55 |
| `all` index | 3,658 chunks |
| `narrative` index | 1,214 chunks |
| `structured` index | 2,444 chunks |
| Embedding | `intfloat/multilingual-e5-small`, 384 dimensions, CPU |
| Final build time | 483.753 seconds |
| OCR/LLM calls during build | 0 |

All six local index payloads (`chunks.jsonl` and `index.faiss` for three views) match the SHA-256 values in `manifest.json`. Shipped size is approximately 34 MiB.

## Retrieval smoke check

Before answer generation, the three known representation failures were checked against the final index:

| Query | Gold evidence | A3.1 result |
|---|---:|---:|
| Total assets at 31/12/2025 | page 5 | rank 2 |
| Total operating income and 2018–2025 CAGR | page 5 | rank 1 |
| Retail Banking loan balance and YoY growth | page 59 | rank 1 |

The page-59 child contains both `328.1 trillion VND` and `26.9% YoY` and inherits the RBG heading. These ranks were measured before any answer prompt.

## Controlled public results (10 questions)

Nine answerable questions are the denominator for retrieval and citation metrics. Automatic accuracy is the existing deterministic heuristic: complete numeric recall plus token-F1 at least 0.45; it is not exact string match, but it penalizes valid concise answers when `gold_answer` contains unasked descriptive text.

| Controller | Automatic | Manual semantic | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | p50/p95 s | Calls | Tokens | Projected GPT-5.6 | Projected Claude Sonnet 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A3.1 + B0 one-shot | 0.40 | 1.00 | 0.5556 | **1.0000** | **1.0000** | 0.6667/0.6667 | 1.00 | 5.282/7.420 | 10 | 36,077 | $0.15026 | $0.11270 |
| A3.1 + B1a rule | 0.40 | 0.90 | 0.5556 | 0.7778 | 0.8889 | 0.6667/0.6667 | 0.90 | 5.341/8.343 | 11 | 42,271 | $0.17558 | $0.13169 |
| A3.1 + B1b LLM | 0.30 | 0.80 | 0.5556 | 0.7778 | 0.7778 | 0.6667/0.6667 | 0.80 | 8.394/10.869 | 21 | 45,981 | $0.19434 | $0.14576 |
| A3.1 + B2 adaptive | 0.30 | 0.70 | 0.2222 | 0.6667 | 0.8889 | 0.5000/0.5556 | 0.90 | 5.089/7.483 | 8 | 33,344 | $0.13886 | $0.10415 |

Manual semantic scoring checks whether every fact explicitly requested by the question is correct. It accepts a different report page only after the retrieved text on that page is verified to contain the claim. The complete rubric is in `a31-manual-review-public.json`.

Public retrieval latency means were 62.0 ms (B0), 71.7 ms (B1a), 3,059.9 ms (B1b, including its router call), and 95.3 ms (B2). B2 averaged 1.2 rounds and triggered round two on 20% of questions.

## Controlled dev results (20 questions)

| Controller | Automatic | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | p50/p95 s | Calls | Tokens | Projected GPT-5.6 | Projected Claude Sonnet 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A3.1 + B0 one-shot | **0.25** | 0.3889 | **0.5556** | **0.7778** | **0.4444/0.4444** | **0.75** | 4.877/8.054 | 21 | **67,358** | $0.28170 | $0.21128 |
| A3.1 + B1a rule | 0.20 | 0.3889 | 0.5000 | 0.6111 | 0.3333/0.3333 | 0.60 | 3.925/11.386 | 22 | 80,374 | $0.33398 | $0.25048 |
| A3.1 + B1b LLM | **0.25** | **0.4444** | **0.5556** | 0.6667 | 0.3889/0.3889 | 0.65 | 7.033/8.822 | 41 | 83,031 | $0.35193 | $0.26395 |
| A3.1 + B2 adaptive | 0.15 | 0.0000 | 0.5000 | 0.6111 | 0.3611/0.3889 | 0.60 | 4.442/12.660 | 18 | 72,727 | $0.30294 | $0.22721 |

B2 averaged 1.15 rounds on dev and triggered round two on 15%. Runtime errors were zero in all eight public/dev runs.

## Interpretation

- A3.1 materially fixes document representation: against the original A0+B0 public baseline, strict Hit@5 rises from 0.4444 to 1.0000 and Hit@10 from 0.5556 to 1.0000.
- The public B0 outputs answer all ten questions correctly under a question-focused manual rubric. Automatic accuracy reports 0.40 because several gold answers contain extra facts not requested in the question.
- Canonical citation P/R remains 0.6667 because three correct answers cite direct supporting duplicates on pages 27, 43, and 53 instead of canonical pages 4, 4, and 5. This is a gold-page coverage issue, not fabricated evidence.
- Neither router improves the fixed A3.1 pipeline. B1a narrows away useful evidence on dev. B1b adds roughly three seconds retrieval/router latency, doubles calls, and introduces an additional public false refusal.
- B2 solves representation crowding: duplicate page slots fall from 4.50 to 0.30 public and 3.65 to 0.40 dev. Its current MMR/sufficiency policy is too aggressive, however, and demotes direct evidence for questions 1, 3, and 4.
- Therefore the current deployment candidate is **A3.1 + B0 one-shot**. Keep B1/B2 behind feature flags until retuned on dev and frozen before holdout.

## Observed failure cases and next iteration

1. Repeated row chunks from printed pages 379 and similar appendices can still crowd B0 top-k. Add page dedup without the current diversity reorder as the safest next ablation.
2. Structured-only routing hurts questions whose answer is present in narrative blocks. The router should select multiple views based on evidence need, not force a single exclusive index.
3. B2 confuses a plausible value with the requested metric. Sufficiency must bind `(entity, metric, year, value, unit)` rather than merely detect years/numbers.
4. Metric OCR typos such as `Du nợ vay` remain searchable but should be normalized in an alias field while preserving the original text for citation.
5. Gold citation pages should list every manually verified supporting page; otherwise correct direct citations are scored as misses.

## Reproduction

From repository root:

```bash
make build-a31
make benchmark-a31
make benchmark-a31-dev
```

Direct server commands:

```bash
cd /home/ubuntu/TCB_Test
PYTHONPATH=/home/ubuntu/TCB_Test/techcombank_rag \
  /home/ubuntu/techcombank_rag/.venv/bin/python \
  techcombank_rag/scripts/build_a31_index.py \
  --output-root techcombank_rag/data/index/a31_semantic_multirepr

PYTHONPATH=/home/ubuntu/TCB_Test/techcombank_rag \
  /home/ubuntu/techcombank_rag/.venv/bin/python \
  techcombank_rag/scripts/benchmark_a31_controllers.py \
  techcombank_rag/data/evaluation/public.json --run-label public
```

## Files added or changed

- `src/ingestion/semantic_multirepr.py`: generic extraction, heading inheritance, scope metadata, A3.1 chunks and route views.
- `src/retrieval/parent_expansion.py`: post-retrieval bounded parent expansion.
- `scripts/build_a31_index.py`: one-pass vector build and three verified FAISS views.
- `scripts/benchmark_a31_controllers.py`: controlled B0/B1a/B1b/B2 public/dev harness.
- `tests/test_a31_semantic_multirepr.py`: 11 A3.1 unit tests.
- `src/config.py`, root/project `.env.example`, root `Makefile`: portable settings and commands.
- `data/index/a31_semantic_multirepr/`: shipped index artifacts and manifest.
- `data/evaluation/experiments/a31/`: raw answers, summaries, search traces, benchmark tables, and manual public review.

Validation: 104 repository tests pass in the CECS server venv.

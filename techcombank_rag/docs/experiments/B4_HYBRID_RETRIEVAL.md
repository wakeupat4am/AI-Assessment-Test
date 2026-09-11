# B4 — Sparse/dense hybrid retrieval over frozen A3.1

## Decision

**B4b Dense E5 + BM25 with weighted Reciprocal Rank Fusion is the recommended deployment retriever.** It improves retrieval and citation coverage across dev and holdout while adding only about 10–15 ms over B0 retrieval. B4d gives the strongest manually reviewed dev answer quality, but its CPU retrieval latency is roughly 11 seconds and it does not improve holdout answer accuracy over B4b.

The experiment was motivated by recent evidence that lexical retrieval remains unusually strong for financial text/table QA and that hybrid retrieval plus reranking is more reliable than dense-only or query expansion. Relevant primary sources include [From BM25 to Corrective RAG](https://arxiv.org/abs/2604.01733), [T²-RAGBench](https://aclanthology.org/2026.eacl-long.8/), [BGE-M3](https://arxiv.org/abs/2402.03216), and [FinCARDS](https://aclanthology.org/2026.findings-acl.1244/).

## Controlled design

The following are fixed across every arm:

- PaddleOCR-VL output and A3.1's 3,658 child chunks
- inherited headings, metric/value extraction, and parent expansion
- `intfloat/multilingual-e5-small` dense index
- answer model, answer prompt, evidence count, evaluation questions, and evaluator
- page metadata used by citation validation

Only retrieval/reranking changes:

| Arm | Changed component |
|---|---|
| B0 | Dense E5 one-shot |
| B4a | BM25-only |
| B4b | Dense top-20 + BM25 top-20, weighted RRF |
| B4c | B4b + deterministic entity/metric/year/number/granularity constraints |
| B4d | B4b + `BAAI/bge-reranker-v2-m3` cross-encoder over 20 candidates |

Public, dev, and holdout were run once on `cecs-server1`. Configuration was frozen before holdout; no threshold or weight was tuned after seeing holdout.

## Exact algorithm

### BM25 artifact

```text
A3.1 chunks.jsonl
  → Unicode NFKC + casefold
  → preserve Vietnamese words, acronyms, years, decimals and percentages
  → unigram + adjacent-bigram tokens
  → Okapi BM25 (k1=1.2, b=0.75)
  → compressed shipped artifact
```

The 1.3 MB `bm25.json.gz` artifact contains postings, document lengths, tokenizer configuration, source-chunk checksum, and build metadata. Loading fails if it does not match the shipped `chunks.jsonl`.

### B4b fusion

```text
query
  ├─ E5 dense top-20 ─┐
  └─ BM25 top-20 ─────┤
                      ↓
     score(d) = Σ source_weight / (60 + rank_source(d))
                      ↓
            return top-10 for evaluation
                      ↓
          top-5 + parent expansion for answer
```

Both source weights are `1.0`. RRF is used instead of directly adding BM25 and cosine scores because their scales are not comparable.

### B4c deterministic reranking

The RRF result is reranked with `0.65 × RRF + 0.35 × constraint score`. The constraint score checks requested years, acronyms, configured financial metric phrases, explicit numbers, and query-aware granularity. It makes no LLM call.

### B4d cross-encoder

The 20 hybrid candidates are jointly scored against the query by `BAAI/bge-reranker-v2-m3`, max length 512, batch size 8, using 8 CPU threads. The model is optional and is not required by the recommended B4b deployment path.

## Aggregate result across 40 questions

There are 36 answerable and four unanswerable questions across public, dev, and holdout.

| Arm | Manual answer accuracy | Hit@1 | Hit@5 | Hit@10 | Gold-page citation P/R | Approx. retrieval latency | GPT-5.6 projected total |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 65.0% | 44.4% | 69.4% | 83.3% | 47.2% | 45–61 ms | $0.586808 |
| B4a | 70.0% | 47.2% | 80.6% | 86.1% | 52.8% | 14–15 ms | $0.594588 |
| **B4b** | **72.5%** | **63.9%** | **83.3%** | **91.7%** | **63.9%** | **56–67 ms** | **$0.579348** |
| B4c | 67.5% | 63.9% | 83.3% | 91.7% | 58.3% | 62–70 ms | $0.629888 |
| B4d | **80.0%** | 47.2% | **86.1%** | **91.7%** | 50.0% | 9.8–11.6 s | $0.619608 |

Manual answer accuracy is separately recorded and auditable. The deterministic automatic heuristic undercounts concise correct answers and alternative phrasings; aggregate automatic accuracies are B0 30.0%, B4a 35.0%, B4b 35.0%, B4c 37.5%, and B4d 35.0%.

Actual monetary cost on the self-hosted Qwen server is USD 0. Projections reuse the exact observed tokens with the repository pricing snapshot and are not provider invoices.

## Split results

### Public — 10 questions

| Arm | Manual | Hit@1/5/10 | Citation P/R | Retrieval | E2E p50/p95 |
|---|---:|---:|---:|---:|---:|
| B0 | **1.00** | 0.556/1.000/1.000 | 0.667 | 59.7 ms | 6.187/7.449 s |
| B4a | 0.90 | 0.444/0.889/0.889 | 0.556 | 15.2 ms | 5.052/6.478 s |
| B4b | 0.90 | **0.667/1.000/1.000** | **0.778** | 67.5 ms | 5.197/9.603 s |
| B4c | 0.80 | 0.667/1.000/1.000 | 0.667 | 70.3 ms | 5.569/9.702 s |
| B4d | 0.90 | 0.333/0.889/0.889 | 0.444 | 9.808 s | 15.303/23.785 s |

B0 remains perfect on these ten visible questions. B4b retrieves all gold pages and improves early rank/citation overlap, but `sq-01` ends in citation-repair refusal even though the correct evidence was retrieved. This is an answer-layer regression, not a retrieval miss.

### Dev — 20 questions

| Arm | Manual | Auto | Hit@1/5/10 | Citation P/R | Retrieval |
|---|---:|---:|---:|---:|---:|
| B0 | 0.55 | 0.25 | 0.389/0.556/0.778 | 0.444 | 45.3 ms |
| B4a | 0.60 | 0.30 | 0.500/0.778/0.833 | 0.556 | 15.0 ms |
| B4b | 0.65 | 0.30 | **0.556/0.778/0.889** | **0.611** | 56.3 ms |
| B4c | 0.60 | **0.35** | 0.556/0.778/0.889 | 0.556 | 62.5 ms |
| B4d | **0.80** | 0.30 | 0.500/**0.833**/0.889 | 0.556 | 11.280 s |

BM25 moves `dev-09` from dense rank 10 to rank 1 and `dev-14`/`dev-15` from ranks 7/6 to rank 1. Hybrid recovers `dev-11` at rank 5 from the larger top-20 source pools. Cross-encoder correctly answers `dev-10`, `dev-11`, `dev-13`, and `dev-17`, explaining its manual-quality gain.

### Untuned holdout — 10 questions

| Arm | Manual | Auto | Hit@1/5/10 | Citation P/R | Retrieval |
|---|---:|---:|---:|---:|---:|
| B0 | 0.50 | 0.30 | 0.444/0.667/0.778 | 0.333 | 61.2 ms |
| B4a | 0.70 | 0.40 | 0.444/0.778/0.889 | 0.444 | 14.4 ms |
| **B4b** | **0.70** | **0.40** | **0.778/0.778/0.889** | **0.556** | **60.8 ms** |
| B4c | 0.70 | 0.40 | 0.778/0.778/0.889 | 0.556 | 68.1 ms |
| B4d | 0.70 | 0.40 | 0.556/**0.889/1.000** | 0.444 | 11.586 s |

B4b improves holdout manual accuracy from 50% to 70%, Hit@1 from 44.4% to 77.8%, Hit@10 from 77.8% to 88.9%, and citation overlap from 33.3% to 55.6% without material end-to-end latency growth.

## Failure analysis

- **Incomplete gold-page sets:** `dev-01` is correctly answered from pages 209/238/244 although the gold page is 20; `dev-10` is correctly answered from page 148 while gold lists 151. Gold-page metrics therefore understate evidence quality when the fact is repeated elsewhere.
- **Citation repair after successful retrieval:** public `sq-01` and calculation questions such as `dev-15` can retrieve the required evidence but end in refusal because a derived number is not verbatim in one chunk.
- **Flattened financial-table ambiguity:** B4c public `sq-03` promotes a semantically related but wrong total-assets row from page 379.
- **Sparse-only blind spot:** B4a loses public `sq-03`; exact words alone do not resolve the correct accounting row. Dense retrieval is required as a safety branch.
- **Fusion dilution:** holdout `holdout-04` is answered by BM25-only but refused by fused/reranked variants, showing that fixed equal-weight fusion is not uniformly optimal.
- **Generic cross-encoder trade-off:** B4d improves deeper recall and difficult dev answers, but regresses public early rank and adds about 11 seconds on CPU. It is not on the deployment Pareto frontier.

## Submission integration

Recommended default:

```text
PaddleOCR-VL → A3.1 child/parent representations
             → Dense E5 top-20 + BM25 top-20
             → weighted RRF (B4b)
             → parent expansion
             → grounded answer + citation validation
```

B2 dedup/diversity/adaptive search and B3 HyDE remain independently switchable research components. They should not be claimed as a combined improvement until a new controlled composition ablation is run. A defensible next composition is B4b as the first search and selective B3b only as B2's second search when required evidence constraints remain unmet.

## Reproduction

```bash
make build-bm25
make verify-b4-index
make ablate-b4-retrieval
make ablate-b4
make ablate-b4-dev
make ablate-b4-holdout
python techcombank_rag/scripts/compare_b4_answers.py
python techcombank_rag/scripts/analyze_b4_evidence.py
```

Enable the recommended retriever without changing provider/model configuration:

```dotenv
INDEX_DIR=techcombank_rag/data/index/a31_semantic_multirepr/all
B4_RETRIEVAL_MODE=hybrid
B4_BM25_ARTIFACT=techcombank_rag/data/index/a31_semantic_multirepr/all/bm25.json.gz
```

## Files and evidence

- `src/retrieval/bm25.py`: tokenizer, artifact builder, Okapi BM25
- `src/retrieval/hybrid.py`: weighted RRF
- `src/retrieval/finance_reranker.py`: deterministic finance constraints
- `src/retrieval/cross_encoder_reranker.py`: optional multilingual B4d
- `src/retrieval/b4_factory.py`: environment-driven composition
- `config/b4_retrieval.json`: all experiment rules/weights
- `scripts/ablate_b4.py`: retrieval-only and end-to-end harness
- `data/evaluation/experiments/b4/`: raw rows, summaries, checksum-bearing reports
- `docs/experiments/B4_PUBLIC_ANSWERS.md`: full answers for all ten public questions
- `docs/experiments/B4_EVIDENCE_ANALYSIS.md`: per-split metrics and per-question rank evidence

# Frozen baseline A0/B0 — 09/09/2026

## Gate status

**PASS.** The 10-question public batch completed with zero runtime errors on
`cecs_server_1` using the existing OpenAI-compatible Qwen endpoint. The raw
result, summary, human labels, question set, and shipped index are all retained
for audit. No Track A or Track B improvement is included in this run.

## Frozen configuration

| Component | A0/B0 value |
|---|---|
| Parser | PyMuPDF text extraction with geometric spread splitting |
| Corpus | 197 PDF sheets → 393 printed-page fragments → 446 chunks |
| Chunking | Page-bounded, 3,200 characters, 480-character overlap |
| Embedding | `intfloat/multilingual-e5-small`, 384 dimensions, normalized |
| Index | FAISS `IndexFlatIP` |
| Retrieval | Dense-only; top 5 sent as evidence, top 10 logged for evaluation |
| Generator | `Qwen/Qwen3.5-9B` on server, OpenAI-compatible endpoint |
| Refusal threshold | Top-1 cosine score `< 0.50` |
| Citation guard | Printed-page allow-list plus verbatim numeric support |

The index artifact contains SHA-256 values for the source PDF, chunks, and
FAISS file. The result summary also records the question-set SHA-256.

## Results

| Metric | Result |
|---|---:|
| Retrieval Recall@1 | 11.11% |
| Retrieval Recall@5 | 44.44% |
| Retrieval Recall@10 | 55.56% |
| Manual answer accuracy | 60.00% |
| Citation precision | 33.33% |
| Citation recall | 33.33% |
| Citation-validator pass rate | 80.00% |
| Refusal accuracy | 70.00% |
| Mean latency | 7.298 s |
| p50 latency | 4.640 s |
| p95 latency | 18.946 s |
| LLM calls | 13: 10 answer + 3 repair |
| Tokens | 51,078 input + 542 output = 51,620 total |

The token source is the Qwen tokenizer loaded from the server model path. The
endpoint's original prompt usage field was invalid (it counted the batch shape),
so the client detects implausible values and records `client_tokenizer` rather
than silently reporting zero.

## Same-token API cost projection

| Model | Estimated cost for this 10-question run |
|---|---:|
| GPT-5.6 Luna | $0.010866 |
| GPT-5.6 Terra | $0.108660 |
| GPT-5.6 Sol | $0.215152 |
| Claude Sonnet 5 | $0.161364 |
| Claude Opus 5 | $0.268940 |
| Claude Fable 5 | $0.537880 |

Qwen's API cost is recorded as $0 because it is self-hosted; that does not mean
GPU infrastructure is free. The projections hold the observed token mix fixed,
so they compare list-price exposure only, not likely quality, latency, or token
count changes under another model. Prices are snapshotted on 09/09/2026 and must
be rechecked or overridden before final submission.

## Failure taxonomy

| Primary failure | Count |
|---|---:|
| Retrieval miss | 3 |
| False refusal | 1 |
| Citation error | 1 |
| No primary failure | 5 |

The baseline establishes the main bottleneck clearly: Recall@10 is only 55.56%,
so a stronger answer model alone cannot recover four of nine answerable gold
pages. There is also one case where the correct page was retrieved at rank 3 but
generation/repair still refused, and one correct definition cited the table of
contents instead of the canonical glossary page. These cases justify separate
Document Intelligence and Retrieval experiments instead of changing several
components at once.

## Reproduce and inspect

```bash
make verify-index
make baseline
```

Frozen artifacts:

- `data/evaluation/baseline/baseline-a0-b0.jsonl`
- `data/evaluation/baseline/baseline-a0-b0.summary.json`
- `data/evaluation/baseline/manual_review.public.json`

The deployed copy is `/home/ubuntu/TCB_Test` on `cecs_server_1`. The submitted
repository remains provider-neutral; that server path is recorded here only as
the provenance of this measurement and is not a runtime dependency.

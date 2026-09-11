# B3 — HyDE retrieval over A3.1

## Outcome

HyDE was implemented and benchmarked on the server against the frozen A3.1+B0 baseline. The deployable default remains **B0 one-shot**. The only promising HyDE variant is **B3b query + HyDE fusion**, but it should be used selectively after a low-coverage signal rather than on every query.

The implementation follows the core HyDE idea: generate one hypothetical answer-like document, encode it with the same passage encoder, and use that vector to retrieve real indexed evidence. Generated text is never sent to the answer model and can never be cited.

References: [HyDE ACL paper](https://aclanthology.org/2023.acl-long.99/), [official implementation](https://github.com/texttron/hyde).

## Fixed experimental controls

- Index: `data/index/a31_semantic_multirepr/all`
- Ingestion/chunks: frozen A3.1 child chunks plus parent expansion
- Embedding model: the same E5 model for every arm
- Answer model, prompt, `top_k`, gold questions, and evaluator: unchanged
- X-Router and B2: disabled, so this is a pure A3.1 retrieval ablation
- All four arms were run from start to finish on `cecs-server1`

## Arms

| Arm | Retrieval query |
|---|---|
| B0 | Original user query only |
| B3a | Hypothetical-document vector only |
| B3b | Weighted RRF fusion of original-query and hypothetical-document rankings |
| B3c | Same fusion, but invented numbers in the hypothetical document are replaced with `[VALUE]` |

For fusion, each candidate receives the sum of weighted reciprocal-rank contributions from the original and HyDE rankings. Real A3.1 chunk metadata and the original dense similarity score are preserved.

## Runtime flow

```text
query
  ├─> original query embedding ─> A3.1 search ─┐
  └─> one HyDE LLM call ─> passage embedding ─> A3.1 search
                                                │
                                  weighted RRF fusion
                                                │
                              real chunks + parent expansion
                                                │
                                  unchanged answer generation
```

Failure behavior is safe: an empty/failed HyDE generation falls back to original-query retrieval. Hypothetical text, mode, latency, candidate counts, pages, cache state, and errors are recorded in a JSONL trace.

## Configuration

```dotenv
ENABLE_HYDE=true
HYDE_MODE=fusion
HYDE_PROMPT_VARIANT=paper_faithful
HYDE_MODEL=<provider model>
HYDE_CANDIDATE_K=20
HYDE_RRF_K=60
HYDE_ORIGINAL_WEIGHT=1.0
HYDE_HYPOTHETICAL_WEIGHT=1.0
HYDE_CACHE_SIZE=256
```

`ENABLE_HYDE=true` is intentionally rejected together with X-Router in the normal chatbot constructor. Their composition has not been calibrated, so silently combining them would invalidate this ablation.

## Public benchmark — 10 questions

| Arm | Auto acc. | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Retrieval mean | E2E p50/p95 | Calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.40 | 0.5556 | 1.0000 | 1.0000 | 0.6667/0.6667 | 1.00 | 0.057 s | 5.007/6.536 s | 10 | 36,077 |
| B3a | 0.40 | 0.4444 | 0.7778 | 0.8889 | 0.5556/0.5556 | 1.00 | 9.721 s | 15.086/17.659 s | 20 | 33,399 |
| B3b | 0.40 | **0.6667** | **1.0000** | **1.0000** | 0.6667/0.6667 | 0.90 | 8.892 s | 13.453/19.033 s | 21 | 41,681 |
| B3c | 0.40 | 0.5556 | 0.6667 | 1.0000 | 0.5556/0.5556 | 1.00 | 7.935 s | 12.562/17.474 s | 20 | 40,085 |

Manual semantic review of the answer content is B0 10/10, B3a 10/10, B3b 9/10, and B3c 9/10. The automatic score is deliberately strict and remains 0.40 because several valid concise answers do not cover every phrase in the verbose gold answer.

Public regressions:

- B3b falsely refuses question 1 after citation validation/repair.
- B3c answers question 3 with the wrong total-assets value.

## Dev benchmark — 20 questions

| Arm | Auto acc. | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Retrieval mean | E2E p50/p95 | Calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.25 | 0.3889 | 0.5556 | 0.7778 | 0.4444/0.4444 | 0.75 | 0.051 s | 4.765/8.180 s | 21 | 67,358 |
| B3a | 0.15 | 0.2778 | 0.5000 | 0.7778 | 0.3333/0.3333 | 0.60 | 8.823 s | 13.127/17.175 s | 41 | 69,745 |
| B3b | 0.25 | 0.3889 | **0.6111** | **0.8333** | **0.5556/0.5556** | **0.85** | 8.657 s | 13.071/15.010 s | 40 | 66,158 |
| B3c | **0.30** | 0.3889 | 0.6111 | 0.6667 | 0.5556/0.5556 | 0.80 | 7.288 s | 12.092/17.342 s | 41 | 73,906 |

B3b gains 5.55 percentage points at both Hit@5 and Hit@10 and improves citation precision/recall by 11.11 points. It notably moves the gold page for `dev-09` from rank 10 to rank 2. However, median end-to-end latency rises from 4.765 s to 13.071 s.

## Cost projection

Actual Qwen endpoint cost is recorded as USD 0 because the server is self-hosted. The same-token projections below use the repository pricing configuration; they are estimates, not provider invoices.

| Split / arm | GPT-5.6 | Claude Sonnet 5 |
|---|---:|---:|
| Public B0 | $0.150260 | $0.112695 |
| Public B3b | $0.188964 | $0.141723 |
| Dev B0 | $0.281704 | $0.211278 |
| Dev B3b | $0.306632 | $0.229974 |

Public B3b uses about 15.5% more total tokens than B0. Dev B3b happens to use 1.8% fewer total tokens because its final evidence/answers are shorter, but it still doubles the number of LLM calls and adds about 8.6 seconds of retrieval latency per question.

## Observed failure modes

- Paper-faithful hypothetical documents invent plausible figures. This is expected from HyDE but unsafe if generated text leaks into grounding; the implementation prevents that leak.
- HyDE-only loses exact financial matches because an invented passage can move the embedding away from the true table row.
- Numeric redaction in B3c removes useful semantic structure as well as risk; it helps some questions but loses gold pages on others.
- Fusion can improve retrieval coverage without improving the strict answer metric when answer synthesis, validation, or citation repair remains the bottleneck.
- HyDE adds a full generation call to every query, making unconditional use expensive and slow.

## Decision

1. Keep A3.1+B0 as the deployment default.
2. Reject B3a HyDE-only.
3. Do not promote B3c despite its higher dev automatic score; its Hit@10 and public stability regress.
4. Retain B3b behind a feature flag and trigger it only when one-shot retrieval lacks required entities/years or has poor evidence coverage.
5. Next iteration: integrate selective B3b as B2's second-search strategy, then tune the trigger and fusion weight on dev and evaluate exactly once on holdout.

## Reproduction

```bash
make ablate-hyde
make ablate-hyde-dev
python scripts/compare_hyde_answers.py
```

Raw rows, summaries, traces, and Markdown tables are in `data/evaluation/experiments/hyde/`.

# B1b — Qwen LLM-Agent X-Router

## Outcome

B1b is implemented and reproducible, but it is **not selected over B1a** in
its current form. On public it ties B1a on retrieval and automatic answer
metrics while adding latency and tokens. On dev it is worse on Hit@5/10 and
citation metrics. This negative result is retained as evidence rather than
hidden by further prompt calibration.

## Controlled change

Only the route decision changes:

```text
B1a: query -> deterministic rules -> route -> frozen retrieval -> frozen answer
B1b: query -> Qwen3.5-9B agent -> route -> frozen retrieval -> frozen answer
```

Frozen across both arms:

- A2 narrative and structured indexes;
- A3 multi-representation index;
- dense encoder, `top_k`, and RRF;
- answer prompt, answer model, citation validation, and refusal policy.

## Runtime architecture

- `127.0.0.1:8000/v1`: Qwen answer endpoint.
- `127.0.0.1:8001/v1`: dedicated Qwen router endpoint.
- Port 8001 is a lightweight local proxy to port 8000, so traffic and config
  are separated without loading a second 9B weight copy into VRAM. Requests
  remain serialized by the current Qwen backend; this is a logical endpoint
  split, not independent GPU inference.

The B1b router uses one temperature-zero call. The LLM emits only `route`,
`confidence`, and one short reason. Deterministic feature extraction then
attaches the eight B1 `query_features`, preserving the original output
contract while avoiding long JSON generations. A reason string is normalized
to a one-item array locally. Invalid JSON, timeout, unsupported route, or
confidence below 0.62 falls back to `multi_repr`.

## Implementation

- `src/routing/llm_xrouter.py`: isolated LLM router, validation, normalization,
  logging, usage accounting, and fallback.
- `config/xrouter_b1b.json`: index descriptions, route policy, prompt, examples,
  threshold, and route-to-index mapping.
- `src/retrieval/routed_retriever.py`: accepts B1a or B1b behind the same API.
- `scripts/serve_router_proxy.py`: dedicated router port.
- `scripts/evaluate_router_b1b.py`: route decision benchmark.
- `scripts/ablate_xrouter_b1b.py`: frozen end-to-end B1a/B1b ablation.
- `tests/test_llm_xrouter.py`: contract, context, fallback, malformed output,
  timeout, logging, and 20 constrained route cases.

All tests passed locally and on the server: **81 passed** after the final
normalizer update (80 passed before the additional contract test update was
synced).

## Route-decision diagnostic

Dataset: 25 Vietnamese route labels. This set was originally used to establish
B1a rules, so B1a's score is expected to be optimistic and is not an unbiased
generalization estimate.

| Metric | B1a rules | B1b Qwen | Delta |
|---|---:|---:|---:|
| Route accuracy | 100% | 72% | -28 pp |
| Structured accuracy | 100% | 66.7% | -33.3 pp |
| Narrative accuracy | 100% | 100% | 0 pp |
| Multi-hop accuracy | 100% | 75% | -25 pp |
| Multi-repr accuracy | 100% | 40% | -60 pp |
| Router latency p50 | 0.0002 s | 3.6075 s | +3.6073 s |
| Router latency p95 | 0.0003 s | 4.9319 s | +4.9316 s |

B1b used 7,686 input and 688 output tokens across 25 calls: 334.96 total
tokens/query. API cost is USD 0 because this run uses local Qwen inference;
hardware time is represented by latency.

The seven route disagreements are concentrated in abbreviations/ratings,
broad questions, mixed fact-plus-explanation intent, and a contextual follow-up.

## End-to-end results

### Public — 10 questions

| Metric | B1a | B1b | Delta B1b-B1a |
|---|---:|---:|---:|
| Retrieval Hit@1 | 0.3333 | 0.3333 | 0 |
| Retrieval Hit@5 | 0.4444 | 0.4444 | 0 |
| Retrieval Hit@10 | 0.6667 | 0.6667 | 0 |
| Automatic answer accuracy | 0.3000 | 0.3000 | 0 |
| Citation precision / recall | 0.3333 / 0.3333 | 0.3333 / 0.3333 | 0 / 0 |
| End-to-end latency p50 | 5.2225 s | 7.7190 s | +2.4965 s |
| End-to-end latency p95 | 9.9487 s | 11.3522 s | +1.4035 s |
| Total tokens | 34,035 | 37,488 | +3,453 |

B1b adds one route call per question: 345.3 tokens/query and approximately
2.91 seconds median model-call latency. Only one public route differs and it
does not change Hit@k or answer correctness.

### Dev — 20 questions

| Metric | B1a | B1b | Delta B1b-B1a |
|---|---:|---:|---:|
| Retrieval Hit@1 | 0.1667 | 0.1667 | 0 |
| Retrieval Hit@5 | 0.7222 | 0.6111 | -0.1111 |
| Retrieval Hit@10 | 0.7222 | 0.6667 | -0.0556 |
| Automatic answer accuracy | 0.1500 | 0.1500 | 0 |
| Citation precision / recall | 0.2778 / 0.2778 | 0.2222 / 0.2222 | -0.0556 / -0.0556 |
| End-to-end latency p50 | 4.7715 s | 7.3995 s | +2.6280 s |
| End-to-end latency p95 | 8.1277 s | 10.5875 s | +2.4598 s |
| Total tokens | 60,407 | 68,469 | +8,062 |

The two lost dev Hit@5 cases are:

- `dev-09`: B1a `multi_repr` -> B1b `multi_hop`;
- `dev-10`: B1a `multi_repr` -> B1b `structured`.

The B1b route calls themselves consumed 6,330 input and 532 output tokens on
dev: 343.1 total tokens/query with approximately 3.13 seconds median call
latency. The larger end-to-end token delta also reflects different evidence
selected by different routes.

## Controlled-A2 ablation

This follow-up removes A3 entirely. Only the frozen A2 structured index and
the 837-chunk narrative projection of A2 are instantiated:

```text
narrative  -> A2 narrative projection
structured -> A2 structured
multi_repr -> A2 structured + A2 narrative (RRF)
multi_hop  -> A2 structured + A2 narrative (RRF)
```

`multi_repr` and `multi_hop` intentionally share the same retrieval operator
here. True iterative multi-hop retrieval is deferred, so this experiment tests
the router over A2 views without introducing another Track B change.

### Controlled-A2 public — 10 questions

| Metric | B1a | B1b | Delta B1b-B1a |
|---|---:|---:|---:|
| Retrieval Hit@1 | 0.3333 | 0.3333 | 0 |
| Retrieval Hit@5 | 0.4444 | 0.4444 | 0 |
| Retrieval Hit@10 | 0.6667 | 0.6667 | 0 |
| Automatic answer accuracy | 0.3000 | 0.3000 | 0 |
| Citation precision / recall | 0.3333 / 0.3333 | 0.3333 / 0.3333 | 0 / 0 |
| End-to-end latency p50 | 4.1195 s | 7.8630 s | +3.7435 s |
| End-to-end latency p95 | 8.9179 s | 12.8580 s | +3.9402 s |
| Total tokens | 34,035 | 37,488 | +3,453 |

### Controlled-A2 dev — 20 questions

| Metric | B1a | B1b | Delta B1b-B1a |
|---|---:|---:|---:|
| Retrieval Hit@1 | 0.1667 | 0.1667 | 0 |
| Retrieval Hit@5 | 0.6111 | 0.6111 | 0 |
| Retrieval Hit@10 | 0.6667 | 0.6667 | 0 |
| Automatic answer accuracy | 0.1500 | 0.1500 | 0 |
| Citation precision / recall | 0.2222 / 0.2222 | 0.2222 / 0.2222 | 0 / 0 |
| End-to-end latency p50 | 3.6880 s | 7.3480 s | +3.6600 s |
| End-to-end latency p95 | 7.3418 s | 10.4381 s | +3.0963 s |
| Total tokens | 61,552 | 68,176 | +6,624 |

Quality is identical between B1a and B1b in Controlled-A2. B1b therefore
provides no measurable retrieval, answer, or citation benefit while adding one
LLM call and roughly 3–4 seconds p50 latency.

Comparing B1a across experiments also isolates the value of A3 on dev:

- B1a best-system with A3: Hit@5 0.7222, Hit@10 0.7222.
- B1a Controlled-A2: Hit@5 0.6111, Hit@10 0.6667.

Therefore A3 helped the rule router on the dev set. B1b did not capture that
benefit because it routed the affected `dev-09` and `dev-10` questions away
from A3 in the best-system experiment.

## Decision

Keep **B1a as the current default**. B1b demonstrates that adding a reasoning
model does not automatically improve a four-way routing problem: the 9B call
adds substantial latency and loses retrieval recall on dev without improving
answer accuracy.

The next defensible B1b iteration is not a larger model. It is a hybrid policy:
retain deterministic high-confidence structured/multi-hop rules and call a
small LLM only for the ambiguous `multi_repr` boundary. That directly targets
the observed errors while reducing route-call frequency and cost.

## Reproduction

```bash
# Terminal 1: existing Qwen answer endpoint on :8000

# Terminal 2: logical router endpoint
make router-proxy

# .env for B1b
XROUTER_ENABLED=true
XROUTER_TYPE=llm
XROUTER_CONFIG=techcombank_rag/config/xrouter_b1b.json
ROUTER_LLM_BASE_URL=http://127.0.0.1:8001/v1

make evaluate-router-b1b
make ablate-b1b
make ablate-b1b-dev
make ablate-b1b-controlled-a2
make ablate-b1b-controlled-a2-dev
```

Frozen artifacts are under `data/evaluation/experiments/b1b/`.

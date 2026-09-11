# B5 — bounded tool-calling agent over A3.1 + B4b

## Scope

B5 keeps the production candidate stack fixed: **A3.1 semantic child/parent chunks** and **B4b E5 dense + BM25 weighted-RRF**. It does not change the answer prompt or model. For a comparison/calculation-shaped query only, an LLM planner may select one host-executed action after the initial retrieval:

1. `hybrid_retrieve` with one short, distinct missing-fact query; or
2. `calculator` (`subtract`, `percentage_change`, or `cagr`), only after both literal operands and their printed pages are found in retrieved evidence.

The agent cannot browse, write files, issue shell commands, fabricate operands, or loop. Easy questions skip planning. The protocol is provider-neutral strict JSON rather than a provider-specific function-call API; a compatibility parser also accepts Qwen's equivalent nested JSON form.

## Controls

`config/b5_tool_agent.json` fixes `initial_top_k=20`, `followup_top_k=12`, and `max_tool_actions_after_initial_retrieval=1`. Every trace records the query, planner output, selected/validated action, evidence pages, and latency. A derived calculation is explicit evidence with source chunk IDs and pages, so the existing answer/citation validation remains in force.

## Frozen B4b vs B5 results

All results below used the same A3.1 index, B4b retrieval, Qwen answer model, answer prompt, question split, and top-k. “Automatic accuracy” is the existing conservative evaluator, not a semantic human judgement.

| Split | Arm | Auto acc. | Hit@5 | Citation P/R | Retrieval mean | p50 / p95 E2E | LLM calls | Tokens | Planner rate | Useful tool actions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dev (20) | B4b | 0.300 | 0.778 | 0.611 / 0.611 | 62 ms | 5.590 / 7.692 s | 21 | 64,131 | 0% | 0 |
| Dev (20) | B5 | 0.300 | 0.778 | 0.611 / 0.611 | 658 ms | 5.367 / 11.031 s | 26 | 70,066 | 25% | 2 second retrievals |
| Public (10) | B4b | 0.400 | 1.000 | 0.778 / 0.778 | 78 ms | 4.925 / 9.745 s | 11 | 38,763 | 0% | 0 |
| Public (10) | B5 | 0.400 | 1.000 | 0.778 / 0.778 | 171 ms | 4.923 / 7.900 s | 12 | 40,045 | 10% | 0 |
| Holdout (10) | B4b | 0.400 | 0.778 | 0.556 / 0.556 | 72 ms | 6.668 / 9.622 s | 11 | 34,983 | 0% | 0 |
| Holdout (10) | B5 | 0.400 | 0.778 | 0.611 / 0.667 | 1,772 ms | 5.874 / 13.613 s | 14 | 37,576 | 40% | 3 second retrievals + 1 calculator |

The calculator correctly resolved holdout-07 from evidence on printed page 257: `8.645 - 6.075 = 2.570 tỷ đồng`. It turned a citation-validation refusal into a grounded cited answer. However, this single win did not improve conservative automatic accuracy over the split, and second retrieval sometimes changed a correct/adequate answer into a worse one (notably holdout-02).

## Verdict

**Do not make B5 the default deployment path yet.** B4b remains the recommended default: it is materially faster and achieves the same aggregate automatic accuracy. Retain B5 as an opt-in, audited path for explicit arithmetic/multi-fact queries, or route only high-precision calculation patterns to it. A next iteration should improve the planner's evidence-sufficiency decision and restrict second retrieval to a verified missing slot, while preserving the calculator guardrails.

## Reproduce

```bash
python scripts/ablate_b5.py data/evaluation/public.json --run-label public-final
python scripts/ablate_b5.py data/evaluation/dev.json --run-label dev-final
python scripts/ablate_b5.py data/evaluation/holdout.json --run-label holdout-final
```

Set `B5_AGENT_ENABLED=true`, `B4_RETRIEVAL_MODE=hybrid`, and leave X-Router/HyDE disabled for the isolated B5 runtime. The code rejects implicit B5 combinations with those alternate query transformations.

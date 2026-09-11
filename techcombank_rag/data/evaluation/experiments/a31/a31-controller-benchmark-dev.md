# A3.1 controller benchmark — dev

| Controller | Auto accuracy | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Avg rounds | Trigger | Evidence tokens | Parent-expanded | Duplicate slots | Retrieval ms | E2E p50/p95 s | LLM calls/tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0_one_shot | 0.2500 | 0.3889 | 0.5556 | 0.7778 | 0.4444/0.4444 | 0.7500 | 1.000 | 0.000 | 2220.7 | 1.50 | 3.65 | 50.1 | 4.877/8.054 | 21/67358 |
| B1a_rule | 0.2000 | 0.3889 | 0.5000 | 0.6111 | 0.3333/0.3333 | 0.6000 | 1.000 | 0.000 | 2623.4 | 2.85 | 3.80 | 55.0 | 3.925/11.386 | 22/80374 |
| B1b_llm | 0.2500 | 0.4444 | 0.5556 | 0.6667 | 0.3889/0.3889 | 0.6500 | 1.000 | 0.000 | 2579.7 | 2.55 | 3.75 | 2724.4 | 7.033/8.822 | 41/83031 |
| B2_adaptive | 0.1500 | 0.0000 | 0.5000 | 0.6111 | 0.3611/0.3889 | 0.6000 | 1.150 | 0.150 | 2668.7 | 2.85 | 0.40 | 84.3 | 4.442/12.660 | 18/72727 |

> A3.1 and answer generation are fixed across arms; only the retrieval controller changes.
> Automatic answer accuracy remains diagnostic until manual review is added.

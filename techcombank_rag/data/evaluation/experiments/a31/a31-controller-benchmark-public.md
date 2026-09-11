# A3.1 controller benchmark — public

| Controller | Auto accuracy | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Avg rounds | Trigger | Evidence tokens | Parent-expanded | Duplicate slots | Retrieval ms | E2E p50/p95 s | LLM calls/tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0_one_shot | 0.4000 | 0.5556 | 1.0000 | 1.0000 | 0.6667/0.6667 | 1.0000 | 1.000 | 0.000 | 2399.0 | 3.00 | 4.50 | 62.0 | 5.282/7.420 | 10/36077 |
| B1a_rule | 0.4000 | 0.5556 | 0.7778 | 0.8889 | 0.6667/0.6667 | 0.9000 | 1.000 | 0.000 | 2545.9 | 4.10 | 4.40 | 71.7 | 5.340/8.343 | 11/42271 |
| B1b_llm | 0.3000 | 0.5556 | 0.7778 | 0.7778 | 0.6667/0.6667 | 0.8000 | 1.000 | 0.000 | 2516.3 | 4.40 | 4.60 | 3059.9 | 8.394/10.869 | 21/45981 |
| B2_adaptive | 0.3000 | 0.2222 | 0.6667 | 0.8889 | 0.5000/0.5556 | 0.9000 | 1.200 | 0.200 | 2783.8 | 3.60 | 0.30 | 95.3 | 5.089/7.483 | 8/33344 |

> A3.1 and answer generation are fixed across arms; only the retrieval controller changes.
> Automatic answer accuracy remains diagnostic until manual review is added.

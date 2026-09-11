# A3.1 HyDE ablation — dev

| Arm | Auto accuracy | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Retrieval ms | E2E p50/p95 s | Calls | Tokens | HyDE fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0_one_shot | 0.2500 | 0.3889 | 0.5556 | 0.7778 | 0.4444/0.4444 | 0.7500 | 50.8 | 4.765/8.180 | 21 | 67358 | 0.000 |
| B3a_hyde_only | 0.1500 | 0.2778 | 0.5000 | 0.7778 | 0.3333/0.3333 | 0.6000 | 8823.2 | 13.127/17.175 | 41 | 69745 | 0.000 |
| B3b_fusion | 0.2500 | 0.3889 | 0.6111 | 0.8333 | 0.5556/0.5556 | 0.8500 | 8657.1 | 13.071/15.010 | 40 | 66158 | 0.000 |
| B3c_safe_fusion | 0.3000 | 0.3889 | 0.6111 | 0.6667 | 0.5556/0.5556 | 0.8000 | 7287.7 | 12.092/17.342 | 41 | 73906 | 0.000 |

> A3.1, E5, answer prompt, answer model, and top-k are fixed. Only query representation/retrieval changes.
> Hypothetical documents are search pivots only and never enter answer evidence.

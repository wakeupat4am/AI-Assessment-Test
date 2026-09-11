# A3.1 HyDE ablation — public

| Arm | Auto accuracy | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Retrieval ms | E2E p50/p95 s | Calls | Tokens | HyDE fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0_one_shot | 0.4000 | 0.5556 | 1.0000 | 1.0000 | 0.6667/0.6667 | 1.0000 | 56.7 | 5.007/6.536 | 10 | 36077 | 0.000 |
| B3a_hyde_only | 0.4000 | 0.4444 | 0.7778 | 0.8889 | 0.5556/0.5556 | 1.0000 | 9720.7 | 15.086/17.659 | 20 | 33399 | 0.000 |
| B3b_fusion | 0.4000 | 0.6667 | 1.0000 | 1.0000 | 0.6667/0.6667 | 0.9000 | 8891.8 | 13.453/19.033 | 21 | 41681 | 0.000 |
| B3c_safe_fusion | 0.4000 | 0.5556 | 0.6667 | 1.0000 | 0.5556/0.5556 | 1.0000 | 7935.0 | 12.562/17.474 | 20 | 40085 | 0.000 |

> A3.1, E5, answer prompt, answer model, and top-k are fixed. Only query representation/retrieval changes.
> Hypothetical documents are search pivots only and never enter answer evidence.

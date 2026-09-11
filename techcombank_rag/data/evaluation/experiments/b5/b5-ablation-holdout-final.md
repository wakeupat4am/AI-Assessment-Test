# B5 tool-calling ablation — holdout-final

| Arm | Auto acc. | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Retrieval ms | E2E p50/p95 s | Calls | Tokens | Agent trigger | Calculator ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B4b_hybrid | 0.4000 | 0.7778 | 0.7778 | 0.8889 | 0.5556/0.5556 | 0.8000 | 72.0 | 6.668/9.622 | 11 | 34983 | 0.000 | 0 |
| B5_tool_agent | 0.4000 | 0.4444 | 0.7778 | 0.8889 | 0.6111/0.6667 | 0.9000 | 1772.4 | 5.874/13.613 | 14 | 37576 | 0.400 | 1 |

> A3.1, B4b hybrid retrieval, answer model/prompt, questions, and top-k are fixed. B5 changes only bounded tool orchestration.

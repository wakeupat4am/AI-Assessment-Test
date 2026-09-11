# B5 tool-calling ablation — public-final

| Arm | Auto acc. | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Retrieval ms | E2E p50/p95 s | Calls | Tokens | Agent trigger | Calculator ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B4b_hybrid | 0.4000 | 0.6667 | 1.0000 | 1.0000 | 0.7778/0.7778 | 0.9000 | 77.6 | 4.925/9.745 | 11 | 38763 | 0.000 | 0 |
| B5_tool_agent | 0.4000 | 0.6667 | 1.0000 | 1.0000 | 0.7778/0.7778 | 0.9000 | 170.5 | 4.923/7.900 | 12 | 40045 | 0.100 | 0 |

> A3.1, B4b hybrid retrieval, answer model/prompt, questions, and top-k are fixed. B5 changes only bounded tool orchestration.

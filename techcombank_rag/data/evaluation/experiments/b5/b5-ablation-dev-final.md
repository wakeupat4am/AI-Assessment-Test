# B5 tool-calling ablation — dev-final

| Arm | Auto acc. | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Retrieval ms | E2E p50/p95 s | Calls | Tokens | Agent trigger | Calculator ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B4b_hybrid | 0.3000 | 0.5556 | 0.7778 | 0.8889 | 0.6111/0.6111 | 0.8000 | 62.2 | 5.590/7.692 | 21 | 64131 | 0.000 | 0 |
| B5_tool_agent | 0.3000 | 0.5556 | 0.7778 | 0.8889 | 0.6111/0.6111 | 0.8000 | 658.4 | 5.367/11.031 | 26 | 70066 | 0.250 | 0 |

> A3.1, B4b hybrid retrieval, answer model/prompt, questions, and top-k are fixed. B5 changes only bounded tool orchestration.

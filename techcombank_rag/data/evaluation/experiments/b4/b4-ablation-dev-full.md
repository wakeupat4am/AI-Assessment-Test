# B4 end-to-end ablation — dev

| Arm | Auto acc. | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Retrieval ms | E2E p50/p95 s | Calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0_one_shot | 0.2500 | 0.3889 | 0.5556 | 0.7778 | 0.4444/0.4444 | 0.7500 | 45.3 | 5.011/9.145 | 21 | 67358 |
| B4a_bm25 | 0.3000 | 0.5000 | 0.7778 | 0.8333 | 0.5556/0.5556 | 0.7500 | 15.0 | 4.698/12.152 | 22 | 68552 |
| B4b_hybrid_rrf | 0.3000 | 0.5556 | 0.7778 | 0.8889 | 0.6111/0.6111 | 0.8000 | 56.3 | 5.404/6.974 | 21 | 64131 |
| B4c_finance_rerank | 0.3500 | 0.5556 | 0.7778 | 0.8889 | 0.5556/0.5556 | 0.8000 | 62.5 | 5.662/9.208 | 22 | 70917 |
| B4d_cross_encoder | 0.3000 | 0.5000 | 0.8333 | 0.8889 | 0.5556/0.5556 | 0.9000 | 11280.1 | 16.622/20.153 | 21 | 68960 |

> A3.1 chunks, E5 dense index, answer model/prompt, questions, and evaluation k are fixed.
> Only the first-stage retrieval/reranking method changes.

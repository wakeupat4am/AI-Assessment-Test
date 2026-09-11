# B4 evidence analysis

## End-to-end metrics

| Split | Arm | Auto acc. | Manual public | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Retrieval ms | p50/p95 s | Tokens | GPT-5.6 projection |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| public | B0_one_shot | 0.4000 | 1.00 | 0.5556 | 1.0000 | 1.0000 | 0.6667/0.6667 | 59.7 | 6.187/7.449 | 36077 | $0.150260 |
| public | B4a_bm25 | 0.4000 | 0.90 | 0.4444 | 0.8889 | 0.8889 | 0.5556/0.5556 | 15.2 | 5.052/6.478 | 34879 | $0.145516 |
| public | B4b_hybrid_rrf | 0.4000 | 0.90 | 0.6667 | 1.0000 | 1.0000 | 0.7778/0.7778 | 67.5 | 5.197/9.603 | 38763 | $0.161804 |
| public | B4c_finance_rerank | 0.4000 | 0.80 | 0.6667 | 1.0000 | 1.0000 | 0.6667/0.6667 | 70.3 | 5.569/9.702 | 38053 | $0.159172 |
| public | B4d_cross_encoder | 0.4000 | 0.90 | 0.3333 | 0.8889 | 0.8889 | 0.4444/0.4444 | 9808.1 | 15.303/23.785 | 39317 | $0.164868 |
| dev | B0_one_shot | 0.2500 | 0.55 | 0.3889 | 0.5556 | 0.7778 | 0.4444/0.4444 | 45.3 | 5.011/9.145 | 67358 | $0.281704 |
| dev | B4a_bm25 | 0.3000 | 0.60 | 0.5000 | 0.7778 | 0.8333 | 0.5556/0.5556 | 15.0 | 4.698/12.152 | 68552 | $0.287632 |
| dev | B4b_hybrid_rrf | 0.3000 | 0.65 | 0.5556 | 0.7778 | 0.8889 | 0.6111/0.6111 | 56.3 | 5.404/6.974 | 64131 | $0.270092 |
| dev | B4c_finance_rerank | 0.3500 | 0.60 | 0.5556 | 0.7778 | 0.8889 | 0.5556/0.5556 | 62.5 | 5.662/9.208 | 70917 | $0.298276 |
| dev | B4d_cross_encoder | 0.3000 | 0.80 | 0.5000 | 0.8333 | 0.8889 | 0.5556/0.5556 | 11280.1 | 16.622/20.153 | 68960 | $0.289632 |
| holdout | B0_one_shot | 0.3000 | 0.50 | 0.4444 | 0.6667 | 0.7778 | 0.3333/0.3333 | 61.2 | 4.619/8.990 | 37103 | $0.154844 |
| holdout | B4a_bm25 | 0.4000 | 0.70 | 0.4444 | 0.7778 | 0.8889 | 0.4444/0.4444 | 14.4 | 5.445/9.453 | 38448 | $0.161440 |
| holdout | B4b_hybrid_rrf | 0.4000 | 0.70 | 0.7778 | 0.7778 | 0.8889 | 0.5556/0.5556 | 60.8 | 5.236/8.625 | 34983 | $0.147452 |
| holdout | B4c_finance_rerank | 0.4000 | 0.70 | 0.7778 | 0.7778 | 0.8889 | 0.5556/0.5556 | 68.1 | 5.874/9.376 | 41310 | $0.172440 |
| holdout | B4d_cross_encoder | 0.4000 | 0.70 | 0.5556 | 0.8889 | 1.0000 | 0.4444/0.4444 | 11585.8 | 16.895/22.538 | 39369 | $0.165108 |

## Dense/BM25 complementarity on dev top-10 outputs

- Answerable questions: 18
- Dense Hit@10: 14/18
- BM25 Hit@10: 15/18
- Union of the two independently returned top-10 sets: 15/18
- Found only by BM25: dev-12
- Found only by dense: none
- Missed by both: dev-01, dev-10, dev-11

## Gold-page rank by dev question

| ID | Gold pages | B0 | B4a | B4b | B4c | B4d |
|---|---|---:|---:|---:|---:|---:|
| dev-01 | 20 | — | — | — | — | — |
| dev-02 | 20 | 5 | 4 | 4 | 4 | 3 |
| dev-03 | 21 | 1 | 1 | 1 | 1 | 5 |
| dev-04 | 59 | 1 | 3 | 1 | 1 | 1 |
| dev-05 | 59 | 1 | 1 | 1 | 1 | 1 |
| dev-06 | 60 | 3 | 1 | 1 | 1 | 2 |
| dev-07 | 71 | 1 | 1 | 1 | 1 | 1 |
| dev-08 | 117 | 1 | 2 | 2 | 2 | 1 |
| dev-09 | 118 | 10 | 1 | 1 | 1 | 1 |
| dev-10 | 151 | — | — | — | — | — |
| dev-11 | 177 | — | — | 5 | 6 | 1 |
| dev-12 | 205 | — | 1 | 9 | 10 | 6 |
| dev-13 | 238 | 7 | 6 | 6 | 5 | 3 |
| dev-14 | 257 | 7 | 1 | 1 | 1 | 1 |
| dev-15 | 257 | 6 | 1 | 1 | 1 | 3 |
| dev-16 | 257 | 3 | 5 | 1 | 1 | 2 |
| dev-17 | 257 | 1 | 2 | 2 | 2 | 1 |
| dev-18 | 286 | 1 | 1 | 1 | 1 | 1 |

Hybrid internally fuses top-20 from each source, so it can recover a gold page that sits below rank 10 in both individual outputs (for example dev-11).

## Manual semantic review

> Answer-level semantic correctness against the exact question and gold fact. Exact requested numbers, direction of change, all requested subparts, and correct refusal are required. Concise wording and alternative directly supporting report pages are accepted.

| Split | Arm | Accuracy | Incorrect IDs |
|---|---|---:|---|
| public | B0_one_shot | 1.00 | none |
| public | B4a_bm25 | 0.90 | sq-03 |
| public | B4b_hybrid_rrf | 0.90 | sq-01 |
| public | B4c_finance_rerank | 0.80 | sq-01, sq-03 |
| public | B4d_cross_encoder | 0.90 | sq-01 |
| dev | B0_one_shot | 0.55 | dev-02, dev-04, dev-09, dev-10, dev-11, dev-12, dev-13, dev-14, dev-15 |
| dev | B4a_bm25 | 0.60 | dev-02, dev-04, dev-10, dev-11, dev-12, dev-13, dev-15, dev-17 |
| dev | B4b_hybrid_rrf | 0.65 | dev-02, dev-04, dev-10, dev-12, dev-13, dev-15, dev-17 |
| dev | B4c_finance_rerank | 0.60 | dev-02, dev-04, dev-10, dev-11, dev-12, dev-13, dev-15, dev-17 |
| dev | B4d_cross_encoder | 0.80 | dev-02, dev-04, dev-12, dev-15 |
| holdout | B0_one_shot | 0.50 | holdout-02, holdout-04, holdout-06, holdout-07, holdout-09 |
| holdout | B4a_bm25 | 0.70 | holdout-02, holdout-06, holdout-07 |
| holdout | B4b_hybrid_rrf | 0.70 | holdout-02, holdout-04, holdout-07 |
| holdout | B4c_finance_rerank | 0.70 | holdout-02, holdout-04, holdout-07 |
| holdout | B4d_cross_encoder | 0.70 | holdout-02, holdout-04, holdout-07 |

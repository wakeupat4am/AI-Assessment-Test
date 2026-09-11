# B2 ablation — public

| Method | Auto | Dedup | Diversity | Auto accuracy | Hit@5 | Hit@10 | Citation P/R | Avg rounds | Trigger | Evidence tokens | Duplicate slots | Retrieval ms | E2E p50 s | LLM tokens |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_baseline_b1a | N | N | N | 0.3000 | 0.4444 | 0.6667 | 0.3333/0.3333 | 1.000 | 0.000 | 2350.9 | 0.50 | 93.2 | 4.829 | 34035 |
| B_page_dedup | N | Y | N | 0.3000 | 0.4444 | 0.6667 | 0.3333/0.3333 | 1.000 | 0.000 | 2350.9 | 0.30 | 66.7 | 4.856 | 34035 |
| C_dedup_diversity | N | Y | Y | 0.3000 | 0.3333 | 0.6667 | 0.3333/0.3333 | 1.000 | 0.000 | 2315.8 | 0.10 | 102.9 | 4.545 | 32924 |
| D_auto_search | Y | N | N | 0.3000 | 0.4444 | 0.6667 | 0.3333/0.3333 | 1.200 | 0.200 | 2180.9 | 0.50 | 72.2 | 4.532 | 28122 |
| E_full_b2 | Y | Y | Y | 0.3000 | 0.3333 | 0.7778 | 0.3333/0.3333 | 1.200 | 0.200 | 2391.0 | 0.10 | 112.0 | 3.956 | 27218 |

> Automatic answer accuracy is diagnostic; no new manual review is applied.

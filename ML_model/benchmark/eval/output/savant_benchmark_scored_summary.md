# Benchmark scoring run - Boltz-2 -> Savant + PRODIGY-LIG

500 pairs from `savant_benchmark_1000.csv`. Label: `neg_log10_affinity_M` (pKd), median 7.08.

## Coverage

| stage | n | % of pairs |
|---|---|---|
| Boltz-2 structure | 496 | 99.2 |
| Boltz-2 affinity | 496 | 99.2 |
| Savant pK | 0 | 0.0 |
| Savant classified | 0 | 0.0 |
| PRODIGY-LIG | 496 | 99.2 |
| PRODIGY-LIG (+elec) | 0 | 0.0 |
| PLAPT | 500 | 100.0 |

## Agreement with measured pKd

Correlation is the meaningful column; RMSE/MAE only mean something for predictions already on the pKd scale.

| prediction | n | pearson_r | spearman_rho | rmse | mae |
|---|---|---|---|---|---|
| Boltz-2 affinity | 496 | 0.412 | 0.393 | 1.388 | 1.078 |
| Savant pK | 0 | n/a | n/a | n/a | n/a |
| PRODIGY-LIG | 496 | 0.333 | 0.340 | 1.537 | 1.258 |
| PRODIGY-LIG (+elec) | 0 | n/a | n/a | n/a | n/a |
| PLAPT | 500 | 0.021 | -0.032 | 1.827 | 1.445 |


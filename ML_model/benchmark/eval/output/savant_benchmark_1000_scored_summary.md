# Benchmark scoring run - Boltz-2 -> Savant + PRODIGY-LIG

1000 pairs from `savant_benchmark_1000.csv`. Label: `neg_log10_affinity_M` (pKd), median 6.35.

## Coverage

| stage | n | % of pairs |
|---|---|---|
| Boltz-2 structure | 960 | 96.0 |
| Boltz-2 affinity | 960 | 96.0 |
| Savant pK | 647 | 64.7 |
| Savant classified | 935 | 93.5 |
| PRODIGY-LIG | 960 | 96.0 |
| PRODIGY-LIG (+elec) | 935 | 93.5 |

Every benchmark pair has a measured Kd, so all 1000 are true binders: Savant's classifier called 647/935 (69.2%) binder. Only those get a pK.

## Agreement with measured pKd

Correlation is the meaningful column; RMSE/MAE only mean something for predictions already on the pKd scale.

| prediction | n | pearson_r | spearman_rho | rmse | mae |
|---|---|---|---|---|---|
| Boltz-2 affinity | 960 | 0.471 | 0.444 | 1.593 | 1.235 |
| Savant pK | 647 | 0.156 | 0.159 | 1.771 | 1.410 |
| PRODIGY-LIG | 960 | 0.197 | 0.205 | 2.142 | 1.696 |
| PRODIGY-LIG (+elec) | 935 | 0.189 | 0.204 | 2.292 | 1.698 |

## By novelty tier

| tier | prediction | n | pearson_r | spearman_rho | rmse |
|---|---|---|---|---|---|
| novel_family | Boltz-2 affinity | 179 | 0.337 | 0.261 | 1.645 |
| novel_family | Savant pK | 98 | 0.015 | 0.070 | 2.073 |
| novel_family | PRODIGY-LIG | 179 | 0.198 | 0.141 | 2.161 |
| novel_family | PRODIGY-LIG (+elec) | 176 | 0.173 | 0.079 | 3.096 |
| same_family | Boltz-2 affinity | 399 | 0.398 | 0.371 | 1.624 |
| same_family | Savant pK | 282 | 0.186 | 0.206 | 1.600 |
| same_family | PRODIGY-LIG | 399 | 0.161 | 0.174 | 1.994 |
| same_family | PRODIGY-LIG (+elec) | 391 | 0.207 | 0.216 | 1.878 |
| same_protein | Boltz-2 affinity | 382 | 0.559 | 0.550 | 1.534 |
| same_protein | Savant pK | 267 | 0.175 | 0.147 | 1.822 |
| same_protein | PRODIGY-LIG | 382 | 0.211 | 0.232 | 2.279 |
| same_protein | PRODIGY-LIG (+elec) | 368 | 0.219 | 0.227 | 2.238 |


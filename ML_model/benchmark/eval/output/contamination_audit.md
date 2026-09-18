# Contamination audit - is Boltz-2's lead memorisation?

Benchmark: `savant_benchmark_scored.csv` (500 pairs)  
Reference: ChEMBL v34 (2024-04-15), the affinity-module training source  
Evidence available: pair_seen, ligand_seen  
Ligand matching: InChIKey skeleton (first 14 chars)

## Strata

| stratum | n | share |
|---|---:|---:|
| `pair_seen` | 704 | 70.4% |
| `ligand_seen` | 175 | 17.5% |
| `analog_seen` | 0 | 0.0% |
| `unseen` | 121 | 12.1% |

## Accuracy by stratum

`label_std` is the pKd spread within the stratum. Pearson r scales with it, so compare r only between strata with similar spread.

| tool | stratum | n | label_std | Pearson r | 95% CI | RMSE | pred_std_ratio |
|---|---|---:|---:|---:|---|---:|---:|
| Boltz-2 | `pair_seen` | 679 | 1.71 | +0.461 | [+0.390, +0.525] | 1.605 | 0.643 |
| Boltz-2 | `ligand_seen` | 165 | 1.68 | +0.545 | [+0.414, +0.667] | 1.428 | 0.575 |
| Boltz-2 | `unseen` | 116 | 1.80 | +0.443 | [+0.266, +0.591] | 1.738 | 0.506 |
| Boltz-2 | `ALL` | 960 | 1.72 | +0.471 | [+0.414, +0.526] | 1.593 | 0.614 |
| Savant v3 | `pair_seen` | 654 | 1.71 | +0.238 | [+0.158, +0.311] | 1.678 | 0.362 |
| Savant v3 | `ligand_seen` | 162 | 1.69 | +0.255 | [+0.106, +0.404] | 1.640 | 0.345 |
| Savant v3 | `unseen` | 115 | 1.80 | +0.303 | [+0.126, +0.457] | 1.708 | 0.341 |
| Savant v3 | `ALL` | 931 | 1.72 | +0.256 | [+0.195, +0.317] | 1.675 | 0.359 |
| PRODIGY-LIG | `pair_seen` | 679 | 1.71 | +0.174 | [+0.105, +0.244] | 2.198 | 0.959 |
| PRODIGY-LIG | `ligand_seen` | 165 | 1.68 | +0.218 | [+0.070, +0.362] | 2.084 | 0.909 |
| PRODIGY-LIG | `unseen` | 116 | 1.80 | +0.314 | [+0.148, +0.472] | 1.875 | 0.652 |
| PRODIGY-LIG | `ALL` | 960 | 1.72 | +0.197 | [+0.138, +0.257] | 2.142 | 0.915 |

## Difference-in-differences

`(Boltz-2 seen - unseen) - (Savant v3 seen - unseen)`. Savant v3 never saw ChEMBL v34, so its own gap is stratum difficulty; subtracting it leaves the part attributable to contamination.

| quantity | value |
|---|---:|
| Boltz-2 gap | +0.046 |
| Savant v3 gap | -0.065 |
| **difference-in-differences** | **+0.111** |
| 95% CI | [-0.090, +0.299] |
| one-sided bootstrap p | 0.142 |

**Reading it.** A CI spanning 0 means this benchmark cannot detect contamination-driven inflation at this sample size - the headline comparison stands as reported, and you can say so. A CI clearly above 0 means Boltz's lead is partly memorisation and the benchmark should be rebuilt from post-v34 data (ChEMBL 35+, publication year >= 2024).

---

Caveat that belongs in the paper: Boltz-2's training set was never released, so ChEMBL v34 membership is a *proxy* for what the model saw, not set subtraction. It also drew on BindingDB and PubChem, so `unseen` here means 'not in v34', not 'not in training'. This bounds the contamination; it does not eliminate it.

# Phase 1 - affinity harvesting and harmonisation

Kd and Ki only. IC50/EC50 are excluded at every loader (PLAN.md Section 0.4 rule 1).

## Corpus

- measurements after dedupe: **1,008,442**
- unique (protein, ligand) pairs: **682,755**
- proteins: **12,298**  ligands: **281,084**
- types: {'Ki': 530013, 'Kd': 152742}
- censored (>/<): 144,534 (one-sided loss, Section 5.2)
- sequences over the 1022-aa PLM window: 161,920 (cropped; metrics reported by length bin)
- ligands per protein: median 1, max 10,086

Section 4.4 expects several hundred thousand Kd/Ki pairs, Ki-dominated, after dedupe against ChEMBL. Compare.

## Per source

- `chembl`: 703,758 rows, types {'Ki': 527003, 'Kd': 176755}, relations {'=': 531965, '>': 152414, '<': 19379}
- `bindingdb`: 721,888 rows, types {'Ki': 593504, 'Kd': 128384}, relations {'=': 595930, '<': 99801, '>': 26157}
- `papyrus`: 66,708 rows, types {'Ki': 54162, 'Kd': 12546}, relations {'=': 66708}
- `iuphar`: 4,969 rows, types {'Ki': 3912, 'Kd': 1057}, relations {'=': 4823, '<': 98, '>': 48}
- `pdbbind`: 12,000 rows, types {'Kd': 7039, 'Ki': 4961}, relations {'=': 11917, '>': 83}

## Dropped

| reason | n |
|---|---|
| too_large | 24,740 |
| bad_smiles | 3,697 |
| too_small | 1,306 |
| organometallic | 322 |
| standardize_failed | 12 |
| bad_sequence | 342 |
| peptide_like | 45,560 |
| quarantined | 706 |

Counts only - Section 5.2 forbids carrying the dropped values forward.

## Noise ceiling (measured on our own replicates)

```json
{
 "n_replicated": 170461,
 "frac_of_corpus": 0.24966642499871844,
 "median_spread_pK": 0.0,
 "mean_spread_pK": 0.28797451376841104,
 "frac_spread_gt_1": 0.09453775350373399,
 "implied_sd_pK": 0.0,
 "note": "an RMSE below the implied sd is fitting noise, not signal"
}
```

Appendix A asks for this to be quantified rather than borrowed. An RMSE below the implied sd is fitting noise.

## Ki vs Kd on identical pairs

- n = 5,135
- median Ki - Kd = +0.145 pK
- |delta| > 1 log unit: 25.7%

This is the evidence for or against pooling Ki with Kd (`label_types: [Kd, Ki]`). The Kd-only ablation is in `configs/preregistration.yaml`.

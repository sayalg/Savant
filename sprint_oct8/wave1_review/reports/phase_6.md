# Phase 6 - baselines (Gate G1)

Every later result is relative to these. PLAN.md Section 0.3.2 requires
`ligand_only` and `target_mean` in every table.

On the previous corpus, ligand-only reached median per-target rho **+0.325**
on 613 held-out protein clusters while the best 7,680-dimension ESM model
reached **0.158**. Read the ligand_only row first.

`target_mean` is an ORACLE, not a model: it is given each test target's
own mean and is unavailable at inference. `const` is the RMSE floor.
Both are constant within a target, so neither has a per-target rho.

## split: `protein_cluster`

| model | pearson | per-target rho | RMSE | RMSE const | n targets | EF@10 |
|---|---|---|---|---|---|---|
| `const` *(floor)* | - | - | 1.641 | 1.641 | 3476 | - |
| `target_mean` *(oracle)* | +0.672 | - | 1.149 | 1.641 | 3476 | - |
| `ligand_only` **(the floor to beat)** | +0.624 | +0.350 | 1.217 | 1.551 | 655 | 1.66 |
| `protein_only` | +0.194 | - | 1.609 | 1.551 | 3476 | - |
| `fusion_gbdt` | +0.334 | +0.093 | 1.531 | 1.551 | 655 | 0.97 |

> **`fusion_gbdt` is BELOW the ligand-only floor** by 0.290 Pearson. The protein features are hurting, not merely failing to help.

Shortcut check (Section 5.3.5):

```json
{
 "fusion_gbdt": {
  "shortcut": true,
  "below_ligand_only": true,
  "delta_pearson_vs_ligand_only": -0.29019440938646207,
  "note": "the model is BELOW the ligand-only floor by 0.290 Pearson: the protein features are hurting, not merely failing to help"
 }
}
```

## split: `double`

| model | pearson | per-target rho | RMSE | RMSE const | n targets | EF@10 |
|---|---|---|---|---|---|---|
| `const` *(floor)* | - | - | 1.617 | 1.617 | 1303 | - |
| `target_mean` *(oracle)* | +0.694 | - | 1.076 | 1.617 | 1303 | - |
| `ligand_only` **(the floor to beat)** | +0.392 | +0.082 | 1.485 | 1.495 | 351 | 0.96 |
| `protein_only` | +0.345 | - | 1.470 | 1.495 | 1303 | - |
| `fusion_gbdt` | +0.469 | +0.143 | 1.330 | 1.495 | 351 | 1.00 |

Shortcut check (Section 5.3.5):

```json
{
 "fusion_gbdt": {
  "shortcut": false,
  "below_ligand_only": false,
  "delta_pearson_vs_ligand_only": 0.07652085425297245,
  "note": "model is meaningfully above the ligand-only floor"
 }
}
```

## split: `scaffold`

| model | pearson | per-target rho | RMSE | RMSE const | n targets | EF@10 |
|---|---|---|---|---|---|---|
| `const` *(floor)* | - | - | 1.568 | 1.568 | 4641 | - |
| `target_mean` *(oracle)* | +0.642 | - | 1.201 | 1.568 | 4641 | - |
| `ligand_only` **(the floor to beat)** | +0.620 | +0.259 | 1.248 | 1.568 | 1388 | 1.50 |
| `protein_only` | +0.595 | - | 1.261 | 1.568 | 4641 | - |
| `fusion_gbdt` | +0.787 | +0.450 | 0.981 | 1.568 | 1388 | 1.87 |

Shortcut check (Section 5.3.5):

```json
{
 "fusion_gbdt": {
  "shortcut": false,
  "below_ligand_only": false,
  "delta_pearson_vs_ligand_only": 0.1670686118093231,
  "note": "model is meaningfully above the ligand-only floor"
 }
}
```

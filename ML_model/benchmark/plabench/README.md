# Savant on PLABench's leakage-filtered ChEMBL35 set

External validation against a benchmark we had no hand in building.

PLABench (Wang & Cheng, 2026; code [Zenodo 10.5281/zenodo.22817206], data
[Zenodo 10.5281/zenodo.22716174]) scores 9 methods on a ChEMBL35 test set
filtered for ligand leakage against 14 training corpora. It covers 71 targets
against the 20 our own benchmark spans, and its numbers were produced by someone
else, which is what makes it worth running.

## Running it

```bash
python scripts/01_score_savant.py            # ~20 min, CPU, 4,862 pairs
python scripts/02_audit_contamination.py     # ~25 min, caches training fingerprints
python scripts/03_evaluate.py                # Savant inside their published table
python scripts/04_compare_decontaminated.py  # every method on the same subset
```

`01` needs `ML_model/models_full/savant_full_esm650.pkl` and the `savant` package
at the repository root. `02` needs `input/input.csv`.

## The result

Their headline "Pearson" is the N-weighted mean of per-target Pearson, not a
pooled correlation. `03` and `04` both reproduce their 9 published numbers from
their own files before adding Savant, so a protocol mismatch would surface as a
reproduction failure rather than as a difference flattering us.

On the full filtered set Savant places 4th of 10. On the subset that also clears
Savant's own 0.50 gate, with every method rescored on those same rows:

| Method | Pearson | Spearman | RMSE |
|---|---|---|---|
| flowr_root | 0.3296 | 0.3228 | 1.0206 |
| boltz2 | 0.3013 | 0.2971 | 1.3735 |
| llf | 0.2391 | 0.2332 | 1.5858 |
| **Savant** | **0.2266** | **0.2232** | **1.0282** |
| deepdta | 0.2033 | 0.2021 | 1.4111 |
| mixingdta | 0.1840 | 0.1706 | 1.4172 |
| flowdock | 0.1745 | 0.1641 | 1.3538 |
| bapred | 0.1641 | 0.1419 | 1.2753 |
| mfe | 0.1549 | 0.1700 | 1.3013 |
| haiping | 0.0940 | 0.0945 | 1.5411 |

2,025 pairs across 63 targets. Savant is mid-field on correlation and 2nd on
RMSE, behind flowr_root and well ahead of boltz2.

## Why the decontamination step exists

PLABench's filter protects their 9 models, not ours: Savant's corpus
(`jglaser/binding_affinity`) is not among the 14 they audit, and their gate is
ECFP4 Tanimoto 0.85 against our 0.50.

The audit found **0 InChIKey skeleton matches** and **0 ligands at or above
0.85** — their filter does transfer to our corpus, because ours overlaps
BindingDB, PDBbind and Binding MOAD, which they audit. The highest similarity to
anything Savant trained on is 0.849, just under their cut. But 52.1% of their
ligands sit at or above 0.50, so 58.3% of pairs fail our own standard.

Savant moves 0.2518 → 0.2266 when those are removed. A model leaning on
memorisation would fall much further, so the small drop is evidence that it is
not.

## What this says that our own benchmark cannot

Savant's between-target correlation here is **0.51**, against 0.10 on our 500-pair
benchmark. Our benchmark selects pairs to span affinity bins within each target,
which makes its between-target axis adversarial by construction; PLABench does
not do that. The near-zero between-target figure we report is therefore partly a
property of our benchmark rather than a flat statement about the method, and
that should be said plainly rather than left as an unqualified limitation.

## Caveats

- 2,025 pairs over 63 targets is a smaller set than the full 4,862, and the
  intervals around every number here are correspondingly wider. PLABench
  publishes bootstrap CIs for the full set; we have not computed them for the
  subset.
- Their affinity column mixes assay types, as ours does.
- 4 pairs Savant scored lack predictions from one or more of their models, so the
  cross-method table uses the 2,025 common to all.
- `boltz2` is reported by PLABench on this filtered set for all 71 targets, but
  their unfiltered summary covers only 559 pairs across 10 series. We did not
  chase that discrepancy; it is theirs, not ours.

## Outputs

| File | What |
|---|---|
| `output/savant_chembl35_filtered.csv` | Savant pKd for all 4,862 pairs |
| `output/contamination_audit.csv` | per-ligand max Tanimoto to Savant's training set |
| `output/comparison_vs_plabench.csv` | Savant inside their published table |
| `output/comparison_decontaminated.csv` | every method, both scopes |
| `output/savant_per_target_*.csv` | per-target metrics |

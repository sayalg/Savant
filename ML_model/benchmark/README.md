# The Savant benchmark — 500 Kd/Ki pairs, 20 targets, training-disjoint

The evaluation set of the paper. One row per (target, ligand) pair, built by
`scripts/build_benchmark.py` from ChEMBL and gated three ways, each aimed at a
specific way a benchmark can flatter a model.

An earlier 1000-pair build was retired: it left only 33 targets carrying ≥ 8
pairs, too few to report within-target ranking, so its headline number was
dominated by between-target variance — on it, a protein-mean lookup using no
ligand information beat both Savant and Boltz-2. That build is on the `dev`
branch; nothing here depends on it.

## The three gates

1. **Temporal, for the comparison methods.** Only measurements from documents
   dated 2025 or later, and only ligands whose InChIKey skeleton is absent from
   ChEMBL v34 (2024-04-15) — the release most plausibly inside the training
   window of the methods compared. Boltz-2's training set was never published,
   so this *bounds* contamination rather than eliminating it. Say so when
   reporting.
2. **Ligand novelty, for Savant.** ECFP4 Tanimoto < 0.50 to every ligand in
   `Input/input.csv`, plus no shared InChIKey skeleton. This gates against the
   full 991 k-ligand corpus, not the 32 k in `combined_50k_filtered.csv`. Only
   about 12% of typical med-chem ligands clear it. The realized maximum in the
   shipped set is **0.494**.
3. **Target-first selection, for the metric.** Targets are selected before pairs:
   only targets carrying at least `--min-per-target` qualifying pairs with real
   label spread are admitted, so within-target Spearman ρ is computable across
   the whole set rather than on a minority of it.

**Assay filtering:** Kd and Ki only, at a ChEMBL confidence score of 8 or
greater. Censored measurements pinned at the round reporting caps (10, 100,
1,000, 10,000 µM) are dropped, as are records above pKd 12 — those are source
unit errors, not measurements.

## Proteins are deliberately not gated

`Input/input.csv` holds 22,836 sequences covering essentially every well-measured
target, so requiring novel proteins would leave only sparsely characterized ones,
for which the ground truth is itself unreliable — and short of pairs per target.
All 20 benchmark targets therefore appear in training **with other ligands**.

This set measures the ranking of novel chemistry against an established target,
which is the realistic screening scenario. Results from it must not be described
as zero-shot with respect to the target.

## Label

`neg_log10_affinity_M` is pKd = 9 − log10(Kd [nM]), the same target the regressor
was trained on. `deltaG_kcal_per_mol` = −1.364 × pKd at 298 K.

## Build it

```bash
cd ML_model/benchmark
bash eval/scripts/fetch_chembl.sh          # ChEMBL 37 sqlite
bash eval/scripts/fetch_chembl34.sh        # ChEMBL 34 sqlite, for the temporal gate

python scripts/build_benchmark.py \
    --chembl37 work/chembl_37/chembl_37.db \
    --chembl34 work/chembl_34/chembl_34.db \
    --train ../../Input/input.csv \
    --workers 16
```

Output: `output/savant_benchmark.csv`, which is committed — so the scoring
pipeline runs from a fresh clone without rebuilding it.
`eval/output/bench_scored.csv` carries the same 500 rows with every method's
scores appended.

Selection is reproducible from this directory alone under the fixed seed in
`scripts/common.py`.

## Then score it

`eval/` runs Boltz-2, Savant, PRODIGY-LIG and PLAPT over the same 500 pairs and
computes every table in the paper. See [`eval/README.md`](eval/README.md).

## Enrichment factors are not reported

The set is 88.6% binders, a density at which enrichment saturates near 1.13 and
cannot separate a perfect ranker from a mediocre one. Truchon and Bayly showed
the enrichment factor is in any case insensitive to ranking performance before
and after its cutoff. Within-target Spearman ρ is the metric used instead.

## Layout

```
benchmark/
├── output/savant_benchmark.csv   the 500 pairs
├── scripts/
│   ├── common.py                shared constants, censor caps, the fixed seed
│   └── build_benchmark.py       ChEMBL + input.csv -> the 500 pairs
└── eval/                        the four-tool comparison
```

`work/` and `cache/` hold the ChEMBL downloads and API caches; both are
gitignored and resumable.

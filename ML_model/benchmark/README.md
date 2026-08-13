# Savant benchmark v2 — 1000 Kd pairs, class-stratified, training-disjoint

An external benchmark of ~1000 protein–ligand pairs with **empirically measured Kd**,
spanning a wide range of protein classes and ligand/drug classes, for testing where
Savant works and where it breaks — and for running the same pairs through
AutoDock Vina, FEP+, and Boltz-2.

This supersedes the old `benchmark_builder/`, which produced a benchmark with
three defects (see [Why v2](#why-v2)). Nothing here depends on it.

## Deliverable

| file | what |
|---|---|
| `output/savant_benchmark_1000.csv` | the benchmark, one row per (target, ligand) pair |
| `output/savant_benchmark_targets.fasta` | unique target sequences, keyed by UniProt accession |
| `output/savant_benchmark_ligands.smi` | unique ligands, `SMILES<TAB>InChIKey` |
| `output/savant_benchmark_1000_summary.md` | QC report + class distributions |

### Label

`neg_log10_affinity_M` is **pKd = 9 − log10(Kd[nM])**, the same target the Savant
regressor was trained on. `deltaG_kcal_per_mol` = −1.364 × pKd (298 K).

Censored (`>X`) measurements pinned at BindingDB's round caps are dropped, as are
records with **pKd > 12** (Kd < 1 pM). The latter are source unit errors, not real
measurements — BindingDB carries a 268 Da catecholamine logged at 10 fM against the
β1-adrenergic receptor, which is physically impossible for a drug-like small
molecule. A single label 8 log units out visibly distorts RMSE and Pearson r, so
these are removed rather than shipped. Retained extremes (trypsin at pKd 11.7, HuR
at 11.0, NTRK1 at 10.8) are all physically sensible.

### Disjointness from the training set

Overlap is controlled on both axes, but they are enforced differently and it
matters which is which:

- **Ligand — a hard gate.** Every pair whose ligand is an exact structure match,
  an InChIKey-skeleton match, or **ECFP4 Tanimoto ≥ 0.50** to *any* of the 32,086
  training ligands was dropped from the candidate pool before selection. Nothing
  in the benchmark is chemically close to a training ligand, so no
  (protein, ligand) pair can be a training pair.
- **Protein — a label, not a gate.** Each pair carries a `novelty_tier`:

  | tier | meaning | probes |
  |---|---|---|
  | `same_protein` | accession is in training (or ≥0.90 8-mer containment) | new chemistry on a known target |
  | `same_family` | new protein, UniProt family seen in training | transfer within a known fold |
  | `novel_family` | protein family absent from training entirely | unseen biology |

  Tiering rather than hard-filtering the protein axis is deliberate: a strict
  30%-identity cut removes every kinase, protease, GPCR and nuclear receptor —
  i.e. essentially all of druggable space — which is incompatible with spanning a
  variety of protein classes. Splitting performance by tier answers the actual
  question ("where does Savant struggle?") better than a single filtered number.
  Filter to `novelty_tier == "novel_family"` if you want the strict held-out set.

### Stratification

Selection round-robins over **novelty tier → ChEMBL protein class → pKd bin**,
capped at 20 pairs per target and 12 per Bemis–Murcko scaffold, so no target,
scaffold, class or affinity range dominates. It is reproducible from this
directory alone under the fixed seed in `common.py`.

## Inputs and self-containment

Everything the build needs lives here, except one file:

| path | role |
|---|---|
| `../combined_50k_filtered.csv` | **the only external dependency** — training ligands for the overlap gate |
| `reference/train_proteins.csv` | 11,965 training protein chains (k-mer index, mmseqs target DB) |
| `reference/train_protein_families.csv` | training UniProt accessions + 912 protein families (novelty tiers) |
| `work/bindingdb_kd.tsv` | raw source, Dataverse file 4291555 (pinned snapshot) |

`reference/` is vendored from what used to be `ML_model/taxonomy/output/`. Keep it:
`train_protein_families.csv` derives from a ~4 h RCSB + UniProt annotation run and
cannot be cheaply regenerated. With it in place, `ML_model/taxonomy/` and
`ML_model/benchmark_builder/` can both be deleted.

### What is safe to delete

Regenerable at any time: `work/candidates.csv` (step 01, ~3 min),
`work/selected.csv` (step 04, seconds), `work/mmseqs/` (step 07, seconds),
`work/*.log`, and the `cache/chembl_*` + `cache/uniprot` subdirectories (re-query
in minutes).

Do **not** delete `cache/classyfire/` — it holds ~11 h of rate-limited ChemOnt
lookups. `work/protein_annotation.csv` and `work/ligand_annotation.csv` are also
worth keeping: with them, step 05 rebuilds with no network access at all.

## Pipeline

```bash
cd scripts
python 01_filter_candidates.py     # clean BindingDB, ligand gate, same-protein flags
python 03_annotate_proteins.py     # UniProt + ChEMBL target class per accession
python 04_select.py                # stratified selection -> work/selected.csv
python 02_annotate_ligands.py --from-selected   # ClassyFire + ChEMBL (slow, cached)
python 05_build_final.py           # assemble + QC gate -> output/
```

Step 02 is run *after* selection on purpose: ClassyFire rate-limits to roughly
10–15 lookups/min, so classifying the ~850 selected ligands takes ~1 h whereas the
full 6,381-ligand candidate pool would take ~11 h. Every response is cached under
`cache/`, so re-runs resume. `--assemble` rebuilds the CSV from cache without
touching the network.

Step 01 needs the BindingDB Kd table at `work/bindingdb_kd.tsv` (TDC-curated,
Harvard Dataverse file 4291555, ~54 MB); the Dataverse API rejects the default
Python user-agent, so pass a browser UA when fetching it.

## Using it with the other tools

The file carries sequence + SMILES, which is what **Savant** and **Boltz-2** need.
Two caveats for the structure-based comparisons:

- **AutoDock Vina** needs a 3D receptor and a defined search box per target. There
  are 355 unique targets here; you will need a holo structure and pocket
  definition for each. `output/savant_benchmark_targets.fasta` is the input for a
  RCSB sequence search (`search.rcsb.org`) to find a representative holo PDB entity
  per target.
- **FEP+** computes *relative* binding free energies within a congeneric series.
  This set is deliberately scaffold-diverse (639 scaffolds / 1000 pairs), so most
  targets carry too few analogs to form a perturbation map. Expect to run FEP+ on
  a hand-picked subset, or rebuild with congeneric blocks reserved.

## Why v2

The previous build (`benchmark_builder/output/bindingdb_kd_benchmark_1000.csv`,
now removed) had three problems, all confirmed by direct audit:

1. **pKd was computed as `6 − log10(Kd[nM])`** instead of `9 − log10(Kd[nM])`, so
   every label sat exactly 3 log units below the training convention (benchmark
   median 5.35 vs training median 6.49).
2. **32% of its pairs leaked.** 72 of its 224 targets — including BRAF, NTRK1 and
   PARP1 — matched a training protein at ≥0.9 8-mer containment. The protein
   filter never ran: `mmseqs` is not installed locally, `work/` was never created,
   and the shipped summary reported median training identity 1.0 without failing.
3. **It held 581 pairs, not 1000**, with no taxonomic stratification.

Its ligand filter was sound — 0 exact matches and 0 above Tanimoto 0.5 — and that
logic is carried forward unchanged.

`05_build_final.py` now **exits non-zero** if the pKd convention drifts, a ligand
crosses the Tanimoto gate, or pairs duplicate, so a broken build fails instead of
shipping.

## Note on the protein-similarity method

`mmseqs` is unavailable on this machine (it lives in the HPC colabfold container),
so same-protein detection uses exact UniProt accession matching plus 8-mer
containment. That is reliable for near-identical proteins but **cannot** detect
remote homology at ~30% identity — no pure-Python k-mer heuristic can. Family-level
novelty therefore comes from UniProt protein-family membership, not from sequence
identity.

### Confirming the tiers with mmseqs on the HPC

```bash
python scripts/07_export_mmseqs_inputs.py     # local: writes work/mmseqs/
```

Copy `work/mmseqs/` and `run_mmseqs_tiers.sbatch` to the cluster, point
`CONTAINER` at the image carrying `mmseqs` (the colabfold `.sif` already has it),
then:

```bash
sbatch run_mmseqs_tiers.sbatch                # ~10 min: 355 queries vs 11,965 chains
```

Copy `bench_vs_train.m8` back and reconcile:

```bash
python scripts/08_reconcile_mmseqs.py --m8 work/mmseqs/bench_vs_train.m8
```

The search uses `--min-seq-id 0.30 -c 0.80 --cov-mode 0 -s 7.5`, matching the
project's clustered-split thresholds, and maps identity onto the same three tiers
(`>=0.90` same_protein, `0.30-0.90` same_family, no hit novel_family). The report
prints a shipped-vs-mmseqs confusion matrix, lists every disagreeing target, and
flags **leaks** — anything tiered below `same_protein` that mmseqs finds is ≥90%
identical to a training chain. Add `--apply` to adopt the mmseqs tiers (the local
ones are preserved as `novelty_tier_local`).

Expect some movement in the `same_family` / `novel_family` boundary: the local
method assigns family novelty from UniProt annotation, whereas mmseqs measures
actual sequence identity, and it will catch remote homologs that annotation
misses. Movement *into* `same_protein` would be a real leak and should be
investigated.

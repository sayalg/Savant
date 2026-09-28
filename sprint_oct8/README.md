# pfpl — pose-free, physics-informed protein–ligand affinity

Implementation of [`PLAN.md`](PLAN.md). Read that first; this file is how to run it.

**Input at inference:** an amino-acid sequence and a ligand SMILES. No structure, no
docking, no folding. **Output:** predicted pKd, an uncertainty, and an interpretable
decomposition — per-term, per-residue and per-ligand-atom contributions plus the
predicted pocket.

**Label policy, enforced in code:** the target is pKd, trained on **Kd and Ki only**.
IC50/EC50 never enter as labels, features, weights, filters or selection criteria
(`src/pfpl/data/schema.py`, `tests/test_labels.py`). A source whose assay type cannot
be established is *refused*, not guessed.

---

## What this replaces, and why

The previous sprint's results are kept at
[`reports/legacy_v1_sprint_results.md`](reports/legacy_v1_sprint_results.md). Its
headline finding is the reason this codebase is shaped the way it is:

> Across 613 held-out protein clusters, a **ligand-only** model — ECFP plus 20 RDKit
> descriptors, **no protein input of any kind** — reached median per-target Spearman
> **+0.325**. The best model with 7,680 ESM dimensions reached **0.158**. Every
> protein-aware model lost to a protein-blind control by roughly a factor of two.

Three consequences are built in rather than remembered:

1. **Baselines come first** (`scripts/phase6_baselines.py`, Gate G1). Nothing may be
   judged against numbers that do not exist yet.
2. **The PLM is an empirical question** (`configs/plm/*.yaml`). `plm.mode=none` is a
   serious candidate, not a control, and it is in the pre-registered matrix.
3. **Every physics claim is measured against a parameter-matched no-physics control**
   (`model=variant_c`), not against a baseline table.

An earlier flagship result — per-target rho +0.28 on influenza A neuraminidase — turned
out to be measured on a protein that was in its own training fold, with 55 of 81
benchmark ligands present as training pairs. Benchmark quarantine now happens in Phase 0,
before the corpus exists, and `tests/test_leakage.py` re-checks it.

---

## Layout

```
configs/          composable YAML: data / model / plm / train + preregistration.yaml
src/pfpl/
  paths.py        root resolution that never depends on the cwd
  conf.py         mini-Hydra: defaults groups, _inherit, dotted overrides, config hash
  preflight.py    dependency guards (NumPy ABI, GLIBCXX, CUDA) that fail in 2 seconds
  ledger.py       compute ledger + pre-registration check (Section 7.4)
  data/           schema (label policy), standardise, sources/, harmonise, splits, quarantine
  features/       ligand physics tiers A-C, ESM per-residue, ligand graph, geometry labels, cache
  physics/        sites, terms, parameter heads, expected-energy assembly
  models/         trunk, pocket, pair, distogram, coordinate hypotheses, the assembled model
  train/          the shared loop: losses, checkpoint/resume, calibration monitor
  baselines/      target_mean, ligand_only, protein_only, fusion
  eval/           metrics with the mandatory baselines and Section 9's gate
scripts/          one entry point per phase
slurm/            sbatch files + the two wave launchers
tests/            physics, labels, leakage, harmonisation, cache/config/paths
```

### There is no `cd` in any Slurm script

The old scripts opened with `cd "$(dirname "$0")/.."`, which breaks whenever `$0` is
not the script's on-disk path — Slurm copies a `--wrap` body to a spool directory, a
symlinked launcher resolves to the link, a container bind-mount rewrites the prefix.
Each of those produced a job that ran in the wrong directory and wrote its artifacts
where nothing would look for them.

Instead, `slurm/_prelude.sh` resolves `PFPL_ROOT` once (from the environment or
`SLURM_SUBMIT_DIR`) and everything uses absolute paths under it. `src/pfpl/paths.py`
re-derives the same root independently from the package location, so the shell and
Python sides cannot disagree. Slurm already starts a job in the directory `sbatch` was
run from, which is why no `cd` was needed in the first place.
`tests/test_cache.py` fails if one creeps back in.

Submit from the project root, or set it explicitly:

```bash
PFPL_ROOT=/scratch/$USER/Savant/sprint_oct8 sbatch slurm/40_embed_protein.sbatch
```

### One root: everything lives at `/scratch/$USER/Savant/sprint_oct8`

Code, data, caches, logs, temp files. Nothing is written to `$HOME` or `/tmp`.

```bash
cd /scratch/$USER/Savant/sprint_oct8
python -m pfpl.paths          # prints every path; exits 1 if anything escapes
```

That is not tidiness. HPC `$HOME` is small and quota'd, and `/tmp` on a compute node
is node-local and reclaimed when the job ends — so an unredirected HuggingFace or
Triton cache is how a job dies with "Disk quota exceeded" six hours into an embedding
array, and an unredirected `TMPDIR` is how an `xtb` shard silently loses its scratch
files mid-run.

`src/pfpl/paths.py:contain()` redirects them all on import — `HF_HOME`,
`HUGGINGFACE_HUB_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`, `MPLCONFIGDIR`,
`TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, `NUMBA_CACHE_DIR`, `WANDB_DIR`, and
`TMPDIR`/`TMP`/`TEMP`. `slurm/_prelude.sh` applies the identical map for the shell,
and every job runs `python -m pfpl.paths` before doing any work, so a leak fails the
job in one second instead of after six hours.

A value that already points inside the root is left alone. A var is *reclaimed* when
it points outside — pure setdefault is not enough, because many shells pre-set `TMP`
and `TEMP`, and `tempfile` reads `TMPDIR`, `TEMP`, `TMP` in that order.

| Path | Default |
|---|---|
| `PFPL_ROOT` | `/scratch/$USER/Savant/sprint_oct8` |
| `PFPL_WORK` | `$PFPL_ROOT/work` — corpus, caches, runs, tmp, hf |
| `splits/`, `reports/`, `logs/` | under `$PFPL_ROOT` |

`splits/` and `reports/` stay under the root *and* in version control — PLAN.md
Section 5.3 requires the splits to be frozen, and every number is measured against a
specific one.

Two escape hatches, both deliberate and both visible in `python -m pfpl.paths`:

```bash
PFPL_ALLOW_EXTERNAL=HF_HOME,HUGGINGFACE_HUB_CACHE   # e.g. a read-only site model cache
PFPL_NO_CONTAIN=1                                    # disable redirection entirely
```

---

## Running it

### Once, before anything else

```bash
python -m pfpl.preflight                       # is this environment usable?
python scripts/phase0_verify.py --benchmark ../ML_model/benchmark/eval/output/bench_scored.csv
```

Phase 0 writes `DATA_MANIFEST.md` with all 18 Appendix-A claims marked **UNVERIFIED**
and freezes the quarantine keys to `splits/quarantine.json`. The manifest rows are not
self-verifying; Section 0.3.9 makes working through them a gate.

That benchmark is 500 pairs / 20 targets / 373 ligands, all 2025 documents (a pure
time split), Kd/Ki only, ligand-gated at Tanimoto ≤ 0.494 against the old training
set, with Boltz-2 already scored on 496 of them (pooled r 0.412, median per-target
ρ 0.242). Because its ligand axis is already gated, quarantine costs the corpus
almost nothing here.

### Building the training corpus from scratch

Data collection is its own phase, and it is the only step that touches the network:

```bash
python scripts/phase1a_acquire.py --all        # or: sbatch slurm/05_acquire.sbatch
python scripts/phase1a_acquire.py --status
```

| source | what | notes |
|---|---|---|
| `chembl` | binding-assay Kd/Ki | `assay_type=B`, confidence 9, nM. Checkpointed — a wall-time kill resumes |
| `bindingdb` | ~560k Ki + ~100k Kd | 1.5 GB, resumable download |
| `papyrus` | pre-curated, standardised | from Zenodo; a cross-check on our own handling |
| `iuphar` | ~7.5k expert-curated Kd/Ki | best GPCR / ion-channel coverage |
| `uniprot` | accession → sequence | **required**: ChEMBL and IUPHAR both carry accessions and no sequence |
| `rcsb_crystals` | crystal Kd/Ki + geometry | **open substitute for PDBbind**: ~4,500 entries, no registration, written in PDBbind layout |
| `pdbbind` | crystal Kd/Ki + geometry | licensed, never downloaded — stage it with `--pdbbind-tarball` |

Everything lands in `work/data/raw/` with a `MANIFEST.json`. Phase 1b then reads
**only** those local files:

```bash
python scripts/phase1_harmonize.py             # or: sbatch slurm/10_phase1.sbatch
```

It standardizes ligands, resolves sequences, dedupes across sources on InChIKey,
applies the benchmark quarantine, aggregates to one pK per (protein, ligand), and
measures the noise ceiling on its own replicates.

**PDBbind does three separate jobs** and it is worth keeping them apart:

1. Kd/Ki **labels** at source weight 1.0 — the highest of any source, because the
   measurement is tied to a solved structure of the very complex being measured
2. distogram + pocket **geometry** for Stage 1 and the anchor (Section 2.1)
3. the **Stage 3 crystal fine-tune** set — pretrain on the big literature corpus,
   fine-tune on the crystal-linked subset

```bash
PDBBIND=/scratch/$USER/Savant/sprint_oct8/work/data/raw/pdbbind bash slurm/submit_wave1.sh
```

### Wave 1 — foundations and honest baselines

```bash
bash slurm/submit_wave1.sh
```

```
[05 acquire]              <- the only network step; checkpointed, resumable
        |
[10 phase1 harmonise]
        |
  +-----+-----+------------------+
  |           |                  |
[20 mmseqs] [30 ligand phys] [40 embed protein]     <- concurrent
  |           |                  |
  +-----+-----+------------------+
        |
[merge + phase5 splits]
        |
[50 baselines]  <- GATE G1
```

Useful knobs: `PDBBIND=/scratch/$USER/pdbbind_v2020` (adds the geometry array — see
below), `PLM=none|full|bottleneck|pocket_window`, `TIER=A|AB|ABC`,
`N_EMBED`/`N_LIGPHYS`, `DRY=1`, `CONDA_LIBS_FIRST=1`.

**Read the `ligand_only` row of Gate G1 before doing anything else.** If nothing beats
it, that is the result, and Wave 2 does not fix it.

### Wave 2 — the MVP physics model and the ablation matrix

```bash
./slurm/submit_wave2.sh
```

Runs the Phase 7 selftest synchronously first (it blocks on failure), then expands
`configs/preregistration.yaml` into array cells and wires the report `afterany` so a
grid with one dead cell still produces a leaderboard.

```bash
python scripts/phase9_ablation.py --emit          # the matrix, one cell per line
python scripts/phase9_ablation.py --gate          # GATE G2 (Tier 1)
python scripts/report.py                          # reports/final.md
python scripts/status.py                          # where did it stop?
```

### A single configuration, by hand

```bash
python scripts/train.py -o model=variant_a -o plm=bottleneck
python scripts/train.py -o model=variant_c                    # the matched control
python scripts/train.py -o train=stage1_geometry              # geometry only
python scripts/train.py -o model=oracle --split double        # cost-of-pose-free
```

---

## The gates

| Gate | Where | What it asks |
|---|---|---|
| Phase 0 | `phase0_verify.py` | manifest written, quarantine keys frozen, Boltz-2 numbers reproduced |
| Phase 1a | `phase1a_acquire.py --status` | every source acquired; ChEMBL checkpoint complete |
| Phase 1 | `phase1_harmonize.py` | no benchmark leakage, label policy holds, counts vs Section 4.4 |
| Phase 3 | `phase3_ligand_phys.py --merge` | Tier-A failure rate < 2% |
| Phase 4 | `phase4_embed.py --verify` | deterministic re-extraction matches the cache |
| Phase 5 | `phase5_splits.py` | every split delivers the guarantee it advertises |
| **G1** | `phase6_baselines.py` | all baselines exist on all splits, with CIs and the shortcut check |
| Phase 7 | `phase7_selftest.py --overfit` | physics tests pass, oracle equivalence holds, the model overfits a tiny batch |
| **G2** | `phase9_ablation.py --gate` | **Tier 1**: beat the matched control by ΔPearson ≥ 0.02 with a CI excluding zero, and beat ligand_only and target_mean |
| Wave 4 | `phase10_benchmark.py` | the sealed benchmark, **once per model version** |

`scripts/phase10_benchmark.py` refuses to score a model version twice without
`--rescore`, and logs every scoring event. Scoring repeatedly and keeping the best is
selection on the test set, which is what the seal exists to prevent.

---

## Two things that will bite you

**1. Without structural data there is no distogram anchor.** Section 2.1 lists the
supervised distogram cross-entropy as a *required* mitigation, and Section 0.3.4 keeps
it on through every later stage. Without it, the affinity loss discovers that moving
probability mass toward each pair's energy minimum lowers the loss, and the model
converges on an over-attractive, geometrically impossible solution that scores well on
pK and is not physics. `scripts/train.py` refuses to run an affinity stage with the
anchor off unless `--allow-unanchored` is passed, which is recorded in `run.json`, and
`scripts/status.py` says so in plain terms when no geometry is staged.

Stage a licensed PDBbind copy and run:

```bash
PDBBIND=/scratch/$USER/pdbbind_v2020 sbatch --array=0-7 slurm/25_geometry.sbatch
python scripts/phase2_geometry.py --merge
```

**2. Twelve GPUs make it easy to overfit a validation set by volume.** Every run
appends a row to `reports/compute_ledger.csv`; `report.py` states the configuration
count; and anything outside `configs/preregistration.yaml` is labelled *exploratory*. A
ΔPearson of 0.02 selected post hoc from 300 trials is weak evidence dressed as a
result.

---

## Expectations

From PLAN.md Section 0.2, and worth re-reading before interpreting any number:

- No-physics fusion control on strict splits: Pearson ≈ 0.40–0.55, RMSE ≈ 1.5–1.7 pK.
- Physics gain over that control: **+0.00 to +0.05 Pearson**, with a real chance of none.
- Tier 1 roughly even odds; Tier 2 low; Tier 3 very unlikely.
- Experimental noise ceiling ≈ 0.5 log units. `reports/phase_1.md` measures it on this
  corpus's own replicates rather than borrowing the number — an RMSE below the implied
  sd is fitting noise.

If Tiers 2–3 fail, the value case is speed and interpretable decompositions, and
Section 9 requires those to be *measured*. `phase10_benchmark.py` records inference
seconds per complex for exactly that reason.

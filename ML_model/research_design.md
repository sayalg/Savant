# Research Design: Protein–Ligand Binding Affinity Prediction

**Goal:** Build a general scoring function that predicts protein–ligand binding affinity and transfers to *unseen proteins*, improving on `haddocking/prodigy-lig`. Two tracks: (A) a tabular gradient-boosting baseline on physics features, and (B) a 3D interaction-graph model that uses the docked poses.

**Prediction target:** `neg_log10_affinity_M` (pKd). Report ΔG alongside using `ΔG = -1.364 × pKd` (kcal/mol, 298 K). Train on pKd only.

**Data:** ~50,000 rows. One CSV of tabular features (columns described below), available in this environment — used by Track A. The corresponding 3D complex structures are **AlphaFold3-derived, stored on an HPC cluster, and are NOT present in this environment** — used by Track B, which is authored here but executed on the HPC (see §4.0). The path convention for the structures is defined by `pipeline.py`.

**Primary evaluation:** protein-clustered split (see §3). A random split must also be reported, but the clustered split is the number that matters. Beating PRODIGY-LIG *on the clustered split* is the success criterion.

---

## 0. Repository layout to create

```
project/
  data/
    raw/                      # input CSV + pose files (read-only)
    splits/                   # generated split assignment files
    processed/                # cleaned feature tables
  src/
    common/
      splits.py               # clustered + scaffold + random split logic (shared)
      metrics.py              # R2, RMSE, MAE, Spearman, per-target-cluster metrics
      targets.py              # pKd <-> dG conversion helpers
    track_a/
      features.py             # column selection / dropping (see §2)
      train_lgbm.py
      evaluate.py
    track_b/
      paths.py                # ALL structure paths, derived from pipeline.py (cite lines)
      graph_build.py          # complex -> interaction graph
      model.py                # GNN definition
      train_gnn.py
      evaluate.py
  hpc/                        # everything needed to run Track B on the cluster
    env.sh                    # cluster-specific vars (partition/account/modules) — EDIT ME
    prep_download.sh          # login-node: fetch any pretrained weights (no compute-node internet)
    build_graphs.sbatch       # array job, sharded by Job_ID, resumable
    train.sbatch
    evaluate.sbatch
    run_all.sh
    container.def             # Apptainer/Singularity recipe (offline-capable)
    README.md                 # step-by-step HPC runbook
  configs/
    track_a.yaml
    track_b.yaml
  reports/                    # metrics tables + plots written here
  results_bundle/             # <- copied BACK from HPC (Track B outputs only)
  environment.yml
  README.md
```

Both tracks import the SAME `src/common/splits.py` and `src/common/metrics.py` so the comparison is apples-to-apples.

---

## 1. Shared requirements (both tracks)

- Python 3.11. Pin dependencies in `environment.yml`.
- Every training run consumes a split assignment file from `data/splits/` — no script generates its own split inline. This guarantees Track A and Track B are evaluated on identical test complexes.
- All metrics computed by `src/common/metrics.py`: **R², RMSE, MAE, Spearman r**, plus **per-protein-cluster Spearman** (median and IQR across test clusters) to show transfer, not just aggregate fit.
- All reports written to `reports/` as both a CSV and a rendered predicted-vs-actual scatter (colored by protein cluster).
- Set and log a global random seed; make runs reproducible.
- Save the fitted model + the exact feature list used, so inference is unambiguous.

---

## 2. Track A — Tabular gradient-boosting baseline

### 2.1 Objective
A strong, interpretable "improved PRODIGY-LIG": LightGBM regressor on physics + core ligand descriptors. This sets the number Track B must beat.

### 2.2 Column handling — **explicit drop list**

The input CSV has ~2070 columns. `src/track_a/features.py` must **drop the columns that are not valuable for training** and keep only a curated set. Implement this as an explicit allow-list (keep these; drop everything else) and log exactly which columns were dropped and why.

**KEEP — interaction / physics features (highest value; these are the signal):**
`score`, `total`, `vdw`, `elec`, `desolv`, `bsa`, `num_interacting_residues`, `iptm`

**KEEP — core ligand descriptors (secondary; cheap, low-dimensional):**
`molecular_weight`, `logP`, `num_h_donors`, `num_h_acceptors`, `num_rotatable_bonds`, `tpsa`, `num_rings`, `num_aromatic_rings`, `num_atoms`, `num_heavy_atoms`, `formal_charge`

**TARGET:** `neg_log10_affinity_M`

**DROP — and the reasons (log these):**
- **All 2048 `morgan_fp_*` columns.** These encode *ligand identity*. On a protein-clustered split they invite the model to memorize scaffolds rather than learn interaction physics, inflating random-split scores and hurting transfer. Excluded from the baseline by default.
- **`Job_ID`** — identifier, not a feature (retain separately as an index for joining/splitting, never as a model input).
- **`smiles`** — raw string, not a numeric feature; retained only for scaffold-based splitting (§3), never fed to the model.

### 2.3 Ablation (built in, not optional)
Provide a config flag `include_fingerprints: [false|true]`. Run both:
1. physics + ligand descriptors (default),
2. the same **plus** the 2048 fingerprint bits.

Compare on BOTH the random and clustered splits. Expected/what to look for: fingerprints help on random, help much less (or hurt) on clustered. This quantifies the memorization gap and justifies the default drop. Report a feature-importance table (gain) and confirm the physics terms dominate in the default model.

### 2.4 Model
- LightGBM regressor. Early stopping on the validation fold.
- Light hyperparameter search (learning rate, num_leaves, min_child_samples, feature/bagging fraction) via the validation fold only — never touch test.
- No feature scaling needed (trees are scale-invariant), but median-impute any missing physics values and log how many were imputed.

### 2.5 Deliverables
Trained model, feature list, `reports/track_a_metrics.csv` (random + clustered, both ablation arms), feature-importance plot, predicted-vs-actual scatter.

---

## 3. Split strategy (`src/common/splits.py`)

Generate and persist three split files under `data/splits/`:

1. **Random split** — 80/10/10 train/val/test. Baseline / optimistic reference only.
2. **Protein-clustered split (PRIMARY)** —
   - Extract protein sequences per complex (from the pose files / a provided sequence map).
   - Cluster with **MMseqs2 at 30% sequence identity** (`easy-cluster`, `--min-seq-id 0.3`).
   - Assign **whole clusters** to train/val/test (target 80/10/10 by complex count) so no test protein shares a cluster with any train protein.
3. **Scaffold split (secondary check)** — Bemis–Murcko scaffolds from `smiles` (RDKit); assign whole scaffolds to folds to test chemotype generalization.

Also implement a **cross-boundary dedup check**: flag/remove near-identical complexes (same protein cluster AND same ligand scaffold) that straddle the train/test boundary. Log counts.

Split files store `Job_ID -> {train,val,test}` plus the cluster id, so both tracks reuse them exactly.

---

## 4. Track B — 3D interaction-graph model (runs on HPC)

### 4.0 Execution context — READ FIRST

**Track B does not run in this environment.** The docked structures are **AlphaFold3-derived complexes that live on an HPC cluster** and are too large / numerous to move. The workflow is:

1. Claude Code **writes** all Track B code + HPC job scripts here.
2. **The user transfers this code to the HPC and runs it there**, against the structures in place.
3. Only a **compact results bundle** (metrics CSVs, predictions, plots, trained model checkpoint, logs — NOT the structures) is copied back for analysis in this environment.

Therefore Claude Code must **not** attempt to read, glob, or validate the structure files locally — they are not present here. It must instead generate portable code that discovers them at runtime on the HPC.

**Directory layout is defined by `pipeline.py`.** `pipeline.py` (the script that generated all the AF3 outputs) is the single source of truth for where structures live and how they are named. Before writing any Track B code, Claude Code must:
- Read `pipeline.py`, extract the exact output-path convention (root dir, per-`Job_ID` subfolder pattern, filenames for the AF3 model, confidence/PAE JSONs, ligand/protein separation, and how `iptm` was pulled).
- Record that convention in `src/track_b/paths.py` as the ONE place paths are constructed, with a comment citing the lines of `pipeline.py` it was derived from.
- If anything in `pipeline.py` is ambiguous, emit a `TODO(user):` in `paths.py` and list the open questions in the README rather than guessing.

### 4.1 Objective
Use the AF3 poses (not the tabular summary) to beat the Track A ceiling. This is where real physics is learned rather than re-weighted.

### 4.2 HPC portability requirements
- **Scheduler:** assume **SLURM**. Provide `hpc/` with `build_graphs.sbatch`, `train.sbatch`, `evaluate.sbatch`, plus a top-level `run_all.sh`. Put all cluster-specific values (partition, account, walltime, GPU type/count, module loads, conda/venv activation) in a single `hpc/env.sh` sourced by every job, with clearly marked `# EDIT ME` placeholders — do not hard-code them across scripts.
- **Environment:** ship a self-contained `environment.yml` **and** a note on building it offline (many clusters block compute-node internet). List exact package versions; prefer `conda-pack` or a container recipe (`hpc/container.def` for Apptainer/Singularity) so it runs where pip install may be unavailable.
- **No internet assumption at runtime.** Any pretrained weights (e.g. for B2) must be downloaded in a login-node prep step (`hpc/prep_download.sh`), never from inside a compute job.
- **Scale for 50k complexes:** graph building must be an **array job** (`--array`) sharded by `Job_ID`, writing one cache file per shard, resumable (skip shards already built). Training reads the cache, not raw structures.
- **Checkpointing:** training must checkpoint every N steps and support `--resume` so a walltime-killed job restarts cleanly.
- **Everything parameterized via `configs/track_b.yaml`** and overridable by CLI, so no code edits are needed on the cluster.

### 4.3 Graph construction (`src/track_b/graph_build.py`)
For each complex (paths resolved via `src/track_b/paths.py`):
- **Ligand nodes:** ligand atoms; node features = element, hybridization, formal charge, aromaticity, degree.
- **Protein nodes:** atoms (or residues) of the **interacting interface only** — residues within a heavy-atom cutoff of the ligand (config param, default 4.5–6 Å). `num_interacting_residues` from the CSV is a cross-check on interface size.
- **Edges:** covalent bonds (intra-ligand, intra-protein); non-covalent protein–ligand edges within cutoff with distance-based edge features (optional interaction typing: H-bond, hydrophobic, electrostatic).
- **AF3-specific signal:** incorporate AF3 confidence where available — per-atom/residue pLDDT as node features and interface PAE as an edge/global feature (both come out of the AF3 outputs `pipeline.py` produced). These are strong, free quality signals; make their inclusion a config flag.
- **Graph-level features:** concat the Track A physics scalars (`vdw, elec, desolv, bsa, iptm, …`) at readout — cheap and strictly additive.
- Cache built graphs to a cluster scratch path (config; default `data/processed/graphs/`) keyed by `Job_ID`.

### 4.4 Model (`src/track_b/model.py`)
1. **Baseline B1 — interaction GNN:** message passing over the bipartite protein–ligand interaction graph (PIGNet / TankBind-style), mean/attention pooling, concat global physics features, MLP head → pKd. **Required deliverable.**
2. **Optional B2 — pretrained scorer fine-tune** (e.g. GNINA-style CNN grid scorer); weights fetched in the login-node prep step. Stretch.
3. **Optional B3 — E(3)-equivariant net**, only if B1 saturates; poses are AF3 co-folded so quality supports it. Highest effort; config-gated. Stretch.

Deliver **B1 first and fully**; B2/B3 must not block B1.

### 4.5 Training
- Same split files, same metrics, same target (pKd) as Track A. **The split files are generated in this environment (§3) and copied to the HPC** — splitting does NOT happen on the cluster, so Track A and Track B provably share test complexes.
- Loss: MSE on pKd (optionally Huber). Early stopping on val; log curves.
- Emit metrics in the **same format** as Track A.

### 4.6 Results bundle to bring back (define exactly what returns)
Track B writes a single `results_bundle/` directory on the HPC containing ONLY:
- `track_b_metrics.csv` (random + clustered splits, same schema as Track A),
- `predictions.csv` (`Job_ID, split, y_true_pKd, y_pred_pKd`),
- `pred_vs_actual.png`, training-curve plot,
- `best_model.ckpt` + `feature_config.json`,
- `run_log.txt` + resolved config + the git commit / env hash.

This bundle is small (no structures) and is what the user copies back here. Final analysis and the §5 comparison happen in THIS environment by loading `predictions.csv` against the Track A predictions.

---

## 5. Final comparison (write to `reports/comparison.md`)

A single table, evaluated on the **protein-clustered test set**:

| Model | R² | RMSE | MAE | Spearman | Median per-cluster Spearman |
|---|---|---|---|---|---|
| PRODIGY-LIG (reference) | | | | | |
| Track A: physics + ligand desc | | | | | |
| Track A: + fingerprints | | | | | |
| Track B1: interaction GNN | | | | | |

Plus the random-vs-clustered gap for each, as the memorization diagnostic.

---

## 6. Data-integrity checks (run before any training; fail loudly)

1. **Pose provenance consistency:** confirm the poses / `iptm` / interaction energies used in training come from the same generation pipeline that will be used at inference. If train poses are crystal-derived but inference will be docked, transfer will fail regardless of model. Log the provenance assumption explicitly in the README.
2. **Cross-boundary leakage:** run the dedup check from §3 and report how many complexes were removed.
3. **Target sanity:** check `neg_log10_affinity_M` distribution (range, NaNs, duplicates); log summary stats.
4. **Interface presence:** verify each complex actually has interacting residues (drop / flag `num_interacting_residues == 0`).

---

## 7. Order of execution for Claude Code

**Phase 1 — this environment (local):**
1. `environment.yml` + repo skeleton.
2. `src/common/` (targets, metrics, splits) + generate the three split files + run the §6 checks that don't need structures.
3. Track A end-to-end: `features.py` (explicit drop list + logging) → `train_lgbm.py` → `evaluate.py`, both ablation arms, both splits. **This gives a working baseline number before any HPC work.**

**Phase 2 — write Track B for the HPC (local authoring, no execution):**
4. Read `pipeline.py`; derive and record paths in `src/track_b/paths.py` (cite the lines; `TODO(user):` for anything ambiguous).
5. Write `graph_build.py`, `model.py` (B1), `train_gnn.py`, `evaluate.py`, the `hpc/` SLURM scripts + `env.sh` + `run_all.sh`, and the container/offline-env recipe.
6. Write `hpc/README.md`: a step-by-step runbook for the user — how to copy the repo + split files to the cluster, edit `hpc/env.sh`, run the login-node prep, submit the array + training jobs, and retrieve `results_bundle/`.

**Phase 3 — user runs on HPC, then returns:**
7. User executes on the cluster; copies `results_bundle/` back into this environment.
8. Claude Code loads `results_bundle/predictions.csv` + Track A predictions → writes `reports/comparison.md` (§5).

Track A must be complete before Phase 2. Nothing in Phase 2 may read structure files locally — they are not present here.

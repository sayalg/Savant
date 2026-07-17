# Savant BindingDB Kd Benchmark Builder

Builds a **fold-disjoint, ligand-disjoint** external benchmark of protein–ligand
pairs with measured **Kd**, drawn from BindingDB, for testing the Savant affinity
model. A pair is kept only when **both** hold against the training set
(`combined_50k_filtered.csv`):

- **Protein — amino-acid sequence similarity:** no training protein chain matches the
  target at ≥ `min_seq_id` identity over ≥ `coverage` of both sequences
  (`mmseqs easy-search`, cov-mode 0). This mirrors the project's
  `mmseqs easy-cluster --min-seq-id 0.3` methodology.
- **Ligand — SMILES similarity:** the ligand is neither identical (canonical SMILES /
  InChIKey skeleton) nor **ECFP4 Tanimoto ≥ `max_tanimoto`** to any training ligand.

Why BindingDB and not PDBbind: the training set was derived from PDBbind, so it
already contains essentially every PDBbind *protein* target — a fold-disjoint set
cannot be drawn from PDBbind. BindingDB supplies Kd pairs on targets outside that set.

## Layout
```
benchmark_builder/
  environment.yml            # conda env (python, pandas, rdkit, mmseqs2, biopython, ...)
  config.yaml                # ALL paths + thresholds — edit here
  run_benchmark.sbatch       # SLURM submit script (EDIT ME markers)
  scripts/
    common.py
    01_add_sequences.py      # input.csv seq -> combined_50k_filtered.csv (+FASTA of train chains)
    02_download_bindingdb.py # download BindingDB Kd table (login node)
    03_build_benchmark.py    # protein + ligand filters -> stratified benchmark
  work/                      # intermediates (created at runtime)
  output/                    # deliverables (created at runtime)
```

## One-time setup
```bash
conda env create -f environment.yml
conda activate savant-benchmark
```
Confirm the input paths in `config.yaml` point to your copies of `Input/input.csv`
and `ML_model/combined_50k_filtered.csv` (defaults assume the repo layout).

## Run
1. **Download BindingDB on a login node** (compute nodes usually have no internet):
   ```bash
   conda activate savant-benchmark
   python scripts/02_download_bindingdb.py --config config.yaml
   ```
   (If your cluster blocks HTTP entirely, download the ~55 MB Dataverse file
   `4291555` elsewhere and place it at `work/bindingdb_kd.tsv`.)

2. **Edit `run_benchmark.sbatch`** — set `--partition`/`--account` and the conda
   activation lines for your cluster.

3. **Submit** (create `logs/` first — SLURM writes the job log there before the
   script body runs):
   ```bash
   mkdir -p logs
   sbatch run_benchmark.sbatch
   ```
   Or run the three steps by hand (steps 01 + 03 need no internet).

## Output
- `output/combined_50k_filtered_with_seq.csv` — training table with a
  `protein_sequence` column joined from `input.csv`.
- `output/bindingdb_kd_benchmark_<N>.csv` — the benchmark. Columns:
  `uniprot_id, ligand_smiles, target_sequence, kd_nM, neg_log10_affinity_M`
  (= pKd, the training target name), `deltaG_kcal_per_mol`,
  `max_train_protein_identity`, `max_train_ligand_tanimoto`.
- `output/bindingdb_kd_benchmark_<N>_summary.md` — QC summary (ranges + that both
  overlap metrics sit below their thresholds).

## Tuning (`config.yaml`)
- `filters.min_seq_id` / `filters.coverage` — protein disjointness strictness.
- `filters.max_tanimoto` — ligand similarity cutoff: 0.5 = same-chemotype boundary;
  raise to ~0.7 for near-duplicates only, lower to ~0.4 to be stricter.
- `filters.protein_method` — `mmseqs` (fast) or `biopython` (portable, slow).
- `selection.n_pairs`, `selection.max_per_uniprot`, `selection.pkd_bins`.

## Notes
- `input.csv` is streamed in chunks (only rows whose `ID` is a training `Job_ID` are
  kept), so step 01 is memory-light despite the ~1.3 GB file.
- BindingDB Kd values pinned at round caps (10/100/1000/10000 µM) are collapsed
  ">X" censored measurements and are dropped; only precise Kd values are used.
- The Kd→pKd conversion is `pKd = 9 − log10(Kd[nM])`; ΔG = `−1.364 × pKd` (298 K),
  per `research_design.md`.

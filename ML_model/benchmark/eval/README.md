# Scoring the benchmark with Boltz-2 -> Savant + PRODIGY-LIG

Runs three predictors over the 1000 pairs in `../output/savant_benchmark_1000.csv`
and writes every score back onto the benchmark rows.

Boltz-2 goes first because it is the only one of the three that works from
sequence + SMILES. Savant and PRODIGY-LIG both need a 3D complex, and they read
the structure Boltz produced.

```
benchmark CSV ──▶ Boltz-2 ──▶ complex PDB ──┬──▶ Savant  (RDKit + HADDOCK3 → pK)
 (sequence,        (GPU)      chain A / B   └──▶ PRODIGY-LIG (contacts → ΔG)
  SMILES)             │
                      └──────────────────────────▶ its own affinity + confidence
```

## The chain contract

Savant is invoked as `-p A -l B` and PRODIGY-LIG as `-c A B:LIG`. Both therefore
need the protein and the ligand on **separate chains**, and PRODIGY additionally
needs a 3-character ligand residue name appearing exactly once in its chain.

That is arranged at the source: every YAML asks Boltz for `protein: id A` and
`ligand: id B`, Boltz uses those ids verbatim, and its PDB writer emits every
non-polymer chain as `HETATM` with residue name `LIG`. So a prediction comes out
already split the way both tools need it.

`03_collect_boltz.py` re-checks it on the actual files rather than trusting it —
chain A is all `ATOM` and matches the expected residue count, chain B is all
`HETATM`/`LIG` with one residue and the expected heavy-atom count, no other
chains. A mis-chained model would not crash the downstream tools; it would
quietly score the wrong thing (Savant handing HADDOCK a receptor with no ligand,
PRODIGY aborting on a missing residue), which is exactly the failure worth
catching early.

Two things the same step fixes up:

- **Ligand `CONECT` records.** Boltz writes ligand coordinates with no bond
  records. HADDOCK3's `autotoppar` then has to perceive ligand topology from
  geometry alone. RDKit derives the connectivity from the same coordinates and it
  gets written into the canonical PDB.
- **Chain relabelling.** If a Boltz release ever writes the ids differently, an
  unambiguous single-`ATOM`-chain / single-`HETATM`-chain file is relabelled to
  A/B and flagged as `chains_remapped` in the output. Anything more tangled is
  reported as a failure instead of guessed at.

## Run it

Everything is resume-safe: each stage skips work that is already on disk, so a
timed-out or requeued job is fixed by re-submitting the same script.

```bash
cd ML_model/benchmark/eval
bash check_env.sh                        # boltz, savant, haddock3, prodigy_lig, GPU, inputs
```

**1. Inputs** (minutes, CPU — run it on the cluster so the MSA paths it writes
into the YAMLs are the paths the compute nodes will see):

```bash
python scripts/01_prepare_inputs.py --num-shards 25
```

**2. MSAs** (~1 h, needs outbound internet — a login node is fine):

```bash
python scripts/02_fetch_msas.py
```

355 targets, one query each, cached as `work/msa/<uniprot>.a3m`. This exists so
the 1000 pairs do not issue 1000 queries for 355 distinct answers. If your site
blocks the ColabFold server entirely, run step 01 with `--msa-mode server` and set
`BOLTZ_EXTRA_ARGS="--use_msa_server"` in the Boltz job instead.

**3. Boltz-2** (the long pole — one GPU-hour or so per few pairs):

```bash
sbatch --array=0-24 sbatch/run_boltz.sbatch
python scripts/03_collect_boltz.py
```

Targets run 86–1951 aa (68 pairs are over 1200), so the tail of the distribution
is what sets the memory and walltime you need — if tasks OOM, raise `--mem`,
ask for the larger GPU, and re-submit only the affected shards; completed pairs
are skipped.

`--array` must be `0-(num_shards-1)` from step 1. `03` writes
`work/structures/<id>.pdb` plus `work/boltz_scores.csv` (confidence + affinity),
and prints anything that failed the chain contract; re-run the affected shards
and re-run `03`.

**4. Savant** (CPU, the slowest per pair — HADDOCK3 topology + minimisation):

```bash
sbatch sbatch/run_savant.sbatch          # --array=0-19 <-> NUM_SHARDS=20
```

**5. PRODIGY-LIG** (seconds per pair, one serial job) **and the merge**:

```bash
sbatch sbatch/run_prodigy.sbatch
python scripts/06_merge_scores.py
```

Running PRODIGY after Savant is deliberate: PRODIGY-LIG's better-performing
published model takes a HADDOCK electrostatics term (`-e`), and Savant has
already computed one for the same complex. When the Savant JSON exists, PRODIGY
runs twice and both numbers are reported. The `elec` fed in comes from Savant's
`emref` stage rather than a full water refinement, so `prodigy_pkd_elec` is close
to, but not exactly, the quantity the published model was fitted on — hence a
separate column rather than a replacement.

## Output

`output/savant_benchmark_1000_scored.csv` — the benchmark, unchanged, plus:

| column | from | meaning |
|---|---|---|
| `boltz_status`, `structure_path` | 03 | did the pair get a usable complex, and where |
| `boltz_confidence_score`, `boltz_ptm`, `boltz_iptm`, `boltz_ligand_iptm`, `boltz_complex_plddt` | Boltz | pose confidence — use `ligand_iptm` to filter poses before believing any downstream score |
| `boltz_affinity_pred_value`, `boltz_affinity_probability_binary` | Boltz | raw affinity head: log10(IC50 [µM]), and P(binder) |
| `boltz_affinity_pic50` | derived | `6 − affinity_pred_value` |
| `savant_classification`, `savant_p_binder`, `savant_pk` | Savant | binder/nonbinder call, and pK for binders only |
| `haddock_score`, `haddock_total`, `haddock_vdw`, `haddock_elec`, `haddock_desolv`, `haddock_bsa`, `haddock_num_interacting_residues` | Savant | the 7 pose-dependent features behind that prediction |
| `prodigy_dg_kcal_per_mol`, `prodigy_dgscore`, `prodigy_pkd` | PRODIGY | contacts-only ΔG, and pKd = −ΔG/1.364 |
| `prodigy_dg_elec_kcal_per_mol`, `prodigy_dgscore_elec`, `prodigy_pkd_elec` | PRODIGY | same with the HADDOCK electrostatics term |
| `docked_smiles`, `was_desalted` | 01 | what actually entered the structure |

`output/savant_benchmark_1000_scored_summary.md` reports coverage per stage,
agreement with the measured pKd, and the same broken out by `novelty_tier`.
`output/failures.csv` collates every per-pair failure.

Pairs a tool failed on keep NaN rather than being dropped, so coverage stays
visible instead of silently biasing the comparison.

## Things worth knowing before reading the numbers

- **The scales are not identical.** The label is pKd from BindingDB. Boltz-2's
  affinity head is trained on IC50, so `boltz_affinity_pic50` is comparable in
  rank and roughly in magnitude but is not the same measurement. Savant's
  regressor was trained on this exact target. PRODIGY predicts ΔG, converted at
  298 K. Correlation is the honest comparison; RMSE only means something for the
  two that are genuinely on the pKd scale.
- **37 ligands exceed Boltz-2's 128-heavy-atom affinity limit** (before
  desalting). They still get a structure — which is all Savant and PRODIGY need —
  but no affinity value, and `affinity_requested` is False for them.
- **14 ligands are salts.** They are desalted to the largest organic fragment
  before docking, because nine sodium ions in chain B would break PRODIGY's
  one-ligand-residue rule and distort the HADDOCK energetics. `docked_smiles` is
  what was actually used, and it is also what gets passed to Savant's `-s` so its
  descriptors match its coordinates.
- **Every pair is a true binder** (all have a measured Kd), so Savant's
  classifier can only lose here — anything it calls `nonbinder` gets no pK and
  shows up as a coverage gap, not as a wrong number.
- **Boltz poses are predictions, not crystal structures.** Savant and PRODIGY are
  scoring functions evaluated on a modelled pose; a bad pose gives a bad score
  through no fault of the scorer. `boltz_ligand_iptm` is the handle for
  separating those two failure modes.

## Layout

```
eval/
├── check_env.sh
├── scripts/
│   ├── common.py               paths, the chain contract, shard helpers
│   ├── 01_prepare_inputs.py    CSV      -> work/yaml/shard_NN/<id>.yaml + manifest
│   ├── 02_fetch_msas.py        targets  -> work/msa/<uniprot>.a3m
│   ├── 03_collect_boltz.py     boltz    -> work/structures/<id>.pdb + boltz_scores.csv
│   ├── 04_run_savant.py        structs  -> work/results/savant/<id>.json
│   ├── 05_run_prodigy.py       structs  -> work/results/prodigy/<id>.json
│   └── 06_merge_scores.py      all      -> output/savant_benchmark_1000_scored.csv
└── sbatch/
    ├── run_boltz.sbatch        GPU array, one task per YAML shard
    ├── run_savant.sbatch       CPU array, round-robin over pairs
    └── run_prodigy.sbatch      one serial task
```

`work/` holds everything regenerable. `work/msa/` is the one part worth keeping:
it is the only piece that depends on an external service being up.

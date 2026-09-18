# Scoring the benchmark: Boltz-2, Savant, PRODIGY-LIG, PLAPT

Runs four predictors over the 500 pairs in `../output/savant_benchmark.csv`
and writes every score back onto the benchmark rows.

Boltz-2 goes first because PRODIGY-LIG needs a 3-D complex and reads the
structure Boltz produced. The structure-based model reads it too, for the
HADDOCK3 energy terms.

```
benchmark CSV ──▶ Boltz-2 ──▶ complex PDB ──┬──▶ structure-based  (RDKit + HADDOCK3 → pK)
 (sequence,        (GPU)      chain A / B   └──▶ PRODIGY-LIG     (contacts → ΔG)
  SMILES)  │          │
           │          └──────────────────────────▶ its own affinity + confidence
           ├──────────────────────────────────────▶ savant          (seq + SMILES, CPU)
           └──────────────────────────────────────▶ PLAPT           (seq + SMILES)
```

**Two Savant generations appear in the output, and they are not the same model.**
`04_run_savant.py` runs `savant_structure_based/savant_structure_based.py` on the
Boltz pose and writes `savant_pk` plus the seven `haddock_*` columns. The **model
of record** is the `savant` package, run from sequence and SMILES alone by
`score_savant_benchmark.ipynb`, which writes `savant_esm650_pk`. Every number in
the paper is the `esm650` column.

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
# both already default to these; export them only for a different benchmark
export BENCHMARK_CSV=$PWD/../output/savant_benchmark.csv
export EVAL_WORK=$PWD/work
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

One query per distinct target, cached as `work/msa/<uniprot>.a3m`. This exists so
the 500 pairs do not issue 500 queries for 20 distinct answers. If your site
blocks the ColabFold server entirely, run step 01 with `--msa-mode server` and set
`BOLTZ_EXTRA_ARGS="--use_msa_server"` in the Boltz job instead.

**3. Boltz-2** (the long pole — one GPU-hour or so per few pairs):

```bash
sbatch --array=0-25 sbatch/run_boltz.sbatch
python scripts/03_collect_boltz.py
```

Use the `--array` range step 1 printed, not `--num-shards`. The two differ
because **a shard cannot mix affinity and non-affinity records**: Boltz enters
its affinity stage if *any* record in the input directory requests affinity,
then iterates *every* record in that directory and loads a
`pre_affinity_<id>.npz` that only affinity records are ever given. One
oversized ligand among 39 normal ones kills the whole task partway through with
a `FileNotFoundError`. Step 1 therefore puts the 37 ligands above the
128-heavy-atom limit in their own shard, where `any(record.affinity)` is False
and the affinity stage is skipped entirely; `work/shards.csv` records the split
and the job re-checks it before spending the allocation.

Targets run 86–1951 aa (68 pairs are over 1200), so the tail of the distribution
is what sets the memory and walltime you need — if tasks OOM, raise `--mem`,
ask for the larger GPU, and re-submit only the affected shards; completed pairs
are skipped.

The job prunes unresumable prediction directories before it starts, via
`scripts/prune_partial_boltz.py`. That is not optional bookkeeping: Boltz skips a
record whose `predictions/<id>/` directory merely *exists*, but queues its
affinity unless `affinity_<id>.json` exists. A directory left half-written by a
killed run therefore makes the affinity stage look for a `pre_affinity_<id>.npz`
that the skipped structure stage never wrote, and that `FileNotFoundError` takes
down the whole array task. The pruner deletes exactly those directories and
leaves finished work alone; run it by hand with `--dry-run` to see what it would
remove.

`--array` must be `0-(num_shards-1)` from step 1. `03` writes
`work/structures/<id>.pdb` plus `work/boltz_scores.csv` (confidence + affinity),
and prints anything that failed the chain contract; re-run the affected shards
and re-run `03`.

**4. The structure-based model** (CPU, the slowest per pair — HADDOCK3 topology
and minimisation):

```bash
sbatch sbatch/run_savant.sbatch          # --array=0-19 <-> NUM_SHARDS=20
```

**5. PRODIGY-LIG** (seconds per pair, one serial job), **PLAPT**, **and the merge**:

```bash
sbatch sbatch/run_prodigy.sbatch
sbatch sbatch/run_plapt.sbatch
python scripts/06_merge_scores.py
```

Running PRODIGY after Savant is deliberate — see below.

**6. Scoring and the paper's tables** (CPU, minutes). The notebook runs
`savant` from sequence and SMILES, assembles every method into one frame, and
writes the `bench_*` files that `report/make_figures.py` and `Article.tex` read:

```bash
jupyter nbconvert --to notebook --execute score_savant_benchmark.ipynb \
    --ExecutePreprocessor.timeout=-1
python scripts/10_steiger.py             # the paired within-target test
python scripts/08_contamination_audit.py # the disjointness audit
```

PRODIGY-LIG's better-performing published model takes a HADDOCK electrostatics
term (`-e`), and step 4 has already computed one for the same complex — so
PRODIGY runs after it. When that JSON exists, PRODIGY runs twice and both
numbers are reported. The `elec` fed in comes from the `emref` stage rather than
a full water refinement, so `prodigy_pkd_elec` is close to, but not exactly, the
quantity the published model was fitted on — hence a separate column rather than
a replacement. The paper reports the contacts-only `prodigy_pkd`.

## Output

`output/savant_benchmark_scored.csv` — the benchmark, unchanged, plus:

| column | from | meaning |
|---|---|---|
| `boltz_status`, `structure_path` | 03 | did the pair get a usable complex, and where |
| `boltz_confidence_score`, `boltz_ptm`, `boltz_iptm`, `boltz_ligand_iptm`, `boltz_complex_plddt` | Boltz | pose confidence — use `ligand_iptm` to filter poses before believing any downstream score |
| `boltz_affinity_pred_value`, `boltz_affinity_probability_binary` | Boltz | raw affinity head: log10(IC50 [µM]), and P(binder) |
| `boltz_affinity_pic50` | derived | `6 − affinity_pred_value` |
| `savant_classification`, `savant_p_binder`, `savant_pk` | 04 | binder/nonbinder call, and pK for binders only |
| `haddock_score`, `haddock_total`, `haddock_vdw`, `haddock_elec`, `haddock_desolv`, `haddock_bsa`, `haddock_num_interacting_residues` | 04 | the 7 pose-dependent features behind that prediction |
| `prodigy_dg_kcal_per_mol`, `prodigy_dgscore`, `prodigy_pkd` | PRODIGY | contacts-only ΔG, and pKd = −ΔG/1.364 |
| `prodigy_dg_elec_kcal_per_mol`, `prodigy_dgscore_elec`, `prodigy_pkd_elec` | PRODIGY | same with the HADDOCK electrostatics term |
| `plapt_pk`, `plapt_affinity_uM` | 09 | PLAPT, from sequence + SMILES |
| `docked_smiles`, `was_desalted` | 01 | what actually entered the structure |

`output/savant_benchmark_scored_summary.md` reports coverage per stage and
agreement with the measured pKd. `output/failures.csv` collates every per-pair
failure.

Pairs a tool failed on keep NaN rather than being dropped, so coverage stays
visible instead of silently biasing the comparison. Boltz-2 and PRODIGY-LIG
return nothing for 4 of the 500 pairs, which is why each paired comparison in the
paper is computed on its own intersection and Savant's own within-target *r*
differs slightly between rows of `bench_steiger.csv`.

`score_savant_benchmark.ipynb` then adds the model-of-record columns
(`savant_esm650_pk`, and the `esm8` / `v2` variants) and writes:

| file | what the paper uses it for |
|---|---|
| `bench_scored.csv` | every method on every pair — the single source for the tables below |
| `bench_A_pooled.csv` | pooled *r*, ρ, RMSE, spread ratio (`tab:pooled`, `fig:spread`) |
| `bench_A_decomposition.csv` | between- vs. within-target *r* (`tab:decomp`, `fig:decomp`) |
| `bench_B_summary.csv`, `bench_B_per_target.csv`, `bench_B_vs_ref.csv` | within-target ρ and the cluster bootstrap over targets (`tab:testb`, `fig:within`) |
| `bench_runtime.csv` | measured CPU timings (`tab:runtime`) |
| `bench_steiger.csv` | the paired within-target test, from `scripts/10_steiger.py` (`tab:steiger`) |
| `contamination_flags.csv` | per-ligand disjointness audit, from `scripts/08_contamination_audit.py` |

The notebook also computes stratified breakdowns by protein family, ligand
chemotype and ligand size, and per-target QSAR / Tanimoto k-NN baselines. **None
of those are reported in the paper** — the baselines were removed from it on
2026-09-17. They are left in the notebook so the reported cells run unmodified.

## Things worth knowing before reading the numbers

- **No method beats a constant on RMSE.** Predicting the benchmark mean (7.296)
  for every pair gives RMSE 1.273; the best method tested reaches 1.353. This is
  the label, not the models: replicate measurements of the same pair disagree by
  0.81 pKd, which caps achievable *R²* on this set near 0.60. Read every error
  figure against that floor, not against zero, and report no absolute-error claim
  from this benchmark for any method.
- **The scales are not identical.** The label is pKd from ChEMBL. Boltz-2's
  affinity head is trained on IC50, so `boltz_affinity_pic50` is comparable in
  rank and roughly in magnitude but is not the same measurement. Savant's
  regressor was trained on this exact target. PRODIGY predicts ΔG, converted at
  298 K. Correlation is the honest comparison.
- **Pooled correlation and within-target correlation rank the methods in nearly
  opposite orders.** Boltz-2 leads pooled (0.412 vs. 0.227); Savant leads
  within-target (0.436 vs. 0.386), on the same 500 pairs. The difference is how
  each metric weights the between- and within-target components, and that
  weighting is a property of the benchmark. Report both.
- **Ligands exceeding Boltz-2's 128-heavy-atom affinity limit** still get a
  structure — which is all PRODIGY and the structure-based model need — but no
  affinity value,
  and `affinity_requested` is False for them.
- **Salts are desalted** to the largest organic fragment before docking, because
  counter-ions in chain B would break PRODIGY's one-ligand-residue rule and
  distort the HADDOCK energetics. `docked_smiles` is what was actually used, and
  it is what gets passed to Savant's `-s` so its descriptors match its
  coordinates.
- **Every pair is a true binder** (all have a measured Kd or Ki), so a classifier
  can only lose here — anything called `nonbinder` gets no pK and shows up as a
  coverage gap, not as a wrong number.
- **Boltz poses are predictions, not crystal structures.** PRODIGY and the
  structure-based model are scoring functions evaluated on a modelled pose; a bad
  pose gives a bad score through no fault of the scorer. `boltz_ligand_iptm` is
  the handle for separating those two failure modes. The `savant` package
  touches no structure at all, so it is unaffected.

## Layout

```
eval/
├── check_env.sh
├── score_savant_benchmark.ipynb    savant + all methods -> output/bench_*
├── scripts/
│   ├── common.py                   paths, the chain contract, shard helpers
│   ├── fetch_chembl.sh             ChEMBL 37 sqlite
│   ├── fetch_chembl34.sh           ChEMBL 34 sqlite, for the temporal gate
│   ├── 01_prepare_inputs.py        CSV      -> work/yaml/shard_NN/<id>.yaml + manifest
│   ├── 02_fetch_msas.py            targets  -> work/msa/<uniprot>.a3m
│   ├── 03_collect_boltz.py         boltz    -> work/structures/<id>.pdb + boltz_scores.csv
│   ├── 04_run_savant.py            structs  -> work/results/savant/<id>.json   (structure-based)
│   ├── 05_run_prodigy.py           structs  -> work/results/prodigy/<id>.json
│   ├── 06_merge_scores.py          all      -> output/savant_benchmark_scored.csv
│   ├── 08_contamination_audit.py   scored   -> output/contamination_*
│   ├── 09_run_plapt.py             CSV      -> plapt_pk
│   ├── 10_steiger.py               scored   -> output/bench_steiger.csv
│   └── prune_partial_boltz.py      removes unresumable Boltz prediction dirs
└── sbatch/
    ├── run_boltz.sbatch            GPU array, one task per YAML shard
    ├── run_savant.sbatch           CPU array, round-robin over pairs
    ├── run_prodigy.sbatch          one serial task
    └── run_plapt.sbatch            one serial task
```

`work/` holds everything regenerable. `work/msa/` is the one part worth keeping:
it is the only piece that depends on an external service being up.

# Savant v2 — CPU-only affinity prediction from a docked complex

A rebuild of Savant as a fast, dependency-light, interpretable binding predictor
that runs on any docked complex — an AlphaFold Server prediction, a Boltz
output, a Vina pose, a crystal structure — in about a second on one CPU core.

Nothing outside this directory is modified. v1 stays exactly as it is.

## Why the rebuild

Measured on the 1000-pair benchmark, training on all 47,619 complexes and
testing on the ligand-disjoint benchmark:

| model | Pearson r |
|---|---|
| Savant v1 as shipped | +0.156 |
| v1 features, retrained | +0.233 |
| + ECFP4 ligand fingerprint | +0.272 |
| **ECFP4 alone — no protein, no pose, no HADDOCK** | **+0.237** |
| PRODIGY-LIG | +0.216 |
| Boltz-2 | +0.484 |

Read rows 4 and 2 together: **a model that never sees the protein or the docked
pose matched v1's entire feature set.** Hours of AF3 folding and HADDOCK
minimisation per complex were contributing nothing measurable.

The ablation that follows explains why, and sets the architecture:

| feature set (all with ECFP4) | r |
|---|---|
| ligand only | +0.247 |
| + contact count (pure distance) | +0.259 |
| + `elec` + `desolv` (needs HADDOCK) | +0.258 |
| all 7 HADDOCK terms | +0.271 |

Seven HADDOCK energies buy **+0.012** over a distance-based contact count. So
v2 drops HADDOCK entirely — losing the slow step, the Linux-only dependency, and
the reason a bench scientist could not run this — and spends the effort instead
on describing the interface rather than summing it.

## What v2 adds

**Interactions enumerated, not summed.** An interface is hundreds of atom–atom
contacts; v1 compressed it to six energies. v2 emits a typed fingerprint:
hydrophobic contacts, hydrogen bonds split by ligand role, ionic pairs,
π-stacking split into face-to-face and edge-to-face by ring-normal angle,
cation–π, halogen bonds, metal coordination, contacts broken out by residue
type, and element-pair counts at two distance shells. Geometry criteria follow
PLIP/ProLIF, typed by element and residue so no hydrogens are needed — the
structures coming out of AlphaFold and Boltz are heavy-atom only.

**The protein finally enters the feature vector.** v1's training table had no
protein column *at all* — only ligand descriptors and interaction energies — so
two different targets with similar contact energetics were literally the same
input. Per-residue-type contacts and pocket composition fix that without needing
a protein identifier.

**Burial replaces the energy terms.** A self-contained Shrake–Rupley gives
buried surface split apolar/polar, which is the desolvation trade-off in a form
a model can use. Validated against HADDOCK on a real complex: **850.9 Å² vs
830.6 Å², a 2.4% difference**, computed in 0.3 s instead of minutes.

**Calibrated uncertainty.** Split-conformal intervals with a locally adaptive
width, so a prediction arrives as `pKd 7.15 [6.4–7.9]` with an
applicability-domain flag rather than a bare point estimate.

**A gate that flags instead of discarding.** The binder/nonbinder pre-filter is
retained, but v1 took the classifier's argmax and rejected 288 of 935 benchmark
pairs — all of them measured binders — because it trained on 94% binders and
then used a 0.5 threshold. v2 tunes the threshold to a target recall on a
held-out split, and reports what it screens out instead of silently dropping it.
Measured on the benchmark with v1's own features: **coverage rises from 647 to
846 of 935 (69.2% → 90.5%) while r rises from 0.156 to 0.255.**

## Models: two files, two stages

v1's flow is kept exactly — one model decides binder vs nonbinder, a second
predicts pKd for what passes — and both are written as `.pkl`:

```
models/classifier.pkl            gradient-boosted, binder gate
models/regressor.pkl             gradient-boosted, pKd + conformal interval
models/attention_classifier.pkl  cross-attention, binder gate
models/attention_regressor.pkl   cross-attention, pKd
```

The two families train on the same labels and the same protein-grouped splits,
so they are directly comparable. Every result is reported with **RMSE, MAE,
MAPE, Pearson r and Spearman rho** (`savant2/metrics.py`). MAPE is computed on
pKd, never on Kd — Kd spans twelve orders of magnitude, so percentage error on
it is meaningless.

Measured with v1's own 18 features, held out by protein:

```
    n  rmse   mae   mape  pearson_r  spearman_rho
10743 1.411 1.141 19.642      0.435         0.385

classifier: binder recall 0.950, precision 0.963, ROC-AUC 0.864
conformal:  nominal 90% -> empirical 90.0%, median width 3.41 log units
```

## Cross-attention over pocket and ligand

`savant2/attention.py` is the second model family, and the part that is
genuinely new rather than a better-engineered version of v1:

```
ligand atoms    (M, 16)   RDKit typing: element, aromaticity, charge, degree
pocket residues (K, 640)  ESM-2 per-residue states, pocket only
attention bias  -d/tau    real interatomic distance, per-head learned scale
```

Every ligand atom attends to every pocket residue, biased by how far apart they
actually are. Two things this buys over the gradient-boosted models:

**Cooperativity.** "This carboxylate against that arginine, given that ring is
stacked over there" is a product of interactions. A bag of counts can only sum
them.

**The attention matrix is the explanation** — one weight per (ligand atom,
pocket residue) pair, readable as a binding map and checkable against known
pharmacophores. That is a validation you can run, not a vanity plot, and neither
Boltz-2 nor PRODIGY-LIG can produce it.

ESM-2 is scoped to the pocket, not the whole chain, for a measured reason:
whole-protein amino-acid composition was worth +0.012 Pearson r on this
benchmark. Binding happens in one small site, and a ~48-residue pocket also
keeps the attention sequence short enough to train fast and read residue by
residue. Embeddings are cached per sequence, so the 47,619 complexes cost only
11,965 ESM passes.

## Set up the environment

```bash
cd savant_v2
conda env create -f environment.yml
conda activate savant2
python -c "import savant2, rdkit, sklearn; print('savant2 ready')"
```

That gives the CPU build of PyTorch, which is all the geometric path, both
gradient-boosted models and the CLI need. **For the ESM stage on a GPU node,
install the PyTorch build that matches the node's driver** — not whatever PyPI
defaults to, which is how the Boltz run lost an afternoon:

```bash
nvidia-smi                                   # read the CUDA version in the header
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.4
# or, with pip:
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then confirm the GPU is actually usable before submitting anything:

```bash
python -c "import torch; torch.zeros(1, device='cuda').add_(1); print(torch.cuda.get_device_name(0))"
```

`requirements.txt` covers the same set for pip-only environments. Note what is
*absent*: no `haddock3`, no `biopython`, no `click`. v2 needs numpy, scipy,
pandas, pyarrow, RDKit, scikit-learn and joblib — plus torch and transformers
only for the cross-attention model.

## Run it

```bash
python scripts/00_build_manifest.py --source training --structure-dir /path/to/docked
sbatch --array=0-19 sbatch/extract_features.sbatch work/manifest_training.csv
python scripts/02_train.py
```

Then the benchmark:

```bash
python scripts/00_build_manifest.py --source benchmark
sbatch --array=0-19 sbatch/extract_features.sbatch work/manifest_benchmark.csv
python scripts/03_score_benchmark.py
```

The cross-attention family, which needs a GPU for the ESM pass:

```bash
sbatch --array=0-7 sbatch/build_attention.sbatch work/manifest_training.csv
python scripts/05_train_attention.py --examples work/attention_training

# then score the benchmark with them, the same way step 03 does for the trees
sbatch --array=0-3 sbatch/build_attention.sbatch work/manifest_benchmark.csv
python scripts/06_score_benchmark_attention.py
```

The shard count is taken from `SLURM_ARRAY_TASK_COUNT`, so `--array` alone
decides how the manifest is divided and the two cannot disagree. Setting
`NUM_SHARDS` by hand is what silently processed 4/20ths of the benchmark once.

Point `--structure-dir` at the AF3 predictions and `--pattern` at their layout:

```bash
python scripts/00_build_manifest.py --source training \
    --structure-dir /scratch/sguirale/<run>/alphafold3/predictions \
    --pattern '{id}/{id}_model.pdb'
```

## Steps 7-8: the double-disjoint split experiment

The benchmark run produced a result the internal split could not have predicted:

| | protein-grouped held-out | external benchmark |
|---|---|---|
| GBDT regressor | r 0.580 | r 0.327 |
| attention regressor | r 0.598 | r 0.269 |

Internally the two are tied, with the transformer marginally ahead. Externally
the trees win decisively. **The internal split did not merely flatter both
models - it ranked them the wrong way round**, so any selection based on it
would have shipped the worse one.

The cause is measurable. Grouping by protein alone leaves the ligand axis free:
on the 47,619 training complexes that split shares **2,298 Bemis-Murcko
scaffolds** across the fold boundary, while the benchmark gates ligands at ECFP4
Tanimoto <= 0.50 against everything in training. A model can meet a new protein
in the held-out fold while scoring a ligand whose near-twin it trained on.

Steps 7 and 8 reproduce the benchmark's conditions internally, holding out
proteins **and** scaffolds so neither axis leaks.

```bash
python scripts/07_split_experiment.py                       # both GBDT stages

python scripts/05_train_attention.py \
    --examples work/attention_training \
    --split-mode protein_scaffold \
    --out-dir models/disjoint
```

Step 7 trains the gradient-boosted classifier and regressor under both splits
and prints them side by side; step 8 is step 5 again with `--split-mode
protein_scaffold`. Models from the disjoint split land in `models/disjoint/`,
trained on the training fold only so they can be scored against the benchmark
without having seen their own test rows.

**Two things the split costs, both reported.** Rows that straddle the held-out
sets - new protein but familiar scaffold, or the reverse - are discarded rather
than assigned to either side, because keeping them reintroduces the leak being
measured. That is ~44% of the data, and training shrinks from 36k to 22k
complexes. Step 7 therefore also runs a **size-matched control**: the
protein-only split with its training fold subsampled to the same 22k, so the
remaining difference is attributable to the split rather than to less data.

**How to read the result.** If the double-disjoint numbers land near the
benchmark's 0.327 and 0.269, the split is validated as the one that predicts
deployment behaviour - a methods contribution independent of Savant, and the
evidence that the transformer's apparent parity was leakage. If they stay well
above the benchmark, some other gap remains, and pose source (AF3 training vs
Boltz benchmark) and assay heterogeneity are the candidates to examine next.

And a single complex:

```bash
python -m savant2.cli complex.pdb -s "CCCCCCCCCCCCCCCCCCCC(=O)O"
```

```
  dock_1_model.pdb   (0.342s)

  prediction   BINDER   (P(binder) = 1.00)
  affinity     pKd 7.15  [4.40 - 9.89]   confidence: low
               Kd  71.2 nM  [0.1 - 39,652.7]

  interface    18 residues, 851 A^2 buried, ligand 99% enclosed

  interactions
    hydrophobic contacts                134
    hydrogen bonds                        4
      ligand donates                      2
      ligand accepts                      2
    salt bridges (ligand anion)           6

  contacting residues  TYR x4, ILE x3, VAL x3, ALA x1, GLU x1, PHE x1, LYS x1, LEU x1
```

That is arachidic acid in a fatty-acid binding protein: fully enclosed, packed
by hydrophobic contacts along the chain, carboxylate anchored by salt bridges.
No other tool in the comparison set reports that — Boltz-2 returns a scalar,
PRODIGY-LIG returns contact counts with no chemistry, Vina returns a score.

## Layout

```
savant_v2/
├── savant2/
│   ├── structure.py     PDB -> numpy arrays, pocket selection
│   ├── ligand.py        descriptors, ECFP4, per-atom roles via SMILES template
│   ├── interactions.py  the typed interaction fingerprint  <- the core
│   ├── burial.py        Shrake-Rupley SASA and buried surface
│   ├── features.py      assembles the 650-feature vector
│   ├── conformal.py     split-conformal intervals
│   ├── models.py        gate + pK + sigma bundle
│   ├── pipeline.py      score one complex
│   └── cli.py           savant2 command line
├── scripts/  00 manifest -> 01 extract -> 02 train -> 03 benchmark
└── sbatch/   extract_features.sbatch
```

650 features: 11 ligand descriptors, 138 structural, 512 ECFP4 bits.

## Attention path: validated

Exercised end to end on real complexes with torch 2.12 + transformers 5.15 and
ESM-2 8M, 48 examples over 8 protein groups:

```
pocket extraction   48 residues, nearest 2.48 A
ESM embedding       (48, 320) float16, cached re-read in 7 ms
tensors             ligand (22,16)  pocket (48,320)  distance (22,48)
training            classifier + regressor, both stages, loss 6.47 -> 2.29
attention           (batch, 4 heads, 22 ligand atoms, 48 pocket residues)
.pkl round-trip     deterministic on reload
```

The distance bias behaves as designed: **corr(attention, distance) = -0.903**,
and residues within 5 A receive 20x the weight of residues beyond 15 A
(0.0593 vs 0.0030). Different complexes produce different outputs, so masking
and pooling are correct.

Three defects were found and fixed in the process: `float(loss)` on a
grad-tracking tensor, correlation metrics emitting numpy warnings on
constant-output epochs, and - the one that mattered - a nan score becoming
`best_score`, after which every `score > nan` comparison was False and the run
silently kept its first epoch's weights.

## Honest limits

- **The +0.30–0.38 target for the full interaction set is an estimate, not a
  measurement.** Everything in the first table above is measured; the value of
  the new interaction features will not be known until step 02 runs on the
  re-extracted training set.
- **Conformal coverage degrades on novel targets.** Measured with v1 features:
  nominal 90%, empirical 68.7%. The benchmark is deliberately protein- and
  ligand-disjoint, which breaks the exchangeability the guarantee assumes. This
  is worth reporting as a finding rather than hiding — it quantifies what a
  distribution-free guarantee is actually worth under domain shift.
- **The adaptive interval width was not yet informative** on v1 features
  (narrowest third MAE 1.34 vs widest third 1.40). If that persists with v2
  features, the σ head needs replacing with ensemble variance or quantile
  regression.
- **Boltz-2 at r ≈ 0.48 remains ahead.** v2 is aiming to beat PRODIGY-LIG
  decisively and close part of the gap, while winning outright on speed,
  hardware, interpretability and calibrated uncertainty.

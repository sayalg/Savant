# Savant

## A protein–ligand binding affinity predictor

Savant predicts binding affinity (pKd) for a protein–ligand pair from **an amino-acid
sequence and a ligand SMILES string**. It needs no experimental or predicted
structure, no docking, and no GPU.

```bash
python savant_v1/savant1.py --seq MTVPDRSEIAGKWYVV... --smiles "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
```

## Why this shape

On a 500-pair benchmark of measurements published in 2025 or later, against
ligands unlike anything in training, savant v1:

- **matches Boltz-2 at ranking compounds within a target** — the lead-optimisation
  task — despite Boltz-2 folding every complex on a GPU;
- **beats models fitted on each target's own measured compounds**, the baseline a
  computational chemist would otherwise reach for;
- **is the only tool tested that exceeds chance at predicting selectivity** once
  target base rates are removed.

The expensive step is embedding the protein, and it happens once per target. Every
additional ligand costs ~0.07 s on CPU, so screening one target against a large
library is cheap in a way that co-folding methods cannot be.

Full methodology, benchmark design and limitations: [`report/savant_report.qmd`](report/savant_report.qmd).

## Install

```bash
poetry install
```

Savant v1 needs `torch`, `transformers`, `xgboost`, `rdkit` and `joblib`. It runs on
CPU; a GPU speeds up the first pair but barely changes screening throughput, since
the per-ligand path is RDKit- and tree-bound rather than neural.

## Usage

```bash
# one pair
python savant_v1/savant1.py --seq MTVPDRSEI... --smiles "CCO"

# the smaller, faster protein encoder
python savant_v1/savant1.py --model esm8 --seq MTVPDRSEI... --smiles "CCO"

# batch: a CSV with `seq` and `smiles` columns
python savant_v1/savant1.py --csv pairs.csv --out scored.csv

# read the sequence out of a structure you happen to have
python savant_v1/savant1.py --pdb complex.pdb --smiles "CCO"
```

```python
import sys; sys.path.insert(0, "savant_v1")
from savant1 import Savant

sv = Savant(model="esm650")               # or "esm8"
sv.predict(["MTVPDRSEI..."], ["CCO"])     # -> p_binder, is_binder, pKd
```

| Option | Description |
| --- | --- |
| `--seq` / `--smiles` | Protein sequence and ligand SMILES. |
| `--csv` / `--out` | Batch mode: columns `seq` + `smiles`, or `pdb` + `smiles`. |
| `--model` | `esm650` (default, ESM2-650M) or `esm8` (ESM2-8M, 80× smaller encoder). |
| `--pdb` | Read the sequence from a structure. Bond orders inferred from geometry are guesses, so pass `--smiles` too. |
| `--models` | Override the model directory, or point at a bundle `.pkl`. |

**Pass `--smiles` whenever you have it.** Every ligand feature is computed from the
SMILES; connectivity perceived from 3-D coordinates turns `C(=O)O` into `C(O)O`.

## Output

Three columns per pair:

- `pKd` — predicted `neg_log10_affinity_M`
- `is_binder` — `pKd` above the binder threshold (6.0)
- `p_binder` — classifier probability, or `NaN` for regressor-only bundles

The current models ship without a separate classifier: on the benchmark it matched
the regressor at one threshold and lost at another, so thresholding the predicted
pKd is both simpler and no worse — and it gives you every threshold, chosen at
inference time.

## Models

`savant_v1` looks for `ML_model/work_full_v3/savant_full_<variant>.pkl`, falling back
to `work_full_v2`, then to the raw training output in `<dir>/<variant>/`. Both
boosters travel as `save_model()` JSON inside the bundle, so they carry their own
column list and intercept and load across XGBoost versions.

Train them with [`ML_model/train_savant_full_dedup.ipynb`](ML_model/train_savant_full_dedup.ipynb).

## Limitations

Read these before trusting a number:

- **No absolute affinity.** Predictions are compressed toward the mean and carry a
  systematic negative bias. Rankings survive this; calibrated pKd values do not.
- **No cross-target comparison.** Savant is weak at telling you which of two
  *proteins* binds more tightly. Do not use it to rank targets.
- **RMSE loses to a constant predictor** on the benchmark — as does every tool
  tested, including Boltz-2. No absolute-error claim is available.
- **Not demonstrated on novel targets.** The benchmark is ligand-novel and
  temporal, but its targets appear in training.
- **Label ceiling.** The training corpus pools Kd, Ki and IC50 with no assay-type
  column, and identical pairs in it disagree by a median 0.62 pKd. A perfect model
  would still score ~0.81 RMSE against this data.

## Earlier generations

| Directory | What it is | Status |
| --- | --- | --- |
| [`savant_v1/`](savant_v1/) | **Current.** Structure-free, sequence + SMILES, CPU. | Use this |
| [`savant_v0.5/`](savant_v0.5/) | Structure-based: adds seven HADDOCK3 energy terms from a docked complex. Runs HADDOCK3 itself. | Kept as the benchmark's structure-based arm |
| `savant_v2/` | Replaced HADDOCK3 with fast interface counting. | Superseded |
| `savant/` | The original CLI over an already-docked complex (`tender_comb` + `mango_sponge`). | Superseded |

`savant_v0.5` is the only version that consumes docking energetics. Those six terms
were worth about +0.05 R² internally but only +0.012 Pearson on the benchmark —
inside the noise — at minutes per complex. That is why v1 drops them.

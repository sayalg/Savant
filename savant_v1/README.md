# Savant v1

One file, `savant1.py`. Two models, chosen with `--model`:

| `--model` | features | protein embedding | needs |
|---|---|---|---|
| `esm650` (default) | 2935 | ESM2-650M, 1280-dim | sequence + SMILES |
| `esm8` | 1975 | ESM2-8M, 320-dim | sequence + SMILES |

Both are trained on the same ~1.8M gated pairs with the same rows, split, ligand
features and hyperparameters. They differ **only** in the protein embedding, so a
difference in their scores is attributable to the protein representation and
nothing else.

## Which models get loaded

`MODEL_DIRS` is searched newest-first and the first directory that exists wins;
`--models` overrides it.

| directory | built by | contents |
|---|---|---|
| `ML_model/work_full_v3` | `train_savant_full_dedup.ipynb` | averaged replicate labels, **regressor only** |
| `ML_model/work_full_v2` | `train_savant_full_w_classifier.ipynb` | raw labels, regressor + classifier |

**The classifier is optional.** The current models ship without one: on the
benchmark a trained classifier matched the regressor at the pKd 6 cut (ROC 0.679
vs 0.674) and lost at pKd 7 (0.559 vs 0.574), so it was costing half the training
budget to add nothing. A regressor gives you every threshold, chosen at inference
time.

With no classifier, `predict` returns `p_binder` as NaN and derives `is_binder`
from `pKd > binder_pkd`. The three-column schema never changes, so downstream
code that selects columns by presence keeps working and an all-NaN probability
column is correctly ignored rather than scored as a real prediction.

## What changed from v0.5

v0.5 carried three models. Two of them needed a HADDOCK3 docking run and a
`feature_spec.json` that had to be rebuilt on the training machine. v1 keeps only
the two structure-free full models, so all of that is gone:

- no HADDOCK columns, no `--docking`, no chain arguments for features
- no `feature_spec.json`, no `--build-spec`, no `--split`
- no scikit-learn pickles, so no sklearn version requirement

Both boosters travel as `save_model()` JSON inside a joblib bundle. The JSON
carries its own column list and fitted intercept and loads across XGBoost
versions, which is why there is nothing to build before scoring.

## Use

```bash
python savant1.py --seq MTVPDRSEI... --smiles "CCO"
python savant1.py --model esm8 --seq MTVPDRSEI... --smiles "CCO"
python savant1.py --csv pairs.csv --out scored.csv          # columns: seq, smiles
python savant1.py --pdb complex.pdb --smiles "CCO"          # sequence from the structure
```

```python
from savant4 import Savant
sv = Savant(model="esm650")            # or "esm8"
sv.predict(["MTVPDRSEI..."], ["CCO"])  # -> p_binder, is_binder, pKd
```

Models are looked up as `savant_full_<variant>.pkl` in the first existing
directory from `MODEL_DIRS`, falling back to the raw training output in
`<dir>/<variant>/`. Override with `--models` (a directory or a bundle `.pkl`).

## Feature layout

Order is load-bearing. The width check in `features()` is a backstop, not a
guarantee — two different block orders can produce the same width, so a wrong
order will score silently and wrongly.

| block | width | from |
|---|---|---|
| ChemBERTa ligand embedding | 384 | SMILES |
| RDKit named descriptors | 226 | SMILES, columns from the bundle |
| ECFP4 count fingerprint | 1024 | SMILES |
| ESM2 protein embedding | 1280 \| 320 | sequence |
| AA composition + length | 21 | sequence |

Embeddings are mean-pooled over the last hidden state with padding masked out,
in fp16 on GPU and with a 1024-token cap — matching training. A sequence longer
than 1024 tokens is truncated, exactly as it was during training.

## Known limitations

- **Between-target prediction is weak.** On the 500-pair benchmark, savant's
  correlation across target means was ~0.17 against Boltz-2's ~0.35, while its
  within-target correlation was comparable. Read that as: it ranks compounds
  against one target about as well as Boltz-2, and it does not reliably tell you
  which of two targets binds tighter. Absolute pKd across different proteins is
  the weakest thing it produces.
- **Predictions are compressed.** Squared-error training on a noisy label shrinks
  predictions toward the mean; the predicted spread runs well under the true
  spread. Ranking survives this, absolute calibration does not.
- **Weak binders are underrepresented, not excluded.** Training keeps pKd 2–12,
  so the 68,716 rows below pKd 4 (3.7% of the data) are in there — but they are a
  small minority, and squared-error training on a skewed target pulls low
  predictions upward. Expect weak binders to come back overestimated. (This
  differs from savant v0.5's structure-free models, which were built from
  `combined_50k_filtered.csv` under a hard pKd ≥ 4 cut.)
- **The label itself is noisy.** The training source pools Kd, Ki and IC50 with
  no assay-type column; identical protein–ligand pairs in it disagree by a median
  0.62 pKd. A perfect model would still score RMSE around 0.85 against it.
- **Bond orders from a PDB are guesses.** `read_pdb` recovers the sequence
  exactly, but ligand connectivity is inferred from geometry. Every ligand
  feature comes from the SMILES, so pass `--smiles` whenever possible.
- **RDKit version matters.** The bundle stores the descriptor column list that
  trained it. A different RDKit build that does not produce one of those columns
  raises rather than silently substituting.

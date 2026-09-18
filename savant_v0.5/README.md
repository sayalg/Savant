# Savant v0.5

> **Superseded by [savant v1](../savant_v1/).** v1 is structure-free, runs on CPU,
> and is the tool to use. v0.5 is kept because it is the only generation that
> consumes docked-complex energetics, and it is the **structure-based arm of the
> benchmark comparison**.

One file, `savant05.py`. It **runs HADDOCK3 itself** — earlier revisions imported
the legacy `savant/` package for that, and the code is now inlined, so this file
plus its model weights are the whole dependency. `haddock3` must be on PATH.

Three interchangeable models, chosen with `--model`:

| `--model` | features | needs | source |
|---|---|---|---|
| `structure-free` | 1972 | seq + SMILES | `../ML_model/models_structure_free/` |
| `structure-based` | 1979 | + 7 HADDOCK3 columns | `../ML_model/models_structure_free/` |
| `full` | 2935 | seq + SMILES | `../ML_model/work_full/` |

`structure-free` and `structure-based` are the two tiers from
`ML_model/train_structure_free_model.ipynb` (both keep ECFP4); they differ only
by whether the HADDOCK3 block is available. `full` is the large gated model from
`ML_model/train_full_dataset.py`, trained on ~1.8M pairs with a 1280-dim
ESM2-650M protein embedding.

## Docking

`--model structure-based` docks the complex itself. No `--docking` argument and no
separate HADDOCK3 step:

```bash
python savant05.py --model structure-based --pdb complex.pdb \
    --protein-chain A --ligand-chain B --smiles "CCO"
```

If you already have the terms, pass them and the docking run is skipped:

```bash
python savant05.py --model structure-based --pdb complex.pdb --smiles "CCO" \
    --docking "score=-89.9,total=-218,vdw=-48.4,elec=-169.6,desolv=-7.6,bsa=830.6"
```

The emitted config is the `[topoaa]` + `[emref]` workflow that produced
`combined_50k_filtered.csv`, and the six energy terms are read from the same emref
CAPRI table — byte-identical to what the legacy package used, so features match
training by construction. The seventh column, `num_interacting_residues`, is a
distance query on the pose at 6.0 Å.

`ML_model/benchmark/eval/scripts/04_run_savant.py` imports this module for the
benchmark, so the tool and the benchmark run the same docking code and there is no
second implementation that can drift.

**Docking costs minutes per complex.** Those six terms were worth about +0.05 R²
internally but only +0.012 Pearson on the 500-pair benchmark — inside the noise.
That is why v1 drops them, and why you should reach for v1 unless you specifically
need the structure-based comparison.

## First: build the feature spec (structure-free / structure-based only)

**On the machine that trained the models**, once:

```bash
python savant05.py --build-spec
```

This writes `feature_spec.json`. It is not optional for the structure-free /
structure-based models, and it cannot be copied from another install. Two things
their pickles do not carry live in it:

- **Which RDKit descriptor columns were kept.** Training took the full RDKit
  descriptor set and dropped the columns constant across the training data.
  Different RDKit versions expose different descriptor sets, so the surviving
  list belongs to the install that trained the models — 212 columns on the HPC,
  211 on a 2025.03.6 install here.
- **The XGBoost intercept, as a fallback.** Loaded in the build that trained
  them, the models keep their fitted `base_score` (6.670542). Loaded elsewhere
  it can reset to the 0.5 default, which shifts every predicted pKd down by
  about 6. `--build-spec` recovers the right value by replaying the split, and
  `_load_regressor` applies it only when the model actually came back with 0.5.

The **`full`** model needs no spec: its boosters are stored as `save_model()`
JSON, which keeps its own column list and intercept, and loads across XGBoost
versions.

## Then: score

```bash
# structure-free (no docking)
python savant05.py --model structure-free --seq MTVPDRSEI... --smiles "CCO"

# structure-based (with a docking run)
python savant05.py --model structure-based --pdb complex.pdb \
    --protein-chain A --ligand-chain B --smiles "CCO" \
    --docking "score=-89.9,total=-218,vdw=-48.4,elec=-169.6,desolv=-7.6,bsa=830.6"

# the 1.8M gated model
python savant05.py --model full --seq MTVPDRSEI... --smiles "CCO"

# batch
python savant05.py --model full --csv pairs.csv --out scored.csv
```

```python
from savant3 import Savant
sv = Savant(model="structure-free")           # or "structure-based" or "full"
sv.predict(["MTVPDRSEI..."], ["CCO"])          # -> p_binder, is_binder, pKd
```

If `--model` is omitted, the CLI picks `structure-based` when all 7 HADDOCK
columns are present and `structure-free` otherwise; `full` is only used when
named. `--split` selects the trained variant of the structure-free / -based
models (`protein_grouped` default, `ligand_grouped`, `both_disjoint`); `full`
has no splits. Override the model location with `--models` — a directory, or, for
`full`, a bundle `.pkl`.

## Feature layout

Order is load-bearing — it has to match how each training matrix was built, and
the width check does **not** catch a wrong order (two orders can share a width).

**structure-free / structure-based** (`train_structure_free_model.ipynb`, cell 25):

| block | width | from |
|---|---|---|
| ChemBERTa ligand embedding | 384 | SMILES |
| ESM2-8M protein embedding | 320 | sequence |
| RDKit descriptors (AddHs) | 11 | SMILES |
| RDKit full descriptor set | 212 | SMILES, columns from the spec |
| AA composition + length | 21 | sequence |
| ECFP4 count fingerprint | 1024 | SMILES |
| HADDOCK3 block | 7 | docking run (structure-based only) |

`structure-free` = 1972, `structure-based` = 1979. **AA comes before ECFP.**

**full** (`train_full_dataset.py`):

| block | width | from |
|---|---|---|
| ChemBERTa ligand embedding | 384 | SMILES |
| RDKit named descriptor set | 226 | SMILES, columns from `rdkit_columns.json` |
| ECFP4 count fingerprint | 1024 | SMILES |
| ESM2-650M protein embedding | 1280 | sequence |
| AA composition + length | 21 | sequence |

`full` = 2935.

`num_interacting_residues` is the one HADDOCK column `read_pdb` recovers from
coordinates (residues within 6.0 Å of any ligand atom). The other six come from
the docking run and must be supplied.

## Held-out performance (structure-free / -based)

From `ML_model/models_structure_free/results.csv`:

| features | split | R² | Pearson | RMSE | ROC AUC |
|---|---|---|---|---|---|
| structure-free | protein-grouped | 0.400 | 0.633 | 1.163 | 0.799 |
| structure-free | ligand-grouped | 0.440 | 0.667 | 1.123 | 0.805 |
| structure-free | both-disjoint | 0.355 | 0.599 | 1.261 | 0.774 |
| structure-based | protein-grouped | 0.451 | 0.674 | 1.112 | 0.812 |
| structure-based | ligand-grouped | 0.491 | 0.704 | 1.071 | 0.811 |
| structure-based | both-disjoint | 0.385 | 0.623 | 1.232 | 0.785 |

The HADDOCK3 block is worth about +0.05 R² and +0.035 Pearson across all three
splits. These rows are **not comparable across splits** — each split has its own
test set. `ligand-grouped` scores highest because its test set lets proteins
repeat, not because that model is better; compare down a column, not across.
`full`'s held-out metrics are in `ML_model/work_full/metrics.json`
(R² ≈ 0.33, ROC AUC ≈ 0.78 on its much larger, harder test set).

## Known limitations

- **Bond orders from a PDB are guesses.** `read_pdb` recovers the sequence
  exactly, but ligand connectivity is inferred from geometry, so `C(=O)O` comes
  back as `C(O)O`. Every ligand feature is computed from the SMILES, so pass
  `--smiles` whenever possible.
- **`CONTACT_CUTOFF` is calibrated on one complex.** 6.0 Å reproduces
  `num_interacting_residues` for `dock_1`; that is a single data point.
- **Failed dockings.** Rows with `vdw > 1000` are clashing poses, not energies.
  Training dropped ~400 of them; scoring warns rather than refusing.
- **Below pKd 4 is out of distribution.** Training filtered those rows out, so
  weak binders come back overestimated.
- **The `xgb_*.pkl` regressors are pickled XGBoost objects** and load only under
  a compatible XGBoost version (a mismatched version raises "input stream
  corrupted"). The `full` model avoids this by storing its boosters as portable
  `save_model()` JSON.

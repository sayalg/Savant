# `savant` package internals

Installation and usage are in the [repository README](../README.md). This page
covers the module layout, the Python API and the file formats, for anyone
extending the tool or calling it from their own code.

## Layout

```
savant/
├── __init__.py    the public API, re-exported from the modules below
├── __main__.py    python -m savant
├── cli.py         argument parsing, orchestration, progress on stderr
├── inputs.py      sequences and ligands from strings, FASTA, .smi, .csv, PDB
├── features.py    the feature blocks, and the one order they may go in
├── encoders.py    mean-pooled ESM2 / ChemBERTa embeddings, cached per process
├── model.py       bundle loading, intercept repair, the Savant class
└── report.py      the JSON document and the CSV table
```

`torch` and `transformers` are imported lazily inside `encoders.py`, so reading a
SMILES file or inspecting a bundle does not pull in a deep-learning stack.

## Python API

```python
from savant import Savant

sv = Savant()                                      # esm650; Savant("esm8") for the small one
sv.screen("MKCLLLALAL...", ["CCO", "c1ccccc1"])    # one target, many ligands, ranked
sv.predict(["MKCLLL..."], ["CCO"])                 # arbitrary parallel pairs
```

`screen(sequence, smiles)` returns a DataFrame with `smiles`, `p_binder`,
`is_binder`, `pKd`, `Kd_nM` and `rank`, sorted best-first. `predict(seqs, smiles)`
is the general call and takes two parallel sequences of equal length — one row
per pair, in input order.

Lower-level pieces, if you want to build a matrix once and score it repeatedly:

```python
X = sv.features(seqs, smiles)      # (n, 2935) in the trained block order
sv.predict_from(X)                 # p_binder, is_binder, pKd
```

Reading inputs and writing outputs:

```python
from savant import (Savant, read_targets, read_ligands,
                    build_report, write_report, build_table, write_table)

sv = Savant()
targets = read_targets("target.fasta")             # -> [Target(sequence, id, source)]
ligands = read_ligands(["library.smi", "CCO"])     # -> [Ligand(smiles, name)]
smiles = [lig.smiles for lig in ligands]

results = [(t, sv.screen(t.sequence, smiles)) for t in targets]
write_report(build_report(sv, results), "screen.json")
write_table(build_table(sv, results), "screen.csv")
```

`build_report` and `build_table` both take `precision` and `top`. Passing `top`
to them rather than truncating the frame yourself keeps `n_scored` accurate.

## Input formats

`read_targets(value)` treats a value that names an existing file as a file and
anything else as a literal sequence. Whitespace and digits are stripped, and a
non-residue character raises rather than being embedded silently.

`read_ligands(values)` accepts a list mixing SMILES and paths. `.csv`/`.tsv` are
read by column name (`smiles`, `smi`, `canonical_smiles`, `ligand_smiles`,
`structure`, plus a name column from `name`, `id`, `title`, `compound_id`,
`molecule_chembl_id`, …). Anything else is read line by line in `.smi` style:
one SMILES, optional whitespace-separated name, `#` comments and a header row
skipped.

`dedupe_ligands(ligands)` returns `(unique, n_dropped)`, keeping the first
occurrence and its name.

## Output formats

Both are described in the [repository README](../README.md#output). Two
invariants worth repeating here:

- The protein sequence appears **once per target** in the JSON and **not at all**
  in the CSV, which carries `target_id` to join on instead.
- `n_scored` counts ligands run, `n_reported` counts rows written. They differ
  when `--top` is used.

`_py()` in `report.py` normalises numpy scalars and turns NaN into `null`, since
`json.dump` would otherwise emit bare `NaN`, which is not valid JSON.

## Which models get loaded

Searched in order; the first that exists wins. `SAVANT_MODEL_DIR` overrides the
search, and `--models` / the `models_dir` argument overrides everything (a
directory, or one bundle `.pkl`).

| directory | contents |
|---|---|
| `ML_model/models_full` | the shipped bundles: averaged replicate labels, regressor only |
| `ML_model/work_full_v2` | an earlier raw-label build that also carried a classifier; not part of this branch |

Bundles are looked up as `savant_full_<variant>.pkl`, falling back to the raw
training output in `<dir>/<variant>/xgb_regressor.json` plus
`<dir>/rdkit_columns.json`.

A bundle carries its own RDKit column list, block widths and fitted intercept, so
it loads across XGBoost versions and there is no feature spec to build. Some
XGBoost builds report `base_score` as 0.5 after loading and would then predict
about 6 pKd low; `load_booster` reads the right value out of the saved JSON and
restores it, printing a line when it does.

**The classifier is optional and the shipped bundles omit it.** With no
classifier, `p_binder` is NaN, `is_binder` comes from thresholding pKd at
`binder_pkd`, and the JSON and CSV leave the probability column out entirely
rather than emitting an all-null column that looks like a failed prediction.

## Feature layout

Order is load-bearing. The width check in `features.build_matrix` is a backstop,
not a guarantee — two different block orders can produce the same width, so a
wrong order scores silently and wrongly rather than raising.

| block | width | from |
|---|---|---|
| ChemBERTa ligand embedding | 384 | SMILES |
| RDKit named descriptors | 226 | SMILES, columns from the bundle |
| ECFP4 count fingerprint | 1024 | SMILES |
| ESM2 protein embedding | 1280 \| 320 | sequence |
| AA composition + length | 21 | sequence |

Details that are training decisions rather than preferences, and that change
every prediction if altered:

- Embeddings are mean-pooled over the last hidden state with padding masked out.
- Proteins are encoded in fp16 on GPU with a 1024-token cap; longer sequences are
  truncated.
- The 11 `h_*` descriptors are computed on the hydrogen-added molecule.
- RDKit descriptors are cast to float32 *before* infinities are cleaned, because
  a few (Ipc, Kappa2) only overflow on the cast.
- ECFP4 is a Morgan **count** fingerprint at radius 2, 1024 bits.
- AA composition is the 20 standard residue fractions followed by raw length; a
  non-standard residue contributes to the length but to no fraction.

The bundle stores the descriptor column list that trained it. A different RDKit
build that does not produce one of those columns raises rather than silently
substituting.

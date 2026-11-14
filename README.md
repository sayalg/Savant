# Savant

**Protein–ligand binding affinity prediction from sequence and SMILES.**

Savant predicts the dissociation constant (pKd) of a protein–ligand pair from an
**amino-acid sequence and a SMILES string**. It needs no experimental structure,
no predicted structure, no docking and no GPU, so a screen runs on whatever
hardware you already have.

```bash
savant --seq target.fasta --smiles library.smi --out screen.json --csv screen.csv
```

The expensive step is encoding the protein, and it happens once per target.
Every additional ligand after that costs about 0.018 s on a CPU, so a library of
100,000 compounds against one target takes roughly half an hour.

## Install

Requires Python 3.12–3.14.

```bash
git clone https://github.com/sayalg/Savant.git
cd Savant
poetry install
```

That installs the package and puts a `savant` command on your PATH. Without
installing, `python -m savant` from the repository root is equivalent.

With pip instead of poetry:

```bash
pip install -e .
```

Model weights ship in `ML_model/models_full/`, so a fresh clone scores
immediately — there is nothing to download or build first. The protein and
ligand encoders (ESM2 and ChemBERTa) are pulled from Hugging Face on first use
and cached; set `HF_HOME` to control where.

## Quick start

```bash
# one sequence, one ligand
savant --seq MKCLLLALALTCGAQAL... --smiles "CCO"

# a FASTA target against a library file
savant --seq target.fasta --smiles library.smi --out screen.json

# CSV instead of JSON
savant --seq target.fasta --smiles library.smi --csv screen.csv --no-json

# keep only the 100 best-ranked ligands
savant --seq target.fasta --smiles library.smi --top 100

# stream to stdout and filter with jq
savant --seq target.fasta --smiles library.smi --out - | jq '.targets[0].predictions[:5]'
```

## Input

**`--seq` / `-s`** takes either a literal sequence or a path. A value that names
an existing file is read as a file; anything else is treated as a sequence.

| value | read as |
|---|---|
| `MKCLLLALAL...` | a literal sequence |
| `target.fasta` | FASTA. A multi-record file scores every ligand against every record |
| `target.txt` | plain text; whitespace and residue numbering are stripped |
| `complex.pdb` | the sequence of `--protein-chain` (default `A`) |

**`--smiles` / `-l`** takes any number of values, mixing SMILES and files freely.

| value | read as |
|---|---|
| `"CCO"` | one literal SMILES |
| `"CCO" "c1ccccc1"` | two literal SMILES |
| `library.smi`, `library.txt` | one SMILES per line, optional name after whitespace; `#` comments and a header row are skipped |
| `library.csv`, `library.tsv` | a SMILES column found by name (`smiles`, `canonical_smiles`, `ligand_smiles`, …), plus a name column if present |
| `library.csv hits.smi "CCO"` | all of the above, concatenated |

Repeated SMILES are dropped, keeping the first occurrence and its name; pass
`--keep-duplicates` to score them all. Every sequence and SMILES is validated
before the encoder loads, so a bad input fails in under a second.

## Output

JSON by default, CSV with `--csv`, or both. The protein sequence is written
**once per target**, never once per ligand — repeating a 1,000-residue sequence
across a 100,000-ligand screen would be 100 MB of the same string.

### JSON

`targets` is always a list, so a consumer needs one code path whether you scored
one protein or twenty.

```json
{
  "savant": {
    "version": "1.0.0",
    "model": "esm650",
    "protein_encoder": "facebook/esm2_t33_650M_UR50D",
    "ligand_encoder": "DeepChem/ChemBERTa-77M-MLM",
    "n_features": 2935,
    "prediction": "pKd = -log10(Kd[M]); Kd_nM = 10^(9 - pKd)",
    "binder_call": "pKd > 6.0",
    "n_targets": 1, "n_scored": 5, "n_reported": 5, "top": null,
    "generated": "2026-09-18T19:56:13+00:00"
  },
  "targets": [
    {
      "id": "P02754_BLG",
      "length": 178,
      "source": "target.fasta",
      "n_scored": 5,
      "n_reported": 5,
      "sequence": "MKCLLLALALTCGAQAL...",
      "predictions": [
        {"rank": 1, "smiles": "CCCCCCCCCCCCCCCCCCCC(=O)O",
         "name": "arachidic_acid", "pKd": 6.3983, "Kd_nM": 399.6495,
         "is_binder": true}
      ]
    }
  ]
}
```

`n_scored` is how many ligands were run and `n_reported` how many are in the
file, so a `--top 10` file from a 100,000-ligand screen still records the screen
size. `--no-sequence` drops the sequence and keeps `id`, `length` and `source`.

### CSV

One row per (target, ligand), ordered by rank. `target_id` joins a row back to
its protein; there is no sequence column.

```csv
target_id,rank,smiles,name,pKd,Kd_nM,is_binder
P02754_BLG,1,CCCCCCCCCCCCCCCCCCCC(=O)O,arachidic_acid,6.3983,399.6495,True
P02754_BLG,2,CCCCCCCCCCCCCCCC(=O)O,palmitic_acid,6.214,610.8914,True
P02754_BLG,3,CC(C)Cc1ccc(cc1)C(C)C(=O)O,ibuprofen,5.2573,5529.6448,False
```

## Options

| option | meaning |
|---|---|
| `-s, --seq SEQ\|FILE` | sequence, FASTA, text or PDB (required) |
| `-l, --smiles SMILES\|FILE ...` | SMILES and/or library files (required) |
| `-o, --out FILE` | JSON path, or `-` for stdout (default `savant_predictions.json`) |
| `-c, --csv FILE` | also write CSV here, or `-` for stdout |
| `--no-json` | skip the JSON, for when you only want `--csv` |
| `--top N` | keep only the N best-ranked ligands per target |
| `--no-sequence` | omit the sequence from the JSON |
| `--precision N` | decimal places for pKd and Kd_nM (default 4) |
| `--keep-duplicates` | score repeated SMILES more than once |
| `--protein-chain X` | chain to read when `--seq` is a PDB (default `A`) |
| `-m, --model {esm650,esm8}` | protein encoder variant (default `esm650`) |
| `--models DIR\|PKL` | model directory or one bundle `.pkl` (env `SAVANT_MODEL_DIR`) |
| `--ligand-batch N` / `--protein-batch N` | encoder batch sizes |
| `-q, --quiet` | suppress progress on stderr |

Progress always goes to stderr, so stdout stays clean for piping.

## Models

Two variants, trained on the same ~1.8M pairs with the same rows, split, ligand
features and hyperparameters. They differ only in the protein encoder.

| `--model` | features | protein encoder |
|---|---|---|
| `esm650` (default) | 2935 | ESM2-650M, 1280-dim |
| `esm8` | 1975 | ESM2-8M, 320-dim |

`esm8` is cheaper per target and useful when encoding many proteins; `esm650` is
the default and the stronger model.

Weights are found in `ML_model/models_full/`. Override with `--models` or the
`SAVANT_MODEL_DIR` environment variable.

## Python API

```python
from savant import Savant

sv = Savant()                                     # esm650; Savant("esm8") for the small one
sv.screen("MKCLLLALAL...", ["CCO", "c1ccccc1"])   # ranked: smiles, pKd, Kd_nM, rank
sv.predict(["MKCLLL..."], ["CCO"])                # arbitrary parallel pairs
```

Building the output documents yourself:

```python
from savant import Savant, read_ligands, read_targets, build_report, write_report

sv = Savant()
targets = read_targets("target.fasta")
smiles = [lig.smiles for lig in read_ligands(["library.smi"])]

results = [(t, sv.screen(t.sequence, smiles)) for t in targets]
write_report(build_report(sv, results), "screen.json")
```

The protein encoder is cached per process, so scoring many ligands against one
already-encoded target costs only descriptor computation and tree inference.

## Notes

- Savant ranks ligands against one protein. It is not calibrated for comparing
  different proteins to each other, so select within a target rather than across
  targets, and prefer rank over an absolute pKd or Kd cutoff.
- The model ships as a regressor. Binder calls come from thresholding the
  predicted pKd, so the threshold is chosen at inference time rather than fixed
  during training.
- The training label pools Kd, Ki and IC50, and the model returns a point
  estimate with no uncertainty.
- Sequences longer than 1024 tokens are truncated, matching how the model was
  trained.
- Ligand features come from the SMILES, not from coordinates. When using
  `--seq complex.pdb`, still pass the ligand SMILES explicitly.

## Repository

| path | what |
|---|---|
| [`savant/`](savant/) | the tool; [`savant/README.md`](savant/README.md) covers the package internals |
| `ML_model/models_full/` | the trained model bundles |
| `ML_model/`, `Scripts/` | training data preparation, model training, and the evaluation benchmark |

## License

MIT. See [`LICENSE`](LICENSE).

## Citation

If you use Savant, please cite:

> Guirales-Medrano S, Ford CT, Janies D. *Savant: Fast, Structure-Free
> Protein-Ligand Affinity Prediction Enabled Ligand Screening.*
>
> _Link to be added on publication._

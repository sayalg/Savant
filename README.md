# Savant
## A Protein-Ligand Binding Affinity Predictor

Savant is a protein to ligand binding affinity predictor using a machine learning algorithm. It has been trained on experimently derived protein-ligand docking experiments to predictor the dissociation constant (Kd).

## `savant` command-line tool

Savant ships a CLI that takes a single **already-docked** protein-ligand
complex and predicts whether the pair binds and, if so, its pK. It converts
the structure to PDB if needed, splits it by chain, and processes it with
**RDKit** (ligand descriptors) and **HADDOCK3** (docking energetics). The 18
resulting features are fed — in memory, no intermediate CSV — to two trained
models:

1. **`tender_comb`** (scikit-learn RandomForest) tags the complex as
   `binder` / `nonbinder`.
2. **`mango_sponge`** (XGBoost) predicts pK (`neg_log10_affinity_M`) — but
   only for complexes classified as binders.

The result is written as a single **JSON** document.

### Install

```bash
poetry install
```

> **HADDOCK3 runs on Linux only.** It is declared as a dependency but guarded
> with a `sys_platform == 'linux'` marker, so `poetry install` installs it on
> Linux (e.g. your HPC) and skips it on Windows/macOS, where it cannot build or
> run. Develop the CLI anywhere; run the full pipeline on Linux.

### Usage

```bash
poetry run savant DOCKED_COMPLEX.pdb \
    --protein-chain A \
    --ligand-chain B \
    --ligand-smiles "CC(C)Cc1ccc(cc1)C(C)C(=O)O" \
    --output results.json
```

| Option | Description |
| --- | --- |
| `STRUCTURE` (arg) | Docked complex: `.pdb`/`.ent` or `.cif`/`.mmcif` (converted to PDB automatically). |
| `-p, --protein-chain` | Chain ID of the protein (required). |
| `-l, --ligand-chain` | Chain ID of the ligand (required). |
| `-s, --ligand-smiles` | Ligand SMILES. **Strongly recommended** — descriptors are topological, so the SMILES yields correct chemistry. Without it, bonds are guessed from 3D coordinates and logP / aromaticity / charge may be wrong. |
| `-o, --output` | Output JSON path (default `savant_results.json`). |
| `--cutoff` | Distance cutoff (Å) for interacting-residue counting (default `4.5`). |
| `--ncores` / `--nemsteps` | HADDOCK3 core count and emref minimisation steps. |
| `--model-dir` | Directory holding the `.pkl` models (default `ML_model/train`, env `SAVANT_MODEL_DIR`). |
| `--classifier-model` / `--regressor-model` | Explicit `.pkl` paths (default: auto-discover `tender_comb_*_local.pkl` / `mango_sponge_*_local.pkl` in `--model-dir`). |

### Output

A JSON document with:

- `protein_sequence` — one-letter amino-acid sequence read from the protein chain.
- `ligand_smiles` / `ligand_smiles_source` — the SMILES used (the `--ligand-smiles` input, or one perceived from the 3D ligand).
- `features` — the 18 model inputs: RDKit descriptors (`molecular_weight`, `logP`, `num_h_donors`, `num_h_acceptors`, `num_rotatable_bonds`, `tpsa`, `num_rings`, `num_aromatic_rings`, `num_atoms`, `num_heavy_atoms`, `formal_charge`), HADDOCK3 energetics (`score`, `total`, `vdw`, `elec`, `desolv`, `bsa`), and `num_interacting_residues`.
- `classification` — `binder` or `nonbinder`, with `classification_probabilities`.
- `pK` — predicted `neg_log10_affinity_M` for binders, or `null` for nonbinders.

```json
{
  "protein_sequence": "MTVPDRSEIAGKWYVV...",
  "ligand_smiles": "CCCCCCCCCCCCCCCCCCCC(=O)O",
  "ligand_smiles_source": "input",
  "features": { "molecular_weight": 312.54, "logP": 7.11, "...": "..." },
  "classification": "binder",
  "classification_probabilities": { "binder": 0.536, "nonbinder": 0.464 },
  "pK": 6.64
}
```

> The trained `.pkl` models were pickled with **scikit-learn 1.6.1** (pinned in
> `pyproject.toml`) so they unpickle without version-mismatch warnings.

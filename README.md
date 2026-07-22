# Savant
## A Protein-Ligand Binding Affinity Predictor

Savant is a protein to ligand binding affinity predictor using a machine learning algorithm. It has been trained on experimently derived protein-ligand docking experiments to predictor the dissociation constant (Kd).

## `savant` command-line tool

Savant ships a CLI that turns a single **already-docked** protein-ligand
complex into a feature table. It converts the structure to PDB if needed,
splits it by chain, and processes it with **RDKit** (ligand descriptors) and
**HADDOCK3** (docking energetics).

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
    --output results.csv
```

| Option | Description |
| --- | --- |
| `STRUCTURE` (arg) | Docked complex: `.pdb`/`.ent` or `.cif`/`.mmcif` (converted to PDB automatically). |
| `-p, --protein-chain` | Chain ID of the protein (required). |
| `-l, --ligand-chain` | Chain ID of the ligand (required). |
| `-s, --ligand-smiles` | Ligand SMILES. **Strongly recommended** — descriptors are topological, so the SMILES yields correct chemistry. Without it, bonds are guessed from 3D coordinates and logP / aromaticity / charge may be wrong. |
| `-o, --output` | Output CSV path (default `savant_results.csv`). |
| `--cutoff` | Distance cutoff (Å) for interacting-residue counting (default `4.5`). |
| `--ncores` / `--nemsteps` | HADDOCK3 core count and emref minimisation steps. |

### Output columns

`molecular_weight`, `logP`, `num_h_donors`, `num_h_acceptors`,
`num_rotatable_bonds`, `tpsa`, `num_rings`, `num_aromatic_rings`, `num_atoms`,
`num_heavy_atoms`, `formal_charge` (RDKit); `score`, `total`, `vdw`, `elec`,
`desolv`, `bsa` (HADDOCK3); `num_interacting_residues` (contacts on the docked
pose).

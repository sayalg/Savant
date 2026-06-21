#!/usr/bin/python3
import os
import sys
import gzip
import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
INTERACTION_CUTOFF = 4.5


def compute_morgan_fp(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Could not parse SMILES: {smiles}")
        return {f"morgan_fp_{i}": None for i in range(MORGAN_BITS)}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS)
    return {f"morgan_fp_{i}": int(b) for i, b in enumerate(fp)}


def count_interacting_residues(haddock_dir: str, job_name: str) -> int | None:
    expected = Path(haddock_dir) / job_name / "output" / "1_emref" / "emref_1.pdb.gz"
    if not expected.exists():
        matches = list((Path(haddock_dir) / job_name).rglob("emref_1.pdb.gz"))
        if not matches:
            logger.warning(f"No emref PDB found for {job_name}")
            return None
        expected = matches[0]

    with gzip.open(expected, "rt") as f:
        structure = PDBParser(QUIET=True).get_structure("docked", f)
    model = next(iter(structure))

    ligand_atoms = [a for a in model.get_atoms() if a.get_full_id()[2] == "B"]
    protein_residues = [r for r in model.get_residues() if r.get_full_id()[2] == "A"]

    interacting = set()
    for res in protein_residues:
        for p_atom in res.get_atoms():
            for l_atom in ligand_atoms:
                if (p_atom.get_vector() - l_atom.get_vector()).norm() <= INTERACTION_CUTOFF:
                    interacting.add(res.get_id())
                    break
            else:
                continue
            break

    return len(interacting)

def combine_csv(directory, csv_file, haddock_dir):
    """Combine all CSV files in a directory into a single file."""
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the directory.")
        return

    df_kd = pd.read_csv(csv_file)
    df_kd = df_kd[['ID', 'neg_log10_affinity_M', 'smiles']]

    dfs = [pd.read_csv(os.path.join(directory, f)) for f in csv_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined = pd.merge(combined, df_kd, left_on='Job_ID', right_on='ID', how='left').drop('ID', axis=1)

    # Morgan fingerprints
    logger.info("Computing Morgan fingerprints...")
    morgan_df = pd.DataFrame(combined['smiles'].map(compute_morgan_fp).tolist())
    combined = pd.concat([combined, morgan_df], axis=1)

    # Interacting residues
    logger.info("Counting interacting residues...")
    combined['num_interacting_residues'] = combined['Job_ID'].map(
        lambda job_id: count_interacting_residues(haddock_dir, job_id)
    )

    combined.to_csv('combined.csv', index=False)
    print(f"Combined {len(csv_files)} CSV files into combined.csv")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 combine_csv.py <directory> <csv_file> <haddock_dir>")
        sys.exit(1)

    combine_csv(sys.argv[1], sys.argv[2], sys.argv[3])

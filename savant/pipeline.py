"""Orchestrate the Savant scoring pipeline for one docked complex."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from savant import FEATURE_COLUMNS
from savant.descriptors import ligand_descriptors
from savant.haddock import run_haddock3
from savant.interactions import count_interacting_residues
from savant.structure import ensure_pdb, split_chains

logger = logging.getLogger(__name__)


def run_pipeline(
    structure: str | Path,
    protein_chain: str,
    ligand_chain: str,
    ligand_smiles: Optional[str] = None,
    workdir: Optional[str | Path] = None,
    cutoff: float = 4.5,
    ncores: Optional[int] = None,
    nemsteps: int = 50,
) -> pd.DataFrame:
    """Run the full pipeline and return a one-row DataFrame with FEATURE_COLUMNS.

    Steps:
      1. Normalise input to PDB (convert mmCIF if needed).
      2. Split into protein and ligand chains.
      3. RDKit ligand descriptors (from SMILES if given, else 3D perception).
      4. HADDOCK3 scoring of the two chains.
      5. Count interacting protein residues on the docked pose.
    """
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="savant_"))
    work.mkdir(parents=True, exist_ok=True)
    logger.info("Working directory: %s", work)

    pdb_path = ensure_pdb(structure, work)
    protein_pdb, ligand_pdb = split_chains(pdb_path, protein_chain, ligand_chain, work)

    descriptors = ligand_descriptors(ligand_pdb, smiles=ligand_smiles)
    docking = run_haddock3(
        protein_pdb, ligand_pdb, work, ncores=ncores, nemsteps=nemsteps
    )
    num_interacting = count_interacting_residues(
        pdb_path, protein_chain, ligand_chain, cutoff=cutoff
    )

    row = {
        **descriptors,
        **docking,
        "num_interacting_residues": num_interacting,
    }
    return pd.DataFrame([[row[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

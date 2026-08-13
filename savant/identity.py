"""Derive the molecular identity of the complex: protein sequence and ligand SMILES."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from rdkit import Chem

logger = logging.getLogger(__name__)


def extract_protein_sequence(protein_pdb: Path) -> str:
    """One-letter amino-acid sequence of the protein chain PDB.

    Peptide fragments (e.g. across unresolved gaps) are concatenated in order.
    """
    structure = PDBParser(QUIET=True).get_structure("protein", str(protein_pdb))
    peptides = PPBuilder().build_peptides(structure)
    sequence = "".join(str(pep.get_sequence()) for pep in peptides)
    if not sequence:
        logger.warning(
            "No amino-acid sequence could be extracted from %s. Is the chosen "
            "protein chain actually a polypeptide?",
            protein_pdb.name,
        )
    return sequence


def resolve_ligand_smiles(
    ligand_pdb: Path,
    smiles_input: Optional[str],
) -> Tuple[str, str]:
    """Return (smiles, source).

    If ``smiles_input`` is given it is used verbatim (source ``"input"``).
    Otherwise a SMILES is perceived from the ligand's 3D coordinates
    (source ``"derived_from_structure"``), which may be chemically inaccurate.
    """
    if smiles_input:
        return smiles_input, "input"

    mol = Chem.MolFromPDBFile(str(ligand_pdb), sanitize=True)
    if mol is None:
        logger.warning(
            "Could not derive a SMILES from the ligand structure %s.",
            ligand_pdb.name,
        )
        return "", "derived_from_structure"
    smiles = Chem.MolToSmiles(mol)
    logger.warning(
        "Ligand SMILES was perceived from 3D coordinates and may be "
        "inaccurate. Pass --ligand-smiles for a reliable structure."
    )
    return smiles, "derived_from_structure"

"""Count protein residues in contact with the ligand in the docked pose."""

from __future__ import annotations

import logging
from pathlib import Path

from Bio.PDB import NeighborSearch, PDBParser

logger = logging.getLogger(__name__)


def count_interacting_residues(
    pdb_path: Path,
    protein_chain: str,
    ligand_chain: str,
    cutoff: float = 4.5,
) -> int:
    """Number of distinct protein residues with any atom within ``cutoff`` (A)
    of any ligand atom, measured on the docked pose provided by the user.
    """
    structure = PDBParser(QUIET=True).get_structure("complex", str(pdb_path))
    model = next(iter(structure))

    ligand_atoms = [
        atom
        for atom in model.get_atoms()
        if atom.get_parent().get_parent().get_id() == ligand_chain
    ]
    if not ligand_atoms:
        raise ValueError(f"No atoms found in ligand chain '{ligand_chain}'.")

    search = NeighborSearch(list(model.get_atoms()))
    interacting = set()
    for lig_atom in ligand_atoms:
        for neighbor in search.search(lig_atom.coord, cutoff):
            residue = neighbor.get_parent()
            if residue.get_parent().get_id() == protein_chain:
                # full_id uniquely identifies the residue across the model
                interacting.add(residue.get_full_id())

    logger.info("Interacting protein residues (<= %.1f A): %d", cutoff, len(interacting))
    return len(interacting)

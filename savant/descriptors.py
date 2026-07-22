"""RDKit ligand descriptors.

All eleven descriptors below are *topological* -- they depend only on the
molecular graph, not on 3D coordinates. So when the caller supplies the ligand
SMILES we compute them straight from the SMILES (correct bond orders, correct
chemistry). Only when no SMILES is given do we fall back to perceiving bonds
from the 3D ligand geometry, which RDKit does heuristically and which can be
wrong for aromatic systems and charged groups -- hence the warning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

logger = logging.getLogger(__name__)

DESCRIPTOR_COLUMNS = [
    "molecular_weight",
    "logP",
    "num_h_donors",
    "num_h_acceptors",
    "num_rotatable_bonds",
    "tpsa",
    "num_rings",
    "num_aromatic_rings",
    "num_atoms",
    "num_heavy_atoms",
    "formal_charge",
]


def _descriptors_from_mol(mol: Chem.Mol) -> Dict[str, float]:
    """Compute the descriptor dict from a sanitised RDKit mol.

    Hydrogens are added so ``num_atoms`` and molecular weight include them,
    matching the behaviour of the original Savant pipeline.
    """
    mol = Chem.AddHs(mol)
    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "logP": Crippen.MolLogP(mol),
        "num_h_donors": Lipinski.NumHDonors(mol),
        "num_h_acceptors": Lipinski.NumHAcceptors(mol),
        "num_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "tpsa": Descriptors.TPSA(mol),
        "num_rings": Lipinski.RingCount(mol),
        "num_aromatic_rings": Lipinski.NumAromaticRings(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": Lipinski.HeavyAtomCount(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
    }


def ligand_descriptors(
    ligand_pdb: Path,
    smiles: Optional[str] = None,
) -> Dict[str, float]:
    """Return RDKit descriptors for the ligand.

    If ``smiles`` is provided, descriptors are computed from it (accurate). The
    3D ligand is still cross-checked for a heavy-atom-count mismatch, which
    usually means the SMILES does not correspond to the docked ligand.
    """
    if smiles:
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            raise ValueError(f"Invalid ligand SMILES: {smiles!r}")

        _warn_on_atom_mismatch(ligand_pdb, template)
        return _descriptors_from_mol(template)

    mol = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse the ligand from {ligand_pdb.name}. "
            "Pass --ligand-smiles so descriptors can be computed reliably."
        )
    logger.warning(
        "No --ligand-smiles given: ligand chemistry was perceived from 3D "
        "coordinates. logP, aromatic-ring, H-bond and formal-charge values may "
        "be inaccurate. Supply --ligand-smiles for correct descriptors."
    )
    return _descriptors_from_mol(mol)


def _warn_on_atom_mismatch(ligand_pdb: Path, template: Chem.Mol) -> None:
    """Warn if the SMILES heavy-atom count differs from the docked ligand's."""
    mol_3d = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=True, sanitize=False)
    if mol_3d is None:
        return
    pdb_heavy = mol_3d.GetNumAtoms()
    smiles_heavy = template.GetNumAtoms()  # SMILES mol has implicit Hs
    if pdb_heavy != smiles_heavy:
        logger.warning(
            "Heavy-atom count differs between --ligand-smiles (%d) and the "
            "docked ligand (%d). Check that the SMILES matches the ligand.",
            smiles_heavy,
            pdb_heavy,
        )

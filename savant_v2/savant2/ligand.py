"""Ligand chemistry: descriptors, fingerprint, and per-atom interaction roles.

Two jobs. First, the ligand-only feature block - the eleven physicochemical
descriptors v1 used, plus an ECFP4 fingerprint (measured worth +0.035 Pearson r
on the benchmark, because eleven scalars cannot distinguish two molecules of
equal weight and logP).

Second, and more important for the interaction fingerprint: per-atom roles.
Which ligand atoms donate a hydrogen bond, which accept, which carry charge,
which belong to an aromatic ring. Element alone cannot answer any of those, and
a docked PDB carries no bond orders - so the roles are transferred from the
SMILES onto the 3D atoms by template matching.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ECFP_BITS = 512
ECFP_RADIUS = 2

DESCRIPTOR_NAMES = [
    "molecular_weight", "logP", "num_h_donors", "num_h_acceptors",
    "num_rotatable_bonds", "tpsa", "num_rings", "num_aromatic_rings",
    "num_atoms", "num_heavy_atoms", "formal_charge",
]

# RDKit's standard Lipinski donor/acceptor definitions, used per-atom here
# rather than as molecule-level counts.
_DONOR = Chem.MolFromSmarts(
    "[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]"
)
_ACCEPTOR = Chem.MolFromSmarts(
    "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
)
# Groups that are charged at physiological pH even when the SMILES is drawn
# neutral - BindingDB records carboxylic acids and amines both ways, and the
# ionic-interaction features would otherwise depend on how a curator typed it.
_ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
_BASE = Chem.MolFromSmarts(
    "[$([NX3;H2,H1,H0;!$(N-[!#6]);!$(N=*);!$(N#*);!$(N-a)]),"
    "$([NX3]=[NX2+0]),$([NX3][CX3]=[NX3+]),$(N=[CX3](N)N)]"
)

_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=ECFP_RADIUS, fpSize=ECFP_BITS
)


def ligand_pdb_block(path: str | Path, ligand_chain: str = "B") -> str:
    """Ligand HETATM/ATOM lines, filtered exactly as ``structure.read_complex``.

    The filtering must match atom for atom, or the RDKit atom order will not
    line up with the coordinate arrays and every per-atom role lands on the
    wrong atom.
    """
    lines = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[21] != ligand_chain:
                continue
            if line[16] not in (" ", "A"):
                continue
            element = line[76:78].strip().upper()
            if not element:
                element = "".join(c for c in line[12:16].strip() if c.isalpha())[:1].upper()
            if element in ("H", "D"):
                continue
            lines.append(line.rstrip("\n"))
    return "\n".join(lines) + "\nEND\n"


def ligand_mol(
    path: str | Path,
    ligand_chain: str = "B",
    smiles: Optional[str] = None,
) -> Optional[Chem.Mol]:
    """RDKit ligand with 3D coordinates and, when ``smiles`` is given, correct
    bond orders, aromaticity and formal charges.

    Returns None if the block cannot be parsed at all. Returns the
    proximity-bonded molecule (no bond orders) if the SMILES template does not
    match, which happens when the docked ligand is a different protonation or
    tautomer than the recorded SMILES.
    """
    block = ligand_pdb_block(path, ligand_chain)
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False, proximityBonding=True)
    if mol is None:
        return None
    if smiles:
        template = Chem.MolFromSmiles(smiles)
        if template is not None:
            try:
                return AllChem.AssignBondOrdersFromTemplate(template, mol)
            except (ValueError, RuntimeError):
                pass      # fall through to the untyped molecule
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, Chem.KekulizeException):
        pass
    return mol


def ligand_atom_types(mol: Optional[Chem.Mol], n_atoms: int) -> Dict[str, object]:
    """Per-atom boolean masks and aromatic ring index arrays.

    Length is checked against ``n_atoms`` by the caller: a mismatch means the
    RDKit parse dropped or merged atoms, and the masks must then be ignored
    rather than silently applied to the wrong atoms.
    """
    empty = {
        "donor": np.zeros(n_atoms, bool), "acceptor": np.zeros(n_atoms, bool),
        "cationic": np.zeros(n_atoms, bool), "anionic": np.zeros(n_atoms, bool),
        "rings": [],
    }
    if mol is None or mol.GetNumAtoms() != n_atoms:
        return empty

    donor = np.zeros(n_atoms, bool)
    acceptor = np.zeros(n_atoms, bool)
    cationic = np.zeros(n_atoms, bool)
    anionic = np.zeros(n_atoms, bool)

    try:
        for (i,) in mol.GetSubstructMatches(_DONOR):
            donor[i] = True
        for (i,) in mol.GetSubstructMatches(_ACCEPTOR):
            acceptor[i] = True
        for match in mol.GetSubstructMatches(_ACID):
            for i in match[1:]:            # the two oxygens carry the charge
                anionic[i] = True
        for match in mol.GetSubstructMatches(_BASE):
            cationic[match[0]] = True
    except RuntimeError:
        return empty

    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge > 0:
            cationic[atom.GetIdx()] = True
        elif charge < 0:
            anionic[atom.GetIdx()] = True

    rings = []
    try:
        for ring in mol.GetRingInfo().AtomRings():
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                rings.append(np.asarray(ring, dtype=int))
    except RuntimeError:
        pass

    return {"donor": donor, "acceptor": acceptor, "cationic": cationic,
            "anionic": anionic, "rings": rings}


def ligand_descriptors(smiles: str) -> Dict[str, float]:
    """The eleven v1 physicochemical descriptors, from SMILES."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {name: float("nan") for name in DESCRIPTOR_NAMES}
    with_h = Chem.AddHs(mol)
    return {
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "logP": float(Crippen.MolLogP(mol)),
        "num_h_donors": float(rdMolDescriptors.CalcNumHBD(mol)),
        "num_h_acceptors": float(rdMolDescriptors.CalcNumHBA(mol)),
        "num_rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "num_rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "num_aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "num_atoms": float(with_h.GetNumAtoms()),
        "num_heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "formal_charge": float(Chem.GetFormalCharge(mol)),
    }


def ecfp(smiles: str) -> np.ndarray:
    """ECFP4 bit vector; all zeros if the SMILES will not parse."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return np.zeros(ECFP_BITS, dtype=np.int8)
    return np.asarray(_generator.GetFingerprintAsNumPy(mol), dtype=np.int8)


def ecfp_names() -> List[str]:
    return [f"ecfp_{i}" for i in range(ECFP_BITS)]

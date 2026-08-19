"""Parse a docked complex into flat numpy arrays.

Deliberately not Biopython's object model: every feature downstream is a
distance query over coordinates, and array access is roughly an order of
magnitude faster than walking Structure/Model/Chain/Residue/Atom for the
~48,000 complexes this has to run over. Biopython is not a dependency at all -
burial.py carries its own Shrake-Rupley, so the whole package needs only
numpy, scipy and RDKit.

The chain contract is the same one the Boltz benchmark enforces and the AF3 +
HADDOCK training structures already satisfy: protein on chain A as ATOM
records, ligand on chain B as HETATM.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Elements that count as apolar for hydrophobic contacts and buried-surface
# splits. Sulfur is included: Met/Cys sulfur packs against apolar ligand atoms
# and contributes like carbon.
APOLAR = frozenset({"C", "S"})
HALOGENS = frozenset({"F", "CL", "BR", "I"})
METALS = frozenset({"ZN", "MG", "MN", "FE", "CA", "NA", "K", "CU", "NI", "CO"})

# Rings that stack. Trp is listed once with its full fused system; treating the
# indole as a single plane is a good enough approximation at 5.5 A.
AROMATIC_RING_ATOMS = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}
# Formally charged side-chain groups at physiological pH.
CATIONIC_ATOMS = {"LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2"), "HIS": ("ND1", "NE2")}
ANIONIC_ATOMS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}

AMINO_ACIDS = (
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR"
).split()


@dataclass
class Complex:
    """Coordinates and per-atom annotation for one protein-ligand complex."""

    # protein
    p_xyz: np.ndarray          # (N, 3) float32
    p_element: np.ndarray      # (N,)   '<U2', uppercase
    p_atom_name: np.ndarray    # (N,)   '<U4'
    p_res_name: np.ndarray     # (N,)   '<U3'
    p_res_id: np.ndarray       # (N,)   int32, residue sequence number
    # ligand
    l_xyz: np.ndarray
    l_element: np.ndarray
    l_atom_name: np.ndarray

    @property
    def n_protein_atoms(self) -> int:
        return len(self.p_xyz)

    @property
    def n_ligand_atoms(self) -> int:
        return len(self.l_xyz)


def _element_of(line: str) -> str:
    """Element from columns 77-78, falling back to the atom name.

    Boltz and HADDOCK both fill the element column, but hand-edited and
    converted files often do not, and guessing from the atom name mis-types
    two-letter elements (CL vs C) unless the leading digit/space is handled.
    """
    element = line[76:78].strip().upper()
    if element:
        return element
    name = line[12:16].strip()
    return "".join(c for c in name if c.isalpha())[:1].upper()


def read_complex(
    path: str | Path,
    protein_chain: str = "A",
    ligand_chain: str = "B",
) -> Complex:
    """Read a PDB complex. Raises ValueError if either chain is empty."""
    p_xyz, p_el, p_name, p_res, p_rid = [], [], [], [], []
    l_xyz, l_el, l_name = [], [], []

    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain = line[21]
            if chain not in (protein_chain, ligand_chain):
                continue
            # Altloc: keep only the primary conformer, else atoms are double
            # counted and every contact feature inflates.
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            element = _element_of(line)
            if element == "H" or element == "D":     # structures are heavy-atom only
                continue
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            if chain == protein_chain:
                p_xyz.append(xyz)
                p_el.append(element)
                p_name.append(line[12:16].strip())
                p_res.append(line[17:20].strip())
                p_rid.append(int(line[22:26]))
            else:
                l_xyz.append(xyz)
                l_el.append(element)
                l_name.append(line[12:16].strip())

    if not p_xyz:
        raise ValueError(f"no protein atoms on chain {protein_chain!r} in {path}")
    if not l_xyz:
        raise ValueError(f"no ligand atoms on chain {ligand_chain!r} in {path}")

    return Complex(
        p_xyz=np.asarray(p_xyz, dtype=np.float32),
        p_element=np.asarray(p_el, dtype="<U2"),
        p_atom_name=np.asarray(p_name, dtype="<U4"),
        p_res_name=np.asarray(p_res, dtype="<U3"),
        p_res_id=np.asarray(p_rid, dtype=np.int32),
        l_xyz=np.asarray(l_xyz, dtype=np.float32),
        l_element=np.asarray(l_el, dtype="<U2"),
        l_atom_name=np.asarray(l_name, dtype="<U4"),
    )


def pocket_mask(cx: Complex, cutoff: float = 10.0) -> np.ndarray:
    """Boolean mask over protein atoms within ``cutoff`` of any ligand atom.

    Used to shrink the protein before the expensive per-atom work (SASA in
    particular): a 1951-residue target is ~15,000 atoms, of which a few hundred
    are anywhere near the ligand.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(cx.p_xyz)
    near = tree.query_ball_point(cx.l_xyz, r=cutoff)
    idx = np.unique(np.concatenate([np.asarray(n, dtype=int) for n in near])) if any(
        len(n) for n in near
    ) else np.array([], dtype=int)
    mask = np.zeros(len(cx.p_xyz), dtype=bool)
    mask[idx] = True
    return mask


def residue_keys(cx: Complex) -> np.ndarray:
    """Stable per-atom residue identifier, for counting distinct residues."""
    return np.char.add(np.char.add(cx.p_res_name, "_"), cx.p_res_id.astype(str))

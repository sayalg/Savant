"""Assemble the full v2 feature vector for one complex.

Four blocks, in this order:

  ligand        11 physicochemical descriptors from SMILES
  interaction   typed contacts, per-residue-type counts, element-pair shells
  burial        SASA-derived buried surface, split apolar/polar
  ecfp          512-bit ECFP4

Everything is CPU, no minimisation, no force field. A complex takes roughly a
second, against minutes for the v1 HADDOCK path, and needs nothing outside
numpy/scipy/RDKit.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from savant2.burial import burial_features
from savant2.interactions import interaction_features
from savant2.ligand import (DESCRIPTOR_NAMES, ecfp, ecfp_names, ligand_atom_types,
                            ligand_descriptors, ligand_mol)
from savant2.structure import read_complex


def featurize(
    pdb_path: str | Path,
    smiles: str,
    protein_chain: str = "A",
    ligand_chain: str = "B",
    include_ecfp: bool = True,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Return ``(features, meta)`` for one complex.

    ``meta`` carries what the CLI needs to explain a prediction and what the
    extraction jobs need to audit a run: whether the SMILES template matched,
    atom counts, and wall time.
    """
    started = time.perf_counter()
    cx = read_complex(pdb_path, protein_chain, ligand_chain)

    mol = ligand_mol(pdb_path, ligand_chain, smiles)
    types = ligand_atom_types(mol, cx.n_ligand_atoms)
    # A template mismatch leaves every role mask empty; the interaction block
    # still runs on element typing, but the ionic/aromatic features go to zero
    # and the row should be flagged rather than trusted.
    typed = bool(mol is not None and mol.GetNumAtoms() == cx.n_ligand_atoms)

    features: Dict[str, float] = {}
    features.update(ligand_descriptors(smiles))
    features.update(interaction_features(cx, types))
    features.update(burial_features(cx))
    if include_ecfp:
        fp = ecfp(smiles)
        features.update({name: float(v) for name, v in zip(ecfp_names(), fp)})

    meta = {
        "n_protein_atoms": cx.n_protein_atoms,
        "n_ligand_atoms": cx.n_ligand_atoms,
        "ligand_typed": typed,
        "seconds": round(time.perf_counter() - started, 3),
    }
    return features, meta


def feature_names(include_ecfp: bool = True) -> List[str]:
    """Canonical feature order, taken from a real extraction on a tiny synthetic
    complex so the order can never drift from what ``featurize`` produces."""
    from savant2.testing import synthetic_complex

    with synthetic_complex() as (path, smiles):
        features, _ = featurize(path, smiles, include_ecfp=include_ecfp)
    return list(features)


def structural_feature_names() -> List[str]:
    """Feature names excluding the ECFP block - the interpretable ones."""
    return [n for n in feature_names(include_ecfp=False)]


def to_vector(features: Dict[str, float], names: List[str]) -> np.ndarray:
    """Order a feature dict into the canonical vector, NaN for anything absent."""
    return np.asarray([features.get(n, np.nan) for n in names], dtype=np.float32)

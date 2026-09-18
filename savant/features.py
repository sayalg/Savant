"""The feature blocks, and the one order they are allowed to go in.

BLOCK ORDER IS LOAD-BEARING. :func:`build_matrix` checks the widths, but that is
a backstop and not a guarantee: two different orders can add up to the same
width, and a wrong order scores silently and wrongly rather than raising. The
order below is the order `ML_model/train_savant_full_dedup.ipynb` built the
training matrix in:

    ChemBERTa ligand embedding   384
    RDKit named descriptors      226   (the exact column list comes from the bundle)
    ECFP4 count fingerprint     1024
    ESM2 protein embedding      1280 | 320
    AA composition + length       21
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from savant.encoders import encode_ligands, encode_proteins

AA = "ACDEFGHIKLMNPQRSTVWY"

## Training computed these 11 on the hydrogen-added molecule and prefixed them
## `h_`. Two of them disagree with their bare-molecule values, so the AddHs step
## is not cosmetic.
H11_COLS = ["h_molecular_weight", "h_logP", "h_num_h_donors", "h_num_h_acceptors",
            "h_num_rotatable_bonds", "h_tpsa", "h_num_rings", "h_num_aromatic_rings",
            "h_num_atoms", "h_num_heavy_atoms", "h_formal_charge"]

ECFP_BITS = 1024
ECFP_RADIUS = 2          # radius 2 == ECFP4


def parse_smiles(smiles):
    """RDKit molecules, raising on the first SMILES that will not parse.

    Raising rather than dropping is deliberate: a silently skipped ligand makes
    the output rows stop lining up with the input rows.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    mols = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"RDKit could not parse SMILES: {s!r}")
        mols.append(m)
    return mols


def _descriptors_addhs(mols):
    """(n, 11) the AddHs variants, in H11_COLS order."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    rows = []
    for m in mols:
        m = Chem.AddHs(m)
        rows.append([Descriptors.MolWt(m), Crippen.MolLogP(m),
                     Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
                     Lipinski.NumRotatableBonds(m), rdMolDescriptors.CalcTPSA(m),
                     rdMolDescriptors.CalcNumRings(m),
                     rdMolDescriptors.CalcNumAromaticRings(m), m.GetNumAtoms(),
                     m.GetNumHeavyAtoms(), Chem.GetFormalCharge(m)])
    return np.asarray(rows, dtype=np.float32)


def descriptors_named(mols, columns):
    """(n, len(columns)) named RDKit descriptors, in the bundle's column order.

    Cast to float32 FIRST and clean afterwards: some descriptors (Ipc, Kappa2)
    only overflow to inf on the cast, so cleaning before it leaves infs behind
    and the booster sees them.
    """
    from rdkit.Chem import Descriptors

    d = pd.DataFrame([Descriptors.CalcMolDescriptors(m) for m in mols]).astype(np.float32)
    d = d.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d = pd.concat([d, pd.DataFrame(_descriptors_addhs(mols), columns=H11_COLS)], axis=1)

    missing = [c for c in columns if c not in d.columns]
    if missing:
        raise ValueError(
            f"this RDKit build does not produce {len(missing)} descriptor(s) the "
            f"model was trained on, e.g. {missing[:5]}. Score on a machine whose "
            f"RDKit matches the one that trained the bundle."
        )
    return d[columns].to_numpy(np.float32)


def ecfp4(mols, n_bits=ECFP_BITS):
    """(n, 1024) Morgan count fingerprint at radius 2."""
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=n_bits)
    return np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)


def aa_composition(seqs):
    """(n, 21): the 20 standard residue fractions, then raw sequence length.

    A non-standard residue (X, B, U, Z) contributes to the length but to no
    fraction, which is what training did.
    """
    return np.array([[s.count(a) / len(s) for a in AA] + [len(s)] for s in seqs],
                    dtype=np.float32)


def build_matrix(seqs, smiles, rdkit_columns, esm_model,
                 block_widths=None, n_features=None,
                 ligand_batch=64, protein_batch=8):
    """(n, n_features) in the trained block order.

    `seqs` and `smiles` are parallel and equal-length: one row per pair. The
    caller is responsible for repeating a target's sequence across its ligands.
    """
    if len(seqs) != len(smiles):
        raise ValueError(f"{len(seqs)} sequences against {len(smiles)} SMILES; "
                         "they must be parallel, one row per pair")

    mols = parse_smiles(smiles)
    blocks = [
        encode_ligands(smiles, ligand_batch),
        descriptors_named(mols, rdkit_columns),
        ecfp4(mols),
        encode_proteins(seqs, esm_model, protein_batch),
        aa_composition(seqs),
    ]

    widths = [b.shape[1] for b in blocks]
    if block_widths is not None and widths != list(block_widths):
        raise ValueError(
            f"block widths {widths} do not match the trained layout "
            f"{list(block_widths)}; the RDKit column list or an encoder does not "
            f"match the bundle."
        )

    X = np.concatenate(blocks, axis=1)
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"built {X.shape[1]} features, model expects {n_features}")
    return X

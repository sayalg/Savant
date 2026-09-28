"""Ligand graph: atoms, bonds, ring-centroid pseudo-sites, and per-atom priors.

The ligand side of the model is a graph encoder over heavy atoms plus one pseudo
atom per SSSR ring (PLAN.md Section 6.1, for the pi terms). Uni-Mol atom embeddings
can be concatenated when cached, but the graph encoder is not optional - a ring
centroid has no Uni-Mol vector of its own.

`atom_param_prior` is the part that matters physically. `physics/sites.py` has an
element table, but RDKit knows the protonation state, formal charge, hybridisation
and explicit-H count of *this* molecule, which an element table cannot. Those
per-atom values override the element prior in `LigandSiteParams`, so a carboxylate
oxygen and an ester oxygen do not start from the same charge.

One invariance requirement from Section 6.5: the featurisation must be invariant to
SMILES randomisation and to atom permutation. Atom order here follows RDKit's
canonical ranking rather than input order, and `tests/test_physics.py` checks it.
"""
from __future__ import annotations

import numpy as np

from ..physics.sites import (ELEM_IDX, N_SITE_PARAMS, P_ACC, P_DONOR, P_HAL,
                             P_PHOB, P_Q, P_AROM)
from ..preflight import require_rdkit

HYBRIDIZATIONS = ("SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED", "S")
BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

ATOM_FEAT_NAMES = (
    [f"elem_{e}" for e in ELEM_IDX]
    + [f"hyb_{h}" for h in HYBRIDIZATIONS]
    + ["degree", "formal_charge", "n_hs", "is_aromatic", "in_ring",
       "ring_size_3", "ring_size_4", "ring_size_5", "ring_size_6", "ring_size_7p",
       "is_donor", "is_acceptor", "gasteiger", "crippen_logp", "crippen_mr",
       "is_ring_centroid", "n_ring_atoms"]
)
ATOM_DIM = len(ATOM_FEAT_NAMES)
## +1 for "no bond" and +1 for "ring membership edge"; the explicit no-bond channel
## matters because the adjacency is dense and a zero row would be ambiguous with
## padding.
BOND_DIM = len(BOND_TYPES) + 4


def _onehot(value, options) -> list[float]:
    return [1.0 if value == o else 0.0 for o in options]


def canonical_mol(smiles: str):
    """The heavy-atom molecule in canonical atom order, or None.

    Canonical order makes the featurisation invariant to how the SMILES was
    written, which Section 6.5 requires and which no amount of model capacity fixes
    if it is violated. It is ALSO the contract with the ligand-physics cache: site k
    of the graph is atom k of this molecule (then its SSSR rings, in RingInfo order),
    and `ligand_phys.site_distances` builds its matrix in exactly that order. Both
    sides call this function so the two cannot drift apart.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    order = list(Chem.CanonicalRankAtoms(mol))
    return Chem.RenumberAtoms(mol, [order.index(i) for i in range(mol.GetNumAtoms())])


def build_graph(smiles: str, max_atoms: int = 64, add_ring_centroids: bool = True):
    """SMILES -> dict of arrays for the model, or None if RDKit cannot read it.

    Returns
    -------
    atom_feat  (A, ATOM_DIM)     A = heavy atoms (+ ring centroids), padded by caller
    adj        (A, A)            1 for a covalent bond or a centroid-to-member edge
    bond_feat  (A, A, BOND_DIM)
    elem_idx   (A,)              index into physics.sites.LIGAND_ELEMENTS
    param_prior(A, N_SITE_PARAMS) NaN where the element table should win
    ring_flag  (A,)              1 for a ring-centroid pseudo-site
    """
    require_rdkit("ligand graph construction")
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Crippen, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")

    mol = canonical_mol(smiles)
    if mol is None:
        return None

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    crippen = Crippen._GetAtomContribs(mol)
    donors = {i for m in mol.GetSubstructMatches(
        Chem.MolFromSmarts("[$([N;!H0]),$([O,S;H1])]")) for i in m}
    acceptors = {i for m in mol.GetSubstructMatches(
        Chem.MolFromSmarts("[$([O,S;H0;v2]),$([N;v3;H0]),$([n;+0])]")) for i in m}

    n_heavy = mol.GetNumAtoms()
    rings = [r for r in mol.GetRingInfo().AtomRings()] if add_ring_centroids else []
    n = n_heavy + len(rings)
    if n > max_atoms:
        ## Truncating a molecule silently changes what is being modelled, so it is
        ## reported by the caller rather than hidden here.
        rings = rings[:max(max_atoms - n_heavy, 0)]
        n = min(n_heavy + len(rings), max_atoms)

    af = np.zeros((n, ATOM_DIM), np.float32)
    elem = np.full(n, ELEM_IDX["*"], np.int16)
    prior = np.full((n, N_SITE_PARAMS), np.nan, np.float32)
    ring_flag = np.zeros(n, np.float32)

    ri = mol.GetRingInfo()
    for i in range(min(n_heavy, n)):
        a = mol.GetAtomWithIdx(i)
        sym = a.GetSymbol()
        sizes = [ri.IsAtomInRingOfSize(i, s) for s in (3, 4, 5, 6)]
        try:
            gast = float(a.GetDoubleProp("_GasteigerCharge"))
            gast = gast if np.isfinite(gast) else 0.0
        except Exception:
            gast = 0.0
        af[i] = (_onehot(sym, list(ELEM_IDX))
                 + _onehot(str(a.GetHybridization()), HYBRIDIZATIONS)
                 + [a.GetDegree(), a.GetFormalCharge(), a.GetTotalNumHs(),
                    float(a.GetIsAromatic()), float(a.IsInRing()),
                    *[float(s) for s in sizes],
                    float(a.IsInRing() and not any(sizes)),
                    float(i in donors), float(i in acceptors), gast,
                    crippen[i][0], crippen[i][1], 0.0, 0.0])
        elem[i] = ELEM_IDX.get(sym, ELEM_IDX["*"])
        ## Per-atom overrides of the element prior. Everything left NaN falls back.
        prior[i, P_DONOR] = 1.0 if i in donors else 0.0
        prior[i, P_ACC] = 1.0 if i in acceptors else 0.0
        prior[i, P_Q] = float(np.clip(gast, -1.0, 1.0))
        prior[i, P_AROM] = float(a.GetIsAromatic())
        prior[i, P_PHOB] = float(np.clip((crippen[i][0] + 0.2) / 0.5, 0.0, 1.0))
        if sym in ("Cl", "Br", "I"):
            prior[i, P_HAL] = {"Cl": 0.7, "Br": 0.9, "I": 1.0}[sym]

    for k, ring in enumerate(rings):
        j = n_heavy + k
        if j >= n:
            break
        af[j, ATOM_FEAT_NAMES.index("is_ring_centroid")] = 1.0
        af[j, ATOM_FEAT_NAMES.index("n_ring_atoms")] = len(ring)
        af[j, ATOM_FEAT_NAMES.index("is_aromatic")] = float(
            all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
        ring_flag[j] = 1.0
        ## A ring centroid carries ONLY the aromatic channel. Giving it vdW as well
        ## would double-count the ring atoms it sits on top of.
        prior[j] = np.nan
        prior[j, P_AROM] = af[j, ATOM_FEAT_NAMES.index("is_aromatic")]

    adj = np.zeros((n, n), np.float32)
    bf = np.zeros((n, n, BOND_DIM), np.float32)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i >= n or j >= n:
            continue
        feat = (_onehot(str(b.GetBondType()), BOND_TYPES)
                + [float(b.GetIsConjugated()), float(b.IsInRing()),
                   float(b.GetStereo() != Chem.BondStereo.STEREONONE), 0.0])
        adj[i, j] = adj[j, i] = 1.0
        bf[i, j] = bf[j, i] = feat
    for k, ring in enumerate(rings):
        j = n_heavy + k
        if j >= n:
            break
        for i in ring:
            if i < n:
                adj[i, j] = adj[j, i] = 1.0
                bf[i, j, -1] = bf[j, i, -1] = 1.0     # "ring membership" channel

    return dict(atom_feat=af, adj=adj, bond_feat=bf, elem_idx=elem,
                param_prior=prior, ring_flag=ring_flag,
                n_heavy=min(n_heavy, n), n_sites=n,
                truncated=bool(n_heavy + len(ri.AtomRings()) > max_atoms))


def pad_batch(graphs: list[dict], max_atoms: int | None = None) -> dict:
    """Pad a list of graphs into dense batch tensors with a validity mask."""
    a = max_atoms or max(g["n_sites"] for g in graphs)
    b = len(graphs)
    out = {
        "atom_feat": np.zeros((b, a, ATOM_DIM), np.float32),
        "adj": np.zeros((b, a, a), np.float32),
        "bond_feat": np.zeros((b, a, a, BOND_DIM), np.float32),
        "elem_idx": np.full((b, a), ELEM_IDX["*"], np.int64),
        "atom_param_prior": np.full((b, a, N_SITE_PARAMS), np.nan, np.float32),
        "is_ring_centroid": np.zeros((b, a), np.float32),
        "lig_mask": np.zeros((b, a), np.float32),
    }
    for i, g in enumerate(graphs):
        n = min(g["n_sites"], a)
        out["atom_feat"][i, :n] = g["atom_feat"][:n]
        out["adj"][i, :n, :n] = g["adj"][:n, :n]
        out["bond_feat"][i, :n, :n] = g["bond_feat"][:n, :n]
        out["elem_idx"][i, :n] = g["elem_idx"][:n]
        out["atom_param_prior"][i, :n] = g["param_prior"][:n]
        out["is_ring_centroid"][i, :n] = g["ring_flag"][:n]
        out["lig_mask"][i, :n] = 1.0
    return out

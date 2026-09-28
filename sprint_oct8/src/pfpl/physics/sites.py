"""Interaction sites, and the tabulated force-field values the heads start from.

PLAN.md Section 6.1. Because there is no pose at inference, a "site" is a *typed
location* rather than a coordinate: the model predicts a distance distribution
between each protein site and each ligand site, and the physics is evaluated as an
expectation under it.

Protein, per residue (three pseudo-sites, plus a ring centroid for F/Y/W/H):

    0 BB    backbone / Calpha      - always present
    1 SC    side-chain centroid    - masked for GLY
    2 TIP   side-chain terminal    - masked for GLY and ALA
    3 RING  aromatic ring centroid - only F, Y, W, H

Ligand: every heavy atom, plus one centroid per SSSR ring (Section 6.1, for the
pi terms).

The numbers below are *priors*, not constants. Section 6.4 asks for initialisation
and annealed regularisation toward tabulated force-field values; `priors.py`
consumes this table and `params.py` predicts departures from it. They are
AMBER-ff14SB-flavoured group values, deliberately coarse - the model is residue
level, so a per-atom charge set would be false precision. Provenance for each
column is in the comment above it; none of it is fitted to this project's labels.
"""
from __future__ import annotations

import numpy as np

SITE_KINDS = ("BB", "SC", "TIP", "RING")
BB, SC, TIP, RING = range(4)

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
UNK = len(AA)  # 20 = unknown residue

## Parameters every site carries. Order is load-bearing: params.py emits a tensor
## with this many channels and priors.py indexes it by the same order.
SITE_PARAMS = ("eps", "r0", "donor", "acceptor", "hydrophobic",
               "charge", "aromatic", "cation", "halogen")
P_EPS, P_R0, P_DONOR, P_ACC, P_PHOB, P_Q, P_AROM, P_CAT, P_HAL = range(len(SITE_PARAMS))
N_SITE_PARAMS = len(SITE_PARAMS)

## Physically sane ranges. params.py squashes into these, so a head cannot emit a
## 40 kcal/mol well depth to fit a noisy label.
PARAM_RANGE = {
    "eps": (0.02, 0.60),      # kcal/mol, LJ well depth per site pair (geometric mix)
    "r0": (2.80, 5.20),       # A, LJ minimum (Lorentz-Berthelot arithmetic mix)
    "donor": (0.0, 1.0),
    "acceptor": (0.0, 1.0),
    "hydrophobic": (0.0, 1.0),
    "charge": (-1.0, 1.0),    # e, group charge
    "aromatic": (0.0, 1.0),
    "cation": (0.0, 1.0),
    "halogen": (0.0, 1.0),
}

## Side-chain group charge at pH 7.4 (ff14SB residue net charge, placed on TIP).
_CHARGE = dict(D=-1.0, E=-1.0, K=1.0, R=1.0, H=0.1)
## Kyte-Doolittle hydropathy, rescaled to [0, 1].
_KD_HYDRO = dict(A=1.8, C=2.5, D=-3.5, E=-3.5, F=2.8, G=-0.4, H=-3.2, I=4.5,
                 K=-3.9, L=3.8, M=1.9, N=-3.5, P=-1.6, Q=-3.5, R=-4.5, S=-0.8,
                 T=-0.7, V=4.2, W=-0.9, Y=-1.3)
## Side-chain H-bond donors / acceptors (counts, capped to 1.0 as a propensity).
_DONOR = dict(N=1.0, Q=1.0, K=1.0, R=1.0, W=1.0, S=1.0, T=1.0, Y=1.0, H=0.5, C=0.3)
_ACCEPT = dict(D=1.0, E=1.0, N=1.0, Q=1.0, S=1.0, T=1.0, Y=1.0, H=0.5, M=0.3)
_AROMATIC = dict(F=1.0, Y=1.0, W=1.0, H=0.6)
_CATION = dict(K=1.0, R=1.0, H=0.1)
## Approximate side-chain volume (A^3, Zamyatnin) -> a radius prior via (3V/4pi)^(1/3).
_VOL = dict(A=88.6, C=108.5, D=111.1, E=138.4, F=189.9, G=60.1, H=153.2, I=166.7,
            K=168.6, L=166.7, M=162.9, N=114.1, P=112.7, Q=143.8, R=173.4, S=89.0,
            T=116.1, V=140.0, W=227.8, Y=193.6)


def _hydro01(a: str) -> float:
    return float(np.clip((_KD_HYDRO.get(a, 0.0) + 4.5) / 9.0, 0.0, 1.0))


def _r0_from_volume(a: str) -> float:
    v = _VOL.get(a, 140.0)
    r = (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0)       # ~2.4-3.8 A
    return float(np.clip(r + 0.6, *PARAM_RANGE["r0"]))


def residue_site_priors() -> np.ndarray:
    """(21 residues, 4 site kinds, N_SITE_PARAMS) prior parameter table."""
    t = np.zeros((len(AA) + 1, len(SITE_KINDS), N_SITE_PARAMS), np.float32)
    for a in AA:
        i = AA_IDX[a]
        # ---- backbone: same for every residue; N-H donor and C=O acceptor
        t[i, BB] = [0.12, 3.40, 1.0, 1.0, 0.35, 0.0, 0.0, 0.0, 0.0]
        # ---- side-chain centroid: bulk vdW and hydrophobicity live here
        t[i, SC] = [0.15, _r0_from_volume(a), 0.0, 0.0, _hydro01(a),
                    0.0, _AROMATIC.get(a, 0.0), 0.0, 0.0]
        # ---- terminal functional atom: polarity, charge, H-bonding
        t[i, TIP] = [0.16, 3.20, _DONOR.get(a, 0.0), _ACCEPT.get(a, 0.0),
                     _hydro01(a) * 0.5, _CHARGE.get(a, 0.0), 0.0,
                     _CATION.get(a, 0.0), 0.0]
        # ---- aromatic ring centroid
        t[i, RING] = [0.20, 3.90, 0.0, 0.0, 0.8, 0.0, _AROMATIC.get(a, 0.0), 0.0, 0.0]
    ## unknown residue: the neutral middle of every range
    t[UNK, BB] = [0.12, 3.40, 1.0, 1.0, 0.35, 0.0, 0.0, 0.0, 0.0]
    t[UNK, SC] = [0.15, 3.60, 0.2, 0.2, 0.50, 0.0, 0.1, 0.0, 0.0]
    t[UNK, TIP] = [0.16, 3.20, 0.2, 0.2, 0.25, 0.0, 0.0, 0.05, 0.0]
    t[UNK, RING] = [0.20, 3.90, 0.0, 0.0, 0.80, 0.0, 0.1, 0.0, 0.0]
    return t


def residue_site_mask() -> np.ndarray:
    """(21, 4) bool: which pseudo-sites exist for each residue.

    GLY has no side chain and ALA has no distinct terminal atom; emitting sites for
    them would let the model place H-bond capacity on groups that do not exist.
    """
    m = np.zeros((len(AA) + 1, len(SITE_KINDS)), bool)
    m[:, BB] = True
    m[:, SC] = True
    m[:, TIP] = True
    m[AA_IDX["G"], SC] = False
    m[AA_IDX["G"], TIP] = False
    m[AA_IDX["A"], TIP] = False
    for a in "FYWH":
        m[AA_IDX[a], RING] = True
    return m


def encode_sequence(seq: str) -> np.ndarray:
    """Residue letters -> indices into the tables above; anything odd becomes UNK."""
    return np.array([AA_IDX.get(c, UNK) for c in seq.upper()], np.int16)


# ------------------------------------------------------------------ ligand side

## Element-wise priors for ligand heavy atoms. eps/r0 are GAFF2-flavoured; donor and
## acceptor are refined per atom by RDKit in features/embed_ligand.py, which knows
## about explicit hydrogens and formal charge. These are the fallback.
LIGAND_ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Se", "*")
ELEM_IDX = {e: i for i, e in enumerate(LIGAND_ELEMENTS)}
ELEM_UNK = ELEM_IDX["*"]

_ELEM_PRIOR = {
    #      eps    r0   don  acc  phob   q    arom  cat   hal
    "C":  (0.11, 3.82, 0.0, 0.0, 1.00, 0.00, 0.0, 0.0, 0.0),
    "N":  (0.17, 3.60, 0.7, 0.7, 0.15, -0.30, 0.0, 0.0, 0.0),
    "O":  (0.21, 3.35, 0.3, 1.0, 0.05, -0.45, 0.0, 0.0, 0.0),
    "S":  (0.25, 4.00, 0.1, 0.3, 0.70, -0.10, 0.0, 0.0, 0.0),
    "P":  (0.20, 4.20, 0.0, 0.3, 0.30, 0.30, 0.0, 0.0, 0.0),
    "F":  (0.06, 3.20, 0.0, 0.2, 0.60, -0.20, 0.0, 0.0, 0.1),
    "Cl": (0.27, 3.90, 0.0, 0.1, 0.85, -0.10, 0.0, 0.0, 0.7),
    "Br": (0.39, 4.10, 0.0, 0.1, 0.90, -0.08, 0.0, 0.0, 0.9),
    "I":  (0.50, 4.40, 0.0, 0.1, 0.95, -0.05, 0.0, 0.0, 1.0),
    "B":  (0.18, 3.80, 0.0, 0.2, 0.40, 0.10, 0.0, 0.0, 0.0),
    "Se": (0.30, 4.20, 0.1, 0.3, 0.75, -0.08, 0.0, 0.0, 0.2),
    "*":  (0.15, 3.70, 0.1, 0.2, 0.50, 0.00, 0.0, 0.0, 0.0),
}


def ligand_element_priors() -> np.ndarray:
    t = np.zeros((len(LIGAND_ELEMENTS), N_SITE_PARAMS), np.float32)
    for e, i in ELEM_IDX.items():
        t[i] = _ELEM_PRIOR[e]
    return t


## Ring centroids are pseudo-atoms appended after the heavy atoms; they carry the
## aromatic channel and nothing else, so a benzene does not double-count its vdW.
RING_CENTROID_PRIOR = np.array([0.20, 3.90, 0.0, 0.0, 0.85, 0.0, 1.0, 0.0, 0.0],
                               dtype=np.float32)

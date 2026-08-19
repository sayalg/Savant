"""Typed protein-ligand interaction fingerprint.

This is what replaces HADDOCK3 in v2. The v1 pipeline spent minutes per complex
running topology generation and energy minimisation to produce six scalars
(``vdw``, ``elec``, ``desolv``, ``bsa``, ``score``, ``total``), and an ablation
on the 1000-pair benchmark showed those six add +0.012 Pearson r over a plain
distance-based contact count - i.e. nothing outside the noise. The information
is not missing from the structure, it is destroyed by summing an interface down
to single energies.

So instead of summing, this enumerates: which interaction types occur, how many
of each, at what distance, and against which residue types. Every feature is a
KD-tree query, so a complex takes milliseconds and needs no minimisation, no
force field, and no Linux-only dependency.

Interaction geometry follows the criteria PLIP and ProLIF use, with
distance-and-element typing rather than requiring explicit hydrogens - the
structures coming out of AlphaFold, Boltz and HADDOCK are heavy-atom only, and
adding hydrogens would put a protonation-prediction step (and its errors) in
front of every prediction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from savant2.structure import (AMINO_ACIDS, ANIONIC_ATOMS, APOLAR, AROMATIC_RING_ATOMS,
                               CATIONIC_ATOMS, HALOGENS, METALS, Complex, residue_keys)

# Distance criteria (angstroms). Hydrophobic and hydrogen-bond cutoffs follow
# PLIP; ionic and cation-pi are centroid/charge-group distances so they run
# longer; the pi-stacking cutoff is centroid-to-centroid.
CUT_HYDROPHOBIC = 4.5
CUT_HBOND = 3.6
CUT_IONIC = 5.0
CUT_PISTACK = 5.5
CUT_CATIONPI = 6.0
CUT_HALOGEN = 3.7
CUT_METAL = 3.0
CUT_CONTACT = 4.5          # generic "in contact" for per-residue counting
CUT_POCKET = 8.0           # pocket membership for composition features

# Element-pair shell counts: a strict superset of what PRODIGY-LIG sees (it
# counts C/N/O/X pairs at a single 10.5 A cutoff and feeds a linear model).
SHELLS = (4.0, 6.0)
LIG_ELEMENT_CLASSES = ("C", "N", "O", "S", "X")     # X = halogen
PROT_ELEMENT_CLASSES = ("C", "N", "O", "S")

PI_PARALLEL_DEG = 30.0     # ring planes within this angle -> face-to-face
PI_TILTED_DEG = 60.0       # beyond this -> edge-to-face / T-shaped


def _element_class(elements: np.ndarray, ligand: bool) -> np.ndarray:
    """Map raw elements onto the small class vocabulary used for shell counts."""
    out = np.full(len(elements), "other", dtype="<U5")
    for cls in ("C", "N", "O", "S"):
        out[elements == cls] = cls
    if ligand:
        out[np.isin(elements, list(HALOGENS))] = "X"
    return out


def _ring_centroids_and_normals(
    xyz: np.ndarray, rings: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Centroid and unit normal for each ring, normals via SVD on the ring atoms."""
    if not rings:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    centroids, normals = [], []
    for idx in rings:
        pts = xyz[idx]
        centroid = pts.mean(axis=0)
        # Smallest singular vector of the centred ring is the plane normal.
        _, _, vh = np.linalg.svd(pts - centroid)
        centroids.append(centroid)
        normals.append(vh[2])
    return np.asarray(centroids, np.float32), np.asarray(normals, np.float32)


def _protein_rings(cx: Complex) -> List[np.ndarray]:
    keys = residue_keys(cx)
    rings = []
    for res_name, atom_names in AROMATIC_RING_ATOMS.items():
        sel = cx.p_res_name == res_name
        if not sel.any():
            continue
        for key in np.unique(keys[sel]):
            in_res = keys == key
            idx = np.where(in_res & np.isin(cx.p_atom_name, atom_names))[0]
            if len(idx) >= 5:          # tolerate one missing atom
                rings.append(idx)
    return rings


def _protein_group(cx: Complex, table: dict) -> np.ndarray:
    """Indices of protein atoms belonging to a named side-chain group table."""
    hits = np.zeros(len(cx.p_xyz), dtype=bool)
    for res_name, atom_names in table.items():
        hits |= (cx.p_res_name == res_name) & np.isin(cx.p_atom_name, atom_names)
    return np.where(hits)[0]


def _pair_stats(a_xyz: np.ndarray, b_xyz: np.ndarray, cutoff: float) -> Tuple[int, float]:
    """(number of pairs within cutoff, minimum pair distance) — 0/0.0 if empty."""
    if len(a_xyz) == 0 or len(b_xyz) == 0:
        return 0, 0.0
    pairs = cKDTree(a_xyz).query_ball_tree(cKDTree(b_xyz), r=cutoff)
    count = sum(len(p) for p in pairs)
    if count == 0:
        return 0, 0.0
    dmin = min(
        float(np.linalg.norm(a_xyz[i] - b_xyz[j]))
        for i, js in enumerate(pairs) for j in js
    )
    return count, dmin


def interaction_features(
    cx: Complex,
    ligand_types: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """Full interaction block for one complex.

    ``ligand_types`` comes from :func:`savant2.ligand.ligand_atom_types` and
    carries RDKit-derived per-atom roles (donor, acceptor, aromatic ring
    membership, formal charge). Without it the ligand is typed by element alone,
    which loses aromaticity and charge - accepted as a fallback so a complex
    with an unparseable ligand still scores rather than failing.
    """
    f: Dict[str, float] = {}
    lig = ligand_types or {}
    n_heavy = max(cx.n_ligand_atoms, 1)

    # --- generic proximity -------------------------------------------------
    p_tree = cKDTree(cx.p_xyz)
    l_tree = cKDTree(cx.l_xyz)
    keys = residue_keys(cx)

    contact_pairs = l_tree.query_ball_tree(p_tree, r=CUT_CONTACT)
    contact_atoms = np.unique(np.concatenate(contact_pairs)).astype(int) if any(
        len(p) for p in contact_pairs) else np.array([], dtype=int)
    pocket_pairs = l_tree.query_ball_tree(p_tree, r=CUT_POCKET)
    pocket_atoms = np.unique(np.concatenate(pocket_pairs)).astype(int) if any(
        len(p) for p in pocket_pairs) else np.array([], dtype=int)

    contact_res = np.unique(keys[contact_atoms]) if len(contact_atoms) else np.array([])
    pocket_res = np.unique(keys[pocket_atoms]) if len(pocket_atoms) else np.array([])

    f["n_interacting_residues"] = float(len(contact_res))
    f["n_pocket_residues"] = float(len(pocket_res))
    f["n_contact_atoms"] = float(len(contact_atoms))
    f["contacts_per_ligand_atom"] = len(contact_atoms) / n_heavy

    dists, _ = p_tree.query(cx.l_xyz, k=1)
    f["min_contact_distance"] = float(dists.min())
    f["mean_closest_distance"] = float(dists.mean())
    # Fraction of ligand atoms making any close contact: a compact measure of
    # how much of the molecule is actually engaged rather than dangling.
    f["frac_ligand_atoms_in_contact"] = float(
        np.mean([len(p) > 0 for p in contact_pairs])
    )

    # --- hydrophobic -------------------------------------------------------
    l_apolar = cx.l_xyz[np.isin(cx.l_element, list(APOLAR))]
    p_apolar = cx.p_xyz[np.isin(cx.p_element, list(APOLAR))]
    count, dmin = _pair_stats(l_apolar, p_apolar, CUT_HYDROPHOBIC)
    f["hydrophobic_contacts"] = float(count)
    f["hydrophobic_per_ligand_atom"] = count / n_heavy
    f["hydrophobic_min_dist"] = dmin

    # --- hydrogen bonds ----------------------------------------------------
    # Heavy-atom N/O proximity. Without hydrogens the donor/acceptor direction
    # cannot be resolved on the protein side, so the ligand's RDKit typing is
    # what splits the two populations.
    p_polar = cx.p_xyz[np.isin(cx.p_element, ["N", "O"])]
    l_polar = cx.l_xyz[np.isin(cx.l_element, ["N", "O", "F"])]
    count, dmin = _pair_stats(l_polar, p_polar, CUT_HBOND)
    f["hbond_contacts"] = float(count)
    f["hbond_per_ligand_atom"] = count / n_heavy
    f["hbond_min_dist"] = dmin

    for role in ("donor", "acceptor"):
        sel = lig.get(role)
        xyz = cx.l_xyz[sel] if sel is not None and len(sel) == cx.n_ligand_atoms else np.zeros((0, 3), np.float32)
        count, dmin = _pair_stats(xyz, p_polar, CUT_HBOND)
        f[f"hbond_ligand_{role}"] = float(count)
        f[f"hbond_ligand_{role}_min_dist"] = dmin

    # --- ionic -------------------------------------------------------------
    p_cation = cx.p_xyz[_protein_group(cx, CATIONIC_ATOMS)]
    p_anion = cx.p_xyz[_protein_group(cx, ANIONIC_ATOMS)]
    for label, key, partner in (("ligand_anion", "anionic", p_cation),
                                ("ligand_cation", "cationic", p_anion)):
        sel = lig.get(key)
        xyz = cx.l_xyz[sel] if sel is not None and len(sel) == cx.n_ligand_atoms else np.zeros((0, 3), np.float32)
        count, dmin = _pair_stats(xyz, partner, CUT_IONIC)
        f[f"ionic_{label}"] = float(count)
        f[f"ionic_{label}_min_dist"] = dmin

    # --- aromatic stacking -------------------------------------------------
    p_rings = _protein_rings(cx)
    p_cent, p_norm = _ring_centroids_and_normals(cx.p_xyz, p_rings)
    l_rings = lig.get("rings") or []
    l_cent, l_norm = _ring_centroids_and_normals(cx.l_xyz, l_rings)

    n_parallel = n_tilted = 0
    pi_min = 0.0
    if len(p_cent) and len(l_cent):
        d = np.linalg.norm(l_cent[:, None, :] - p_cent[None, :, :], axis=2)
        close = np.argwhere(d <= CUT_PISTACK)
        if len(close):
            pi_min = float(d[close[:, 0], close[:, 1]].min())
        for li, pi in close:
            cos = abs(float(np.dot(l_norm[li], p_norm[pi])))
            angle = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
            if angle <= PI_PARALLEL_DEG:
                n_parallel += 1
            elif angle >= PI_TILTED_DEG:
                n_tilted += 1
    f["pi_stack_parallel"] = float(n_parallel)
    f["pi_stack_tilted"] = float(n_tilted)
    f["pi_stack_min_dist"] = pi_min
    f["n_ligand_rings_3d"] = float(len(l_cent))
    f["n_pocket_aromatic_rings"] = float(len(p_cent))

    # --- cation-pi ---------------------------------------------------------
    count, dmin = _pair_stats(p_cation, l_cent, CUT_CATIONPI)
    f["cation_pi_protein_cation"] = float(count)
    f["cation_pi_protein_cation_min_dist"] = dmin
    sel = lig.get("cationic")
    l_cation = cx.l_xyz[sel] if sel is not None and len(sel) == cx.n_ligand_atoms else np.zeros((0, 3), np.float32)
    count, dmin = _pair_stats(l_cation, p_cent, CUT_CATIONPI)
    f["cation_pi_ligand_cation"] = float(count)
    f["cation_pi_ligand_cation_min_dist"] = dmin

    # --- halogen bonds and metal coordination ------------------------------
    l_halogen = cx.l_xyz[np.isin(cx.l_element, list(HALOGENS))]
    count, dmin = _pair_stats(l_halogen, p_polar, CUT_HALOGEN)
    f["halogen_bonds"] = float(count)
    f["halogen_min_dist"] = dmin

    p_metal = cx.p_xyz[np.isin(cx.p_element, list(METALS))]
    count, dmin = _pair_stats(cx.l_xyz[np.isin(cx.l_element, ["N", "O", "S"])],
                              p_metal, CUT_METAL)
    f["metal_coordination"] = float(count)
    f["metal_min_dist"] = dmin

    # --- per-residue-type contacts ----------------------------------------
    # Where protein identity finally enters the feature vector: not "which
    # protein is this" but "what chemistry lines this pocket".
    contact_res_names = (
        np.array([k.split("_")[0] for k in contact_res]) if len(contact_res)
        else np.array([], dtype=str)
    )
    pocket_res_names = (
        np.array([k.split("_")[0] for k in pocket_res]) if len(pocket_res)
        else np.array([], dtype=str)
    )
    n_pocket = max(len(pocket_res_names), 1)
    for aa in AMINO_ACIDS:
        f[f"contact_res_{aa}"] = float((contact_res_names == aa).sum())
        f[f"pocket_frac_{aa}"] = float((pocket_res_names == aa).sum()) / n_pocket

    # --- element-pair shell counts ----------------------------------------
    l_cls = _element_class(cx.l_element, ligand=True)
    p_cls = _element_class(cx.p_element, ligand=False)
    for shell in SHELLS:
        pairs = l_tree.query_ball_tree(p_tree, r=shell)
        counts = {(a, b): 0 for a in LIG_ELEMENT_CLASSES for b in PROT_ELEMENT_CLASSES}
        for li, partners in enumerate(pairs):
            a = l_cls[li]
            if a not in LIG_ELEMENT_CLASSES:
                continue
            for pi in partners:
                b = p_cls[pi]
                if b in PROT_ELEMENT_CLASSES:
                    counts[(a, b)] += 1
        for (a, b), value in counts.items():
            f[f"shell{shell:g}_{a}{b}"] = float(value)

    return f


def interaction_feature_names(sample: Optional[Dict[str, float]] = None) -> List[str]:
    """Canonical ordered feature names, derived from a real extraction so the
    list can never drift from what :func:`interaction_features` emits."""
    if sample is not None:
        return list(sample)
    raise ValueError("pass one extracted feature dict to fix the canonical order")

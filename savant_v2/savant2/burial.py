"""Solvent-accessible surface and burial, replacing HADDOCK's ``bsa``/``desolv``.

A self-contained Shrake-Rupley: numeric SASA is a sphere-sampling problem, and
implementing it directly keeps the dependency list at numpy + scipy and lets the
work be restricted to the atoms that matter. Only ligand atoms and pocket
protein atoms are ever evaluated, so cost is independent of how large the
protein is - a 1951-residue target costs the same as a 90-residue one.

The apolar/polar split is what stands in for a desolvation term: burying
hydrophobic surface is favourable, burying polar surface without compensating
hydrogen bonds is not, and a model can learn that trade-off from the two
numbers far more easily than from a single summed energy.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial import cKDTree

from savant2.structure import APOLAR, Complex, pocket_mask

PROBE_RADIUS = 1.4
N_SPHERE_POINTS = 96

# Bondi van der Waals radii, with the common protein/ligand elements covered
# and a carbon-like default for anything exotic.
VDW_RADII = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47,
    "CL": 1.75, "BR": 1.85, "I": 1.98, "SE": 1.90, "B": 1.92,
    "ZN": 1.39, "MG": 1.73, "MN": 1.61, "FE": 1.56, "CA": 1.95,
    "NA": 2.27, "K": 2.75, "CU": 1.40, "NI": 1.63, "CO": 1.52,
}
DEFAULT_RADIUS = 1.70


def _radii(elements: np.ndarray) -> np.ndarray:
    return np.asarray([VDW_RADII.get(e, DEFAULT_RADIUS) for e in elements], np.float32)


def _sphere_points(n: int) -> np.ndarray:
    """Near-uniform points on the unit sphere (Fibonacci/golden spiral)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    ).astype(np.float32)


_UNIT_SPHERE = _sphere_points(N_SPHERE_POINTS)


def atom_sasa(
    query_xyz: np.ndarray,
    query_elements: np.ndarray,
    context_xyz: np.ndarray,
    context_elements: np.ndarray,
) -> np.ndarray:
    """Per-atom SASA for ``query`` atoms, occluded by ``context`` atoms.

    ``context`` must include the query atoms themselves when self-occlusion
    matters (it always does within one molecule).
    """
    if len(query_xyz) == 0:
        return np.zeros(0, np.float32)
    q_r = _radii(query_elements) + PROBE_RADIUS
    c_r = _radii(context_elements) + PROBE_RADIUS

    tree = cKDTree(context_xyz)
    max_reach = float(q_r.max() + c_r.max())
    areas = np.zeros(len(query_xyz), np.float32)

    for i, (centre, radius) in enumerate(zip(query_xyz, q_r)):
        neighbours = tree.query_ball_point(centre, r=max_reach)
        points = centre + radius * _UNIT_SPHERE
        accessible = np.ones(len(points), dtype=bool)
        for j in neighbours:
            d = np.linalg.norm(context_xyz[j] - centre)
            if d < 1e-6:                      # the atom itself
                continue
            if d > radius + c_r[j]:
                continue
            accessible &= np.linalg.norm(points - context_xyz[j], axis=1) > c_r[j]
            if not accessible.any():
                break
        areas[i] = 4.0 * np.pi * radius ** 2 * accessible.sum() / len(points)
    return areas


def burial_features(cx: Complex, pocket_cutoff: float = 10.0) -> Dict[str, float]:
    """Buried surface on both sides of the interface, split apolar/polar."""
    f: Dict[str, float] = {}

    lig_apolar = np.isin(cx.l_element, list(APOLAR))
    free = atom_sasa(cx.l_xyz, cx.l_element, cx.l_xyz, cx.l_element)

    mask = pocket_mask(cx, cutoff=pocket_cutoff)
    p_xyz, p_el = cx.p_xyz[mask], cx.p_element[mask]
    context_xyz = np.vstack([cx.l_xyz, p_xyz]) if len(p_xyz) else cx.l_xyz
    context_el = np.concatenate([cx.l_element, p_el]) if len(p_el) else cx.l_element
    bound = atom_sasa(cx.l_xyz, cx.l_element, context_xyz, context_el)

    buried = np.maximum(free - bound, 0.0)
    total_free = float(free.sum())
    f["ligand_sasa_free"] = total_free
    f["ligand_sasa_bound"] = float(bound.sum())
    f["ligand_sasa_buried"] = float(buried.sum())
    f["ligand_buried_fraction"] = float(buried.sum() / total_free) if total_free > 0 else 0.0
    f["ligand_buried_apolar"] = float(buried[lig_apolar].sum())
    f["ligand_buried_polar"] = float(buried[~lig_apolar].sum())
    f["ligand_buried_apolar_fraction"] = (
        float(buried[lig_apolar].sum() / buried.sum()) if buried.sum() > 0 else 0.0
    )
    # How completely the pocket wraps the ligand: distinguishes a deep occluded
    # site from a shallow surface groove at equal contact count.
    f["ligand_fully_buried_atoms"] = float((bound < 0.1).sum())
    f["ligand_exposed_atoms"] = float((bound > 5.0).sum())

    if len(p_xyz):
        p_free = atom_sasa(p_xyz, p_el, np.vstack([p_xyz, cx.p_xyz[~mask]]) if (~mask).any() else p_xyz,
                           np.concatenate([p_el, cx.p_element[~mask]]) if (~mask).any() else p_el)
        p_bound = atom_sasa(p_xyz, p_el, context_xyz, context_el)
        p_buried = np.maximum(p_free - p_bound, 0.0)
        prot_apolar = np.isin(p_el, list(APOLAR))
        f["protein_sasa_buried"] = float(p_buried.sum())
        f["protein_buried_apolar"] = float(p_buried[prot_apolar].sum())
        f["protein_buried_polar"] = float(p_buried[~prot_apolar].sum())
    else:
        f["protein_sasa_buried"] = 0.0
        f["protein_buried_apolar"] = 0.0
        f["protein_buried_polar"] = 0.0

    # The interface area both HADDOCK's `bsa` and PRODIGY report, kept under a
    # recognisable name so v1 and v2 numbers can be compared directly.
    f["bsa_total"] = f["ligand_sasa_buried"] + f["protein_sasa_buried"]
    return f

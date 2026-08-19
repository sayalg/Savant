"""Score one docked complex end to end.

Featurize, gate, regress, attach a conformal interval, and describe the
interface that produced the number. No HADDOCK, no GPU, no network - the whole
call is numpy, scipy, RDKit and a pickled model, and it returns in about a
second on a laptop core.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from savant2 import LIGAND_CHAIN, PROTEIN_CHAIN
from savant2.features import featurize, to_vector
from savant2.models import SavantBundle

# Interaction counts worth showing a user, in the order a chemist reads them.
PROFILE_FIELDS = [
    ("hydrophobic_contacts", "hydrophobic contacts"),
    ("hbond_contacts", "hydrogen bonds"),
    ("hbond_ligand_donor", "  ligand donates"),
    ("hbond_ligand_acceptor", "  ligand accepts"),
    ("ionic_ligand_anion", "salt bridges (ligand anion)"),
    ("ionic_ligand_cation", "salt bridges (ligand cation)"),
    ("pi_stack_parallel", "pi-stacking, face-to-face"),
    ("pi_stack_tilted", "pi-stacking, edge-to-face"),
    ("cation_pi_protein_cation", "cation-pi"),
    ("halogen_bonds", "halogen bonds"),
    ("metal_coordination", "metal coordination"),
]


def pk_to_kd_nm(pk: float) -> float:
    """pKd -> Kd in nM. pKd = 9 - log10(Kd[nM])."""
    return float(10.0 ** (9.0 - pk))


def score_complex(
    pdb_path: str | Path,
    smiles: str,
    model: SavantBundle,
    protein_chain: str = PROTEIN_CHAIN,
    ligand_chain: str = LIGAND_CHAIN,
) -> Dict:
    """Predict binding for one complex and explain the interface behind it."""
    features, meta = featurize(pdb_path, smiles, protein_chain, ligand_chain)
    X = to_vector(features, model.feature_names).reshape(1, -1)
    out = model.predict(X)

    pk = float(out["pk"][0])
    lo, hi = float(out["pk_lower"][0]), float(out["pk_upper"][0])
    is_binder = bool(out["is_binder"][0])

    profile = {
        label: float(features.get(key, 0.0))
        for key, label in PROFILE_FIELDS
        if features.get(key, 0.0)
    }
    contacts = {
        key.replace("contact_res_", ""): int(value)
        for key, value in features.items()
        if key.startswith("contact_res_") and value
    }

    result = {
        "classification": "binder" if is_binder else "screened_out",
        "p_binder": float(out["p_binder"][0]),
        "gate_threshold": model.gate_threshold,
        "pK": pk,
        "pK_lower": lo,
        "pK_upper": hi,
        "Kd_nM": pk_to_kd_nm(pk),
        # Note the inversion: a HIGHER pK is a LOWER Kd, so the upper pK bound
        # gives the tighter (smaller) Kd.
        "Kd_nM_lower": pk_to_kd_nm(hi),
        "Kd_nM_upper": pk_to_kd_nm(lo),
        "interval_width_log_units": round(hi - lo, 3),
        "confidence": _confidence_label(hi - lo),
        "interaction_profile": profile,
        "contacting_residue_types": dict(sorted(contacts.items(), key=lambda kv: -kv[1])),
        "n_interacting_residues": int(features.get("n_interacting_residues", 0)),
        "buried_surface_A2": round(float(features.get("bsa_total", 0.0)), 1),
        "ligand_buried_fraction": round(float(features.get("ligand_buried_fraction", 0.0)), 3),
        "meta": meta,
    }
    if not is_binder:
        # Retained from v1: only binders get a pK. Unlike v1 the rejected
        # prediction is still reported, so a screened-out compound can be
        # inspected rather than silently disappearing.
        result["note"] = (
            f"gate probability {result['p_binder']:.2f} is below the "
            f"{model.gate_threshold:.2f} threshold; pK shown for reference only"
        )
    return result


def _confidence_label(width: float) -> str:
    """Coarse label for the conformal interval width, in log units."""
    if width <= 1.5:
        return "high"
    if width <= 3.0:
        return "moderate"
    return "low"

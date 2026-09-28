"""Geometry supervision from crystal complexes (PLAN.md Section 6.1, Phase 2).

Produces the two targets that Stage 1 trains on and that keep Stage 2 honest:

  * **distogram targets** - the binned distance between every protein pseudo-site
    and every ligand site, using the same grid as `physics/terms.py`;
  * **pocket labels** - residues with any heavy atom within 8 A of the ligand.

Section 0.3.4 requires the distogram cross-entropy anchor to stay on through EVERY
later stage, including the affinity stage. That is not a nicety. Without a
geometric anchor, the affinity loss discovers that moving probability mass toward
each pair's energy minimum lowers the loss, and the model converges to an
over-attractive, geometrically impossible solution that scores well on pK and means
nothing. If no structural data is staged, Stage 1 cannot run and the physics claim
is not supported - `scripts/phase2_geometry.py` says so rather than proceeding.

A minimal PDB parser is used rather than a Biopython dependency: PDBbind ships
`.pdb`, the ATOM-record format is fixed-column, and one fewer import that can fail
on a compute node is worth more than generality. mmCIF goes through Biopython or
gemmi when either is installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..physics import terms as T
from ..physics.sites import AA_IDX, BB, RING, SC, TIP, UNK

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    ## Common modified residues, mapped to their parent. Dropping them instead
    ## would punch holes in the sequence and shift every later residue index.
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y", "CSO": "C", "HIE": "H",
    "HID": "H", "HIP": "H", "CYX": "C", "ASH": "D", "GLH": "E", "LYN": "K",
}

## Terminal functional atom per residue: the site that carries charge and H-bonding.
TIP_ATOM = {
    "R": "CZ", "N": "CG", "D": "CG", "C": "SG", "Q": "CD", "E": "CD", "H": "CE1",
    "I": "CD1", "L": "CG", "K": "NZ", "M": "SD", "F": "CZ", "P": "CG", "S": "OG",
    "T": "OG1", "W": "CZ2", "Y": "OH", "V": "CB",
}
RING_ATOMS = {
    "F": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "Y": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "W": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "H": ("CG", "ND1", "CD2", "CE1", "NE2"),
}
BACKBONE = {"N", "CA", "C", "O", "OXT"}
POCKET_CUTOFF = 8.0


def parse_pdb(path) -> list[dict]:
    """ATOM/HETATM records -> one dict per residue with its atom coordinates.

    Only the first model and the first altloc are kept; a second conformation would
    otherwise contribute a duplicate set of sites at slightly different distances.
    """
    residues: dict[tuple, dict] = {}
    for ln in Path(path).read_text(errors="replace").splitlines():
        if ln.startswith("ENDMDL"):
            break
        if not ln.startswith("ATOM"):
            continue
        alt = ln[16]
        if alt not in (" ", "A"):
            continue
        element = ln[76:78].strip() or ln[12:16].strip()[:1]
        if element == "H":
            continue
        resn = ln[17:20].strip()
        key = (ln[21], ln[22:27].strip())          # chain, resseq+icode
        r = residues.setdefault(key, {"resn": resn, "chain": ln[21], "atoms": {}})
        try:
            r["atoms"][ln[12:16].strip()] = (float(ln[30:38]), float(ln[38:46]),
                                             float(ln[46:54]))
        except ValueError:
            continue
    return [dict(key=k, **v) for k, v in residues.items()]


def parse_ligand(path) -> np.ndarray | None:
    """Heavy-atom coordinates from an SDF or MOL2 file."""
    p = Path(path)
    if not p.exists():
        return None
    from ..preflight import require_rdkit
    require_rdkit("ligand structure parsing")
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    if p.suffix == ".sdf":
        m = next(iter(Chem.SDMolSupplier(str(p), removeHs=True, sanitize=False)), None)
    else:
        m = Chem.MolFromMol2File(str(p), removeHs=True, sanitize=False)
    if m is None or m.GetNumConformers() == 0:
        return None
    pos = np.asarray(m.GetConformer().GetPositions())
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    return pos[heavy]


def residue_sites(residues: list[dict]):
    """-> (sequence, site_xyz (L,4,3), site_valid (L,4), residue_heavy list)."""
    seq, xyz, valid, heavy = [], [], [], []
    for r in residues:
        one = THREE_TO_ONE.get(r["resn"])
        if one is None:
            continue
        at = r["atoms"]
        pos = np.full((4, 3), np.nan, np.float32)
        ok = np.zeros(4, bool)
        if "CA" in at:
            pos[BB], ok[BB] = at["CA"], True
        side = [v for k, v in at.items() if k not in BACKBONE]
        if side:
            pos[SC], ok[SC] = np.mean(side, 0), True
        tip = TIP_ATOM.get(one)
        if tip and tip in at:
            pos[TIP], ok[TIP] = at[tip], True
        elif side:
            ## Fall back to the atom furthest from CA: a disordered side chain still
            ## has a terminal group, and dropping the site would delete its H-bonding.
            ca = at.get("CA")
            if ca is not None:
                far = max(side, key=lambda v: np.linalg.norm(np.subtract(v, ca)))
                pos[TIP], ok[TIP] = far, True
        if one in RING_ATOMS:
            ra = [at[a] for a in RING_ATOMS[one] if a in at]
            if len(ra) >= 4:
                pos[RING], ok[RING] = np.mean(ra, 0), True
        seq.append(one)
        xyz.append(pos)
        valid.append(ok)
        heavy.append(np.asarray(list(at.values()), np.float32))
    return ("".join(seq), np.stack(xyz) if xyz else np.zeros((0, 4, 3), np.float32),
            np.stack(valid) if valid else np.zeros((0, 4), bool), heavy)


def ligand_sites(lig_xyz: np.ndarray, ring_groups: list | None = None) -> np.ndarray:
    """Heavy atoms plus ring centroids, matching `embed_ligand.build_graph` order."""
    if ring_groups:
        cents = [lig_xyz[list(g)].mean(0) for g in ring_groups if len(g)]
        if cents:
            return np.concatenate([lig_xyz, np.stack(cents)], 0)
    return lig_xyz


def labels_for_complex(protein_pdb, ligand_file, ring_groups=None) -> dict | None:
    """Everything Stage 1 needs for one crystal complex."""
    residues = parse_pdb(protein_pdb)
    if not residues:
        return None
    lig = parse_ligand(ligand_file)
    if lig is None or len(lig) == 0:
        return None
    seq, sxyz, svalid, heavy = residue_sites(residues)
    if not seq:
        return None
    lsites = ligand_sites(lig, ring_groups)

    ## pocket: any heavy atom of the residue within POCKET_CUTOFF of any ligand atom
    pocket = np.zeros(len(seq), np.float32)
    for i, hv in enumerate(heavy):
        if len(hv) and np.linalg.norm(hv[:, None, :] - lig[None, :, :],
                                      axis=-1).min() <= POCKET_CUTOFF:
            pocket[i] = 1.0

    flat_xyz = sxyz.reshape(-1, 3)
    flat_valid = svalid.reshape(-1)
    d = np.full((len(flat_xyz), len(lsites)), np.inf, np.float32)
    if flat_valid.any():
        d[flat_valid] = np.linalg.norm(
            flat_xyz[flat_valid][:, None, :] - lsites[None, :, :], axis=-1)
    bins = np.full(d.shape, T.FAR_BIN, np.int16)
    real = np.isfinite(d) & (d < T.D_MAX)
    bins[real] = np.clip(((d[real] - T.D_MIN) / T.D_STEP).astype(int),
                         0, T.N_DIST_BINS - 1)
    return {
        "sequence": seq,
        "res_idx": np.array([AA_IDX.get(c, UNK) for c in seq], np.int16),
        "site_xyz": sxyz, "site_valid": svalid,
        "lig_xyz": lsites,
        "pocket": pocket,
        "dist": d, "dist_bins": bins,
        "dist_valid": np.repeat(flat_valid[:, None], len(lsites), 1),
        "n_pocket": int(pocket.sum()), "n_res": len(seq), "n_lig_sites": len(lsites),
    }


def contact_statistics(labels: list[dict]) -> dict:
    """Crystal contact statistics - the target for `contact_prior_loss`.

    Measured from the data rather than assumed, because the "expected contacts per
    ligand atom" number is otherwise a magic constant sitting in a loss function.
    """
    per_atom = []
    for L in labels:
        c = (L["dist"] < 4.5).sum(0)
        per_atom.append(c[c > 0])
    if not per_atom:
        return {}
    allc = np.concatenate(per_atom)
    return {"mean_contacts_per_buried_atom": float(allc.mean()),
            "median": float(np.median(allc)), "p90": float(np.percentile(allc, 90)),
            "n_complexes": len(labels), "n_atoms": int(len(allc))}

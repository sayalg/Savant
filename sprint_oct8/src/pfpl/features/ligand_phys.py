"""Ligand-side physics features, no pose required (PLAN.md Section 6.3).

Three tiers, run as an ablation factor (A vs A+B vs A+B+C) in Phase 9:

  Tier A  RDKit: Crippen logP/MR, TPSA, polar/apolar SASA, rotor count, formal
          charge, Gasteiger charges, ring/aromatic flags, an ETKDG conformer
          ensemble with MMFF energies, and the ensemble's internal-distance
          statistics. Always computed; cheap.
  Tier B  GFN2-xTB or AIMNet2 partial charges, dipole, polarizability, and the
          conformer energy spread. GPU/CPU node work, optional.
  Tier C  xTB ALPB/GBSA solvation free energy, which is what the desolvation term
          in `physics/energy.py` consumes.

The conformer ensemble earns its cost twice: the scalar descriptors below, and the
per-SITE distance matrix (`site_distances`) that is the `lig_dist` argument to the
triangle penalty (Section 2.1) - the constraint that stops the independent per-pair
distributions from describing an impossible geometry. Without it, Variant A has no
geometric self-consistency signal at all.

(Until 2026-09-25 that matrix was computed and thrown away: only four scalars were
persisted and training filled `lig_dist` with a constant, so the penalty never
engaged. `tier_a` now returns it as `_site_dist` and Phase 3 stores it.)

Cost: the ensemble is the expensive part. Embedding and MMFF are multi-threaded
through RDKit's `numThreads`; single-threaded, a 4-CPU job used one core (measured
1.16 s -> 0.36 s per ligand at the same 16 conformers / 400 MMFF iterations).
"""
from __future__ import annotations

import numpy as np

from ..preflight import require_rdkit

TIER_A = [
    "mw", "logp", "mr", "tpsa", "hba", "hbd", "n_rot", "n_rings", "n_arom_rings",
    "frac_csp3", "heavy_atoms", "formal_charge", "n_hetero", "n_halogen",
    "labute_asa", "polar_asa", "apolar_asa", "bertz", "qed",
    "conf_n", "conf_energy_spread", "conf_rmsd_mean", "radius_gyration",
]
TIER_B = ["xtb_dipole", "xtb_polarizability", "xtb_q_min", "xtb_q_max",
          "xtb_homo_lumo_gap", "xtb_energy_spread"]
TIER_C = ["dG_solv"]


def tier_names(tier: str) -> list[str]:
    return {"A": TIER_A, "AB": TIER_A + TIER_B,
            "ABC": TIER_A + TIER_B + TIER_C}[tier.upper().replace("+", "")]


def _asa(mol):
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import Crippen
    contribs = rdMolDescriptors._CalcLabuteASAContribs(mol)[0]
    logps = [c for c, _ in Crippen._GetAtomContribs(mol)]
    total = float(sum(contribs))
    ## Split the Labute surface by each atom's Crippen logP contribution: negative
    ## contribution = polar. Crude, but it needs no pose and it is the quantity the
    ## desolvation term actually wants.
    polar = float(sum(c for c, lp in zip(contribs, logps) if lp < 0))
    return total, polar, total - polar


def n_threads() -> int:
    """CPUs this process may use: the Slurm allocation, not the whole node."""
    import os
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except AttributeError:                       # Windows / macOS
        return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


## Per-conformer embedding time limit (s). Guards against the rare macrocycle or
## peptide on which ETKDG grinds for minutes; a ligand that hits it falls back to
## random coordinates rather than stalling the shard. Needs RDKit >= 2024.03.
EMBED_TIMEOUT_S = 30


def conformers(mol, n: int = 16, seed: int = 0xF00D, prune: float = 0.5,
               threads: int | None = None):
    """ETKDG ensemble + MMFF optimisation. Returns (mol_with_confs, energies)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    threads = threads or n_threads()
    m = Chem.AddHs(mol)                      # appends H: heavy indices are unchanged
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    p.pruneRmsThresh = prune
    p.useSmallRingTorsions = True
    p.numThreads = threads
    if hasattr(p, "timeout"):
        p.timeout = EMBED_TIMEOUT_S
    cids = AllChem.EmbedMultipleConfs(m, numConfs=n, params=p)
    if not len(cids):
        ## Embedding fails on strained or unusual molecules. Random coordinates give
        ## a usable (if poor) geometry rather than dropping the ligand entirely.
        p.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(m, numConfs=max(n // 4, 1), params=p)
    energies = []
    if len(cids):
        try:
            res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=400, numThreads=threads)
            energies = [e for conv, e in res]
        except Exception:
            energies = [float("nan")] * len(cids)
    return m, np.asarray(energies, float)


def site_distances(mol_with_confs, heavy_mol) -> np.ndarray | None:
    """(S, S) ensemble-MAX distance between the ligand graph's sites, or None.

    Sites are in `embed_ligand.build_graph` order - heavy atoms 0..n_heavy-1 of the
    canonical molecule, then one centroid per SSSR ring - and are NOT truncated at
    max_atoms: the graph truncates by dropping trailing sites, so the dataset slices
    [:n, :n] and the two still line up.

    MAX over conformers, not mean: the penalty enforces |d_ij - d_ik| <= d_jk, which
    must hold in the bound pose, and the bound pose's d_jk can be anything the
    ligand can reach. A mean would penalise flexible ligands for conformations they
    can actually adopt.
    """
    confs = list(mol_with_confs.GetConformers())
    if not confs:
        return None
    ## Rings from the SAME heavy-atom molecule build_graph reads, in the same order.
    n_heavy = heavy_mol.GetNumAtoms()
    rings = [list(r) for r in heavy_mol.GetRingInfo().AtomRings()]
    xs = []
    for c in confs:
        pos = np.asarray(c.GetPositions())
        heavy = pos[:n_heavy]
        cents = (np.stack([pos[r].mean(0) for r in rings]) if rings
                 else np.zeros((0, 3)))
        xs.append(np.concatenate([heavy, cents]))
    xs = np.stack(xs)                                              # (C, S, 3)
    d = np.linalg.norm(xs[:, :, None, :] - xs[:, None, :, :], axis=-1)
    return d.max(0).astype(np.float16)


def internal_distances(mol_with_confs, heavy_idx) -> tuple[np.ndarray, np.ndarray]:
    """(mean, std) heavy-atom distance matrix over the conformer ensemble.

    The mean is the `lig_dist` bound in `physics/energy.triangle_penalty`; the std
    says how rigid that bound is, which the geometric prior in the pair module uses.
    """
    from rdkit.Chem import rdMolTransforms  # noqa: F401  (import validates the build)
    confs = list(mol_with_confs.GetConformers())
    if not confs:
        n = len(heavy_idx)
        return np.zeros((n, n)), np.zeros((n, n))
    xs = np.stack([np.asarray(c.GetPositions())[heavy_idx] for c in confs])  # (C,A,3)
    d = np.linalg.norm(xs[:, :, None, :] - xs[:, None, :, :], axis=-1)
    return d.mean(0), d.std(0)


def tier_a(smiles: str, n_conf: int = 16, seed: int = 0xF00D,
           threads: int | None = None) -> dict:
    """Tier-A scalars, plus `_site_dist` (flattened fp16 (S, S)) and `_n_sites`."""
    require_rdkit("Tier-A ligand physics")
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
    from .embed_ligand import canonical_mol
    RDLogger.DisableLog("rdApp.*")

    ## Canonical order so the site-distance matrix indexes like the ligand graph.
    ## Atom order does not change any descriptor below.
    mol = canonical_mol(smiles)
    if mol is None:
        return {k: np.nan for k in TIER_A}
    total_asa, polar, apolar = _asa(mol)
    mh, energies = conformers(mol, n_conf, seed, threads=threads)
    sd = site_distances(mh, mol)
    heavy = [a.GetIdx() for a in mh.GetAtoms() if a.GetAtomicNum() > 1]
    dmean, _ = internal_distances(mh, heavy)
    e = energies[np.isfinite(energies)]

    rg = np.nan
    if mh.GetNumConformers():
        pos = np.asarray(mh.GetConformer(0).GetPositions())[heavy]
        rg = float(np.sqrt(((pos - pos.mean(0)) ** 2).sum(1).mean()))

    return {
        "mw": Descriptors.MolWt(mol), "logp": Crippen.MolLogP(mol),
        "mr": Crippen.MolMR(mol), "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol), "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "n_rot": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "n_rings": rdMolDescriptors.CalcNumRings(mol),
        "n_arom_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "frac_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(mol),
        "n_hetero": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "n_halogen": sum(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in mol.GetAtoms()),
        "labute_asa": total_asa, "polar_asa": polar, "apolar_asa": apolar,
        "bertz": Descriptors.BertzCT(mol), "qed": QED.qed(mol),
        "conf_n": int(mh.GetNumConformers()),
        "conf_energy_spread": float(np.ptp(e)) if len(e) > 1 else 0.0,
        "conf_rmsd_mean": float(dmean.mean()) if dmean.size else np.nan,
        "radius_gyration": rg,
        "_n_sites": int(sd.shape[0]) if sd is not None else 0,
        "_site_dist": sd.ravel().tobytes() if sd is not None else b"",
    }


def tier_b(smiles: str, xtb_bin: str = "xtb", timeout: int = 120) -> dict:
    """GFN2-xTB single point. Returns NaN for everything if xtb is unavailable.

    NaN is a first-class value here: `physics/energy.py` treats a NaN solvation term
    as a zero contribution, so a ligand xtb could not handle degrades one feature
    instead of dropping the row.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    from ..paths import TMP
    out = {k: np.nan for k in TIER_B}
    if shutil.which(xtb_bin) is None:
        return out
    require_rdkit("Tier-B ligand physics")
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out
    mh, energies = conformers(mol, 4)
    if not mh.GetNumConformers():
        return out
    TMP.mkdir(parents=True, exist_ok=True)
    ## dir=TMP, not the system default: on a compute node /tmp is
    ## node-local and reclaimed when the job ends, and $HOME is quota'd.
    with tempfile.TemporaryDirectory(dir=TMP) as td:
        xyz = Path(td) / "m.xyz"
        Chem.MolToXYZFile(mh, str(xyz), confId=0)
        try:
            r = subprocess.run([xtb_bin, str(xyz), "--gfn", "2", "--sp", "--chrg",
                                str(Chem.GetFormalCharge(mol))],
                               capture_output=True, text=True, timeout=timeout, cwd=td)
        except Exception:
            return out
        for ln in r.stdout.splitlines():
            if "molecular dipole" in ln.lower():
                out["xtb_dipole"] = _num_after(r.stdout, ln)
            if "HOMO-LUMO GAP" in ln:
                out["xtb_homo_lumo_gap"] = _first_float(ln)
    e = energies[np.isfinite(energies)]
    out["xtb_energy_spread"] = float(np.ptp(e)) if len(e) > 1 else 0.0
    return out


def tier_c(smiles: str, xtb_bin: str = "xtb", solvent: str = "water",
           timeout: int = 180) -> dict:
    """ALPB solvation free energy in kcal/mol. The desolvation term's input."""
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    from ..paths import TMP
    if shutil.which(xtb_bin) is None:
        return {"dG_solv": np.nan}
    require_rdkit("Tier-C ligand physics")
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"dG_solv": np.nan}
    mh, _ = conformers(mol, 2)
    if not mh.GetNumConformers():
        return {"dG_solv": np.nan}
    TMP.mkdir(parents=True, exist_ok=True)
    ## dir=TMP, not the system default: on a compute node /tmp is
    ## node-local and reclaimed when the job ends, and $HOME is quota'd.
    with tempfile.TemporaryDirectory(dir=TMP) as td:
        xyz = Path(td) / "m.xyz"
        Chem.MolToXYZFile(mh, str(xyz), confId=0)
        cmd = [xtb_bin, str(xyz), "--gfn", "2", "--sp", "--alpb", solvent,
               "--chrg", str(Chem.GetFormalCharge(mol))]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=td)
        except Exception:
            return {"dG_solv": np.nan}
        for ln in r.stdout.splitlines():
            if "-> Gsolv" in ln or "total solvation" in ln.lower():
                v = _first_float(ln)
                ## xtb reports Eh; 1 Hartree = 627.5095 kcal/mol.
                return {"dG_solv": v * 627.5095 if np.isfinite(v) and abs(v) < 1 else v}
    return {"dG_solv": np.nan}


def _first_float(line: str) -> float:
    import re
    m = re.search(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", line)
    return float(m.group()) if m else np.nan


def _num_after(text: str, line: str) -> float:
    lines = text.splitlines()
    i = lines.index(line)
    for ln in lines[i:i + 4]:
        if "full:" in ln:
            return _first_float(ln.split("full:")[-1])
    return np.nan


def compute(smiles: str, tier: str = "A", **kw) -> dict:
    d = tier_a(smiles, **{k: v for k, v in kw.items()
                          if k in ("n_conf", "seed", "threads")})
    t = tier.upper().replace("+", "")
    if "B" in t:
        d |= tier_b(smiles, kw.get("xtb_bin", "xtb"))
    if "C" in t:
        d |= tier_c(smiles, kw.get("xtb_bin", "xtb"), kw.get("solvent", "water"))
    return d

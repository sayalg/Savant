"""Ligand standardisation and sequence handling (PLAN.md Section 5.2).

One fixed policy, applied identically to every source, because the whole point of
harmonisation is that ChEMBL's and BindingDB's spellings of the same molecule
collapse to the same key. The previous sprint deduped on the raw SMILES string for
one run and silently kept both copies of thousands of molecules.

Ligands: largest organic fragment, salts and solvents removed, stereochemistry
KEPT (PLAN.md Section 9 asks for a chirality-inversion sanity check, which needs
stereo to survive ingestion), one protonation policy at pH 7.4.

Two InChIKeys are stored. `inchikey_full` is the exact molecule; `inchikey_flat` is
the first block only - the connectivity layer - which is what merges tautomers,
salts and stereoisomers. Dedupe uses the full key and *leakage checks* use the flat
one, because a stereoisomer of a training ligand in the test set is leakage even
though it is a different molecule.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from ..preflight import require_rdkit

MIN_HEAVY, MAX_HEAVY = 5, 80
## Section 5.2: drop organometallics. These are the metals that appear as covalently
## bound centres rather than as counter-ions (counter-ions go with the salt strip).
_METALS = {"Fe", "Zn", "Cu", "Mn", "Co", "Ni", "Mo", "W", "V", "Cr", "Ru", "Rh",
           "Pd", "Pt", "Ir", "Os", "Au", "Ag", "Hg", "Cd", "Pb", "Sn", "Ti", "Gd"}
_AA1 = set("ACDEFGHIKLMNPQRSTVWY")
_NONSTD = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


def sequence_hash(seq: str) -> str:
    return hashlib.sha256(seq.strip().upper().encode()).hexdigest()[:16]


def clean_sequence(seq: str) -> str | None:
    """Uppercase, strip whitespace, map the ambiguity codes, reject the rest.

    B/Z/J/X/U/O appear in UniProt. Mapping them to their common residue is better
    than dropping the protein, but `*` (stop) or digits mean the record is not a
    sequence and it is dropped.
    """
    if not isinstance(seq, str):
        return None
    s = re.sub(r"\s+", "", seq).upper()
    if not s or len(s) < 20:
        return None
    s = (s.replace("B", "N").replace("Z", "Q").replace("J", "L")
          .replace("U", "C").replace("O", "K"))
    s = _NONSTD.sub("X", s)
    if (np.frombuffer(s.encode(), dtype="S1") == b"X").mean() > 0.10:
        return None      # >10% unknown means this is not a usable protein sequence
    return s


def standardize_smiles(smiles, protonate: bool = True, n_jobs_note: bool = True):
    """Vectorised over a list. Returns (std_smiles, inchikey_full, inchikey_flat, ok)."""
    require_rdkit("ligand standardisation")
    from rdkit import Chem, RDLogger
    from rdkit.Chem import inchi
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")

    lfc = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()
    normalizer = rdMolStandardize.Normalizer()

    std, kf, kflat, ok = [], [], [], []
    for s in smiles:
        std.append(None); kf.append(None); kflat.append(None); ok.append("bad_smiles")
        if not isinstance(s, str) or not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            m = normalizer.normalize(m)
            m = lfc.choose(m)
            if protonate:
                ## Fixed policy, not per-molecule tuning: neutralise, then let the
                ## physics heads learn the effective group charge. Dimorphite-DL
                ## (Section 3) is the upgrade path if the charge channel turns out to
                ## matter in the Wave-3 Coulomb ablation.
                m = uncharger.uncharge(m)
            m = rdMolStandardize.Cleanup(m)
        except Exception:
            ok[-1] = "standardize_failed"
            continue
        syms = {a.GetSymbol() for a in m.GetAtoms()}
        n_heavy = m.GetNumHeavyAtoms()
        if syms & _METALS:
            ok[-1] = "organometallic"
            continue
        if n_heavy < MIN_HEAVY:
            ok[-1] = "too_small"
            continue
        if n_heavy > MAX_HEAVY:
            ok[-1] = "too_large"
            continue
        try:
            k = inchi.MolToInchiKey(m)
        except Exception:
            k = None
        std[-1] = Chem.MolToSmiles(m)
        kf[-1] = k
        kflat[-1] = k.split("-")[0] if k else None
        ok[-1] = "ok"
    return std, kf, kflat, ok


def is_peptide_like(smiles: str, min_amides: int = 4) -> bool:
    """Section 5.2 drops long peptides; they are a different modelling problem."""
    require_rdkit("peptide filtering")
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return False
    patt = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
    return len(m.GetSubstructMatches(patt)) >= min_amides


def apply_mutations(seq: str, mutation_string: str) -> tuple[str | None, bool]:
    """Apply 'A123V' / 'A123V,L45F' style mutations. Returns (sequence, parsed_ok).

    Section 5.2: if it does not parse, flag `is_mutant` and exclude from the main
    training set rather than training on the wild-type sequence under a mutant's
    label - that teaches the model that two different affinities share one input.
    """
    if not isinstance(mutation_string, str) or not mutation_string.strip():
        return seq, True
    out = list(seq)
    for tok in re.split(r"[,;/\s]+", mutation_string.strip()):
        m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", tok.upper())
        if not m:
            return None, False
        wt, pos, mut = m.group(1), int(m.group(2)) - 1, m.group(3)
        if pos < 0 or pos >= len(out) or out[pos] != wt or mut not in _AA1:
            return None, False
        out[pos] = mut
    return "".join(out), True


def standardize_frame(df: pd.DataFrame, smiles_col: str = "ligand_smiles_raw",
                      seq_col: str = "sequence", verbose: bool = True) -> pd.DataFrame:
    """Standardise in place and return the frame with the drop reasons attached."""
    std, kf, kflat, ok = standardize_smiles(df[smiles_col].tolist())
    d = df.copy()
    d["ligand_smiles_std"] = std
    d["inchikey_full"] = kf
    d["inchikey_flat"] = kflat
    d["_ligand_status"] = ok
    d["sequence"] = [clean_sequence(s) for s in d[seq_col]]
    d["sequence_hash"] = [sequence_hash(s) if s else None for s in d["sequence"]]
    d["_seq_status"] = np.where(d["sequence"].isna(), "bad_sequence", "ok")
    if verbose:
        print("  ligand standardisation:",
              pd.Series(ok).value_counts().to_dict())
        print("  sequence cleaning:", d["_seq_status"].value_counts().to_dict())
    return d

#!/usr/bin/env python
"""
Step 04 - Select the ~1000-pair benchmark.

Selection is a round-robin over three nested strata so that no single axis is
allowed to dominate:

  1. NOVELTY TIER  - how far the target sits from the training proteins:
        same_protein  the accession (or a >=90% 8-mer-containment match) is in
                      training; the ligand is still novel, so this measures
                      "new chemistry on a known target"
        same_family   a new protein whose UniProt family the model has seen
        novel_family  a protein family absent from training entirely
     Every pair's ligand is already guaranteed dissimilar from training by the
     step-01 gate, so all three tiers are legitimate held-out tests; the tier
     says *which kind* of generalisation is being probed.

  2. PROTEIN CLASS - ChEMBL level-2 class (Kinase, Family A GPCR, Protease,
     Nuclear receptor, ion channels, ...), falling back to level 1.

  3. pKd BIN + LIGAND SCAFFOLD - within a class, pairs are drawn to spread the
     affinity range and to avoid repeating Bemis-Murcko scaffolds.

Selection depends only on work/candidates.csv, work/protein_annotation.csv and
reference/train_protein_families.csv, so it is reproducible from this directory
alone under the fixed RANDOM_SEED.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from common import (CANDIDATES, N_PAIRS, PROT_ANNOT, RANDOM_SEED, TRAIN_PROT_TAX, WORK,
                    ensure_dirs, log)

RDLogger.DisableLog("rdApp.*")

# pairs per novelty tier
TIER_QUOTA = {"same_protein": 400, "same_family": 400, "novel_family": 200}
PKD_EDGES = [0, 5, 6, 7, 8, 9, 20]
MAX_PER_TARGET = 20
MAX_PER_SCAFFOLD = 12


def murcko(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, {}
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        s = None
    return s, {
        "molecular_weight": round(Descriptors.MolWt(m), 3),
        "logP": round(Descriptors.MolLogP(m), 4),
        "num_h_donors": Descriptors.NumHDonors(m),
        "num_h_acceptors": Descriptors.NumHAcceptors(m),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(m),
        "tpsa": round(Descriptors.TPSA(m), 2),
        "num_rings": Descriptors.RingCount(m),
        "num_aromatic_rings": Descriptors.NumAromaticRings(m),
        "num_heavy_atoms": m.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(m),
    }


def build_frame() -> pd.DataFrame:
    cand = pd.read_csv(CANDIDATES)
    prot = pd.read_csv(PROT_ANNOT)
    cand["acc"] = cand["uniprot_id"].astype(str).str.split("-").str[0].str.upper()
    df = cand.merge(prot.rename(columns={"uniprot_id": "acc"}), on="acc", how="left")
    df = df[df["acc"].notna() & df["acc"].ne("NAN")]

    # --- novelty tier ---
    tax = pd.read_csv(TRAIN_PROT_TAX)
    train_fams = set(tax["family_l1"].dropna())

    def tier(r):
        if r["is_same_protein"]:
            return "same_protein"
        if isinstance(r["family_l1"], str) and r["family_l1"] in train_fams:
            return "same_family"
        return "novel_family"

    df["novelty_tier"] = df.apply(tier, axis=1)

    # --- protein class label used for stratification ---
    df["protein_class"] = df["chembl_l2"].fillna(df["chembl_l1"]).fillna("Unclassified")

    # --- ligand scaffold + descriptors ---
    uniq = df["canonical_smiles"].dropna().unique()
    sc, desc = {}, {}
    for s in uniq:
        sc[s], desc[s] = murcko(s)
    df["murcko_scaffold"] = df["canonical_smiles"].map(sc)
    for k in ("molecular_weight", "logP", "num_h_donors", "num_h_acceptors",
              "num_rotatable_bonds", "tpsa", "num_rings", "num_aromatic_rings",
              "num_heavy_atoms", "formal_charge"):
        df[k] = df["canonical_smiles"].map(lambda s, k=k: desc.get(s, {}).get(k))

    df["pkd_bin"] = pd.cut(df["neg_log10_affinity_M"], bins=PKD_EDGES,
                           labels=[f"b{i}" for i in range(len(PKD_EDGES) - 1)])

    log(f"candidates: {len(df)} pairs, {df.acc.nunique()} targets")
    return df


def select(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    chosen: list[int] = []
    per_target: dict = {}
    per_scaffold: dict = {}

    for tier, quota in TIER_QUOTA.items():
        pool = df[df["novelty_tier"] == tier]
        if pool.empty:
            continue
        # order within each (class, pkd_bin) cell is random under the fixed seed
        pool = pool.sample(frac=1.0, random_state=int(rng.integers(1e9)))

        cells: dict = {}
        for cls, sub in pool.groupby("protein_class", sort=False):
            for b, ss in sub.groupby("pkd_bin", sort=False, observed=True):
                if len(ss):
                    cells[(cls, b)] = list(ss.index)

        picked, exhausted = 0, False
        while picked < quota and not exhausted:
            exhausted = True
            # round-robin: one pair from each (class, pKd bin) cell per sweep,
            # visiting sparser classes first so they are not crowded out
            order = sorted(cells, key=lambda k: (len(cells[k]), str(k)))
            for key in order:
                if picked >= quota:
                    break
                bucket = cells.get(key) or []
                while bucket:
                    idx = bucket.pop(0)
                    r = df.loc[idx]
                    if per_target.get(r["acc"], 0) >= MAX_PER_TARGET:
                        continue
                    sk = r["murcko_scaffold"]
                    if sk and per_scaffold.get(sk, 0) >= MAX_PER_SCAFFOLD:
                        continue
                    chosen.append(idx)
                    per_target[r["acc"]] = per_target.get(r["acc"], 0) + 1
                    if sk:
                        per_scaffold[sk] = per_scaffold.get(sk, 0) + 1
                    picked += 1
                    exhausted = False
                    break
        log(f"  {tier}: selected {picked}/{quota} "
            f"({df.loc[chosen[-picked:] if picked else []].acc.nunique()} targets)")

    sel = df.loc[chosen].copy()

    # top up to N_PAIRS if a tier ran dry, respecting the per-target cap
    if len(sel) < N_PAIRS:
        rest = df.drop(index=sel.index).sample(frac=1.0, random_state=RANDOM_SEED)
        add = []
        for idx, r in rest.iterrows():
            if len(sel) + len(add) >= N_PAIRS:
                break
            if per_target.get(r["acc"], 0) >= MAX_PER_TARGET:
                continue
            add.append(idx)
            per_target[r["acc"]] = per_target.get(r["acc"], 0) + 1
        if add:
            log(f"  topped up with {len(add)} pairs to reach {N_PAIRS}")
            sel = pd.concat([sel, df.loc[add]])
    return sel


def main() -> None:
    ensure_dirs()
    df = build_frame()
    sel = select(df).reset_index(drop=True)
    sel = sel.sort_values(["novelty_tier", "protein_class", "neg_log10_affinity_M"])
    sel.insert(0, "benchmark_id", [f"bm_{i+1:04d}" for i in range(len(sel))])
    dest = f"{WORK}/selected.csv"
    sel.to_csv(dest, index=False)
    log(f"wrote {dest}: {len(sel)} pairs, {sel.acc.nunique()} targets, "
        f"{sel.canonical_smiles.nunique()} ligands")
    log(f"tiers: {sel.novelty_tier.value_counts().to_dict()}")


if __name__ == "__main__":
    main()

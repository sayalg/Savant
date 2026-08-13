#!/usr/bin/env python
"""
Step 06 - Independent verification of the shipped benchmark.

Deliberately re-derives every disjointness claim from the raw training files
rather than trusting work/candidates.csv, so a bug anywhere in steps 01-05
cannot launder a bad pair into the deliverable. Exits non-zero on any failure.

Checks:
  1. pKd == 9 - log10(Kd_nM), and Kd values are not censor caps
  2. no benchmark ligand is an exact / skeleton match to a training ligand
  3. no benchmark ligand is ECFP4 Tanimoto >= 0.50 to any training ligand
  4. no (target, ligand) pair is duplicated
  5. every `same_protein` tier row really does match a training protein, and no
     `novel_family` row does
  6. sequences are clean amino-acid strings and SMILES parse
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from common import (CENSOR_CAPS_NM, MAX_LIGAND_TANIMOTO, MORGAN_BITS, MORGAN_RADIUS, OUT,
                    TRAIN_CSV, TRAIN_PROT_TAX, TRAIN_PROTEINS, log, pkd_from_kd_nm,
                    sanitize_seq)

RDLogger.DisableLog("rdApp.*")
AA = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
K = 8


def main() -> None:
    path = os.path.join(OUT, "savant_benchmark_1000.csv")
    b = pd.read_csv(path)
    log(f"verifying {path}: {len(b)} rows")
    fails, warns = [], []

    # 1. label convention -----------------------------------------------------
    bad = ~np.isclose(b["neg_log10_affinity_M"], pkd_from_kd_nm(b["kd_nM"]).round(3), atol=0.011)
    if bad.any():
        fails.append(f"{bad.sum()} rows: neg_log10_affinity_M != 9 - log10(kd_nM)")
    else:
        log("  [ok] pKd == 9 - log10(Kd_nM) on all rows")
    caps = b["kd_nM"].isin(set(CENSOR_CAPS_NM)).sum()
    if caps:
        fails.append(f"{caps} rows carry a censored (>X) Kd cap value")
    else:
        log("  [ok] no censored Kd cap values")

    # 2/3. ligand disjointness, recomputed from the training CSV ---------------
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    tsm = pd.read_csv(TRAIN_CSV, usecols=["smiles"])["smiles"].dropna().unique()
    can_keys, skel_keys, fps = set(), set(), []
    for s in tsm:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        can_keys.add(Chem.MolToSmiles(m))
        try:
            ik = Chem.MolToInchiKey(m)
            if ik:
                skel_keys.add(ik.split("-")[0])
        except Exception:
            pass
        fps.append(gen.GetFingerprint(m))
    log(f"  training ligands re-indexed: {len(fps)}")

    exact, worst, unparsable = 0, 0.0, 0
    for s in b["ligand_smiles"].dropna().unique():
        m = Chem.MolFromSmiles(s)
        if m is None:
            unparsable += 1
            continue
        can = Chem.MolToSmiles(m)
        try:
            skel = (Chem.MolToInchiKey(m) or "-").split("-")[0]
        except Exception:
            skel = None
        if can in can_keys or (skel and skel in skel_keys):
            exact += 1
        worst = max(worst, max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), fps)))
    if unparsable:
        fails.append(f"{unparsable} ligands do not parse")
    if exact:
        fails.append(f"{exact} ligands exactly match a training ligand")
    else:
        log("  [ok] no exact / InChIKey-skeleton ligand matches")
    if worst >= MAX_LIGAND_TANIMOTO:
        fails.append(f"max ligand Tanimoto to training = {worst:.3f} (limit {MAX_LIGAND_TANIMOTO})")
    else:
        log(f"  [ok] max ligand Tanimoto to training = {worst:.3f} < {MAX_LIGAND_TANIMOTO}")

    # 4. duplicates -----------------------------------------------------------
    d = b.duplicated(subset=["uniprot_id", "ligand_smiles"]).sum()
    if d:
        fails.append(f"{d} duplicate (target, ligand) pairs")
    else:
        log("  [ok] no duplicate (target, ligand) pairs")

    # 5. tier consistency, recomputed -----------------------------------------
    tp = pd.read_csv(TRAIN_PROTEINS)
    tax = pd.read_csv(TRAIN_PROT_TAX, usecols=["uniprot_id", "family_l1"])
    train_acc = {str(x).strip().upper() for x in tax["uniprot_id"].dropna()}
    train_fam = set(tax["family_l1"].dropna())

    def kmers(s):
        s = sanitize_seq(s)
        return {s[i:i + K] for i in range(len(s) - K + 1)}

    index, tsets = defaultdict(set), []
    for i, s in enumerate(tp["sequence"]):
        ks = kmers(s)
        tsets.append(ks)
        for k in ks:
            index[k].add(i)

    cont = {}
    for seq in b["target_sequence"].dropna().unique():
        qs = kmers(seq)
        hits = defaultdict(int)
        for k in qs:
            for j in index.get(k, ()):
                hits[j] += 1
        cont[seq] = max((c / max(1, min(len(qs), len(tsets[j]))) for j, c in hits.items()),
                        default=0.0)

    b["_cont"] = b["target_sequence"].map(cont)
    b["_in_train"] = b["uniprot_id"].astype(str).str.upper().isin(train_acc)
    is_same = b["_in_train"] | b["_cont"].ge(0.90)

    mis = b[(b["novelty_tier"] == "same_protein") & ~is_same]
    if len(mis):
        fails.append(f"{len(mis)} rows tiered same_protein but match no training protein")
    leak = b[(b["novelty_tier"] != "same_protein") & is_same]
    if len(leak):
        fails.append(f"{len(leak)} rows tiered {leak.novelty_tier.unique().tolist()} "
                     f"but ARE a training protein")
    if not len(mis) and not len(leak):
        log("  [ok] same_protein tier is exactly the set matching a training protein")

    nf = b[b["novelty_tier"] == "novel_family"]
    bad_fam = nf[nf["family_l1"].isin(train_fam)]
    if len(bad_fam):
        fails.append(f"{len(bad_fam)} novel_family rows have a UniProt family present in training")
    else:
        log(f"  [ok] all {nf.uniprot_id.nunique()} novel_family targets are in families "
            "absent from training")

    # 6. field sanity ---------------------------------------------------------
    bad_seq = b[~b["target_sequence"].fillna("").map(lambda s: bool(s) and set(s) <= AA)]
    if len(bad_seq):
        fails.append(f"{len(bad_seq)} rows have non-amino-acid characters in target_sequence")
    else:
        log("  [ok] all target sequences are clean amino-acid strings")

    cov = b["cf_class"].notna().mean()
    if cov < 0.70:
        warns.append(f"ClassyFire coverage only {cov*100:.0f}% "
                     "(re-run 02_annotate_ligands.py --from-selected to finish)")

    print()
    for w in warns:
        log(f"WARN: {w}")
    if fails:
        for f in fails:
            log(f"FAIL: {f}")
        sys.exit(1)
    log(f"ALL CHECKS PASSED ({len(b)} pairs, {b.uniprot_id.nunique()} targets, "
        f"{b.ligand_smiles.nunique()} ligands)")


if __name__ == "__main__":
    main()

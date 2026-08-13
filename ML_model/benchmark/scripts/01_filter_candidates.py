#!/usr/bin/env python
"""
Step 01 - Clean BindingDB Kd and apply the hard disjointness gates.

Produces the candidate pool that step 03 selects from. Two things happen here:

  (A) LIGAND GATE (hard, drops rows). A candidate ligand is removed if it is
      identical to a training ligand (canonical SMILES or InChIKey skeleton) or
      ECFP4-Tanimoto >= MAX_LIGAND_TANIMOTO to any of them. Nothing chemically
      close to a training ligand survives at any novelty tier.

  (B) SAME-PROTEIN DETECTION (annotates, does not drop). A target is flagged
      `same_protein` when its UniProt accession appears in the training set or
      when 8-mer containment against a training chain is >= 0.9. Family-level
      homology is decided in step 03 from Pfam, which is a more reliable signal
      than any k-mer heuristic at 30% identity.

Also fixes the pKd convention: 9 - log10(Kd_nM), not 6 - log10(Kd_nM).
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from common import (BINDINGDB_TSV, CANDIDATES, CENSOR_CAPS_NM, COVERAGE, MAX_CHAIN_LEN,
                    MAX_LIGAND_TANIMOTO, MAX_PKD, MIN_CHAIN_LEN, MORGAN_BITS, MORGAN_RADIUS,
                    TRAIN_CSV, TRAIN_PROTEINS, TRAIN_PROT_TAX, WORK, ensure_dirs, log,
                    pkd_from_kd_nm, sanitize_seq)

RDLogger.DisableLog("rdApp.*")
K = 8


# --------------------------------------------------------------------------- #
def load_bindingdb() -> pd.DataFrame:
    log(f"loading {BINDINGDB_TSV}")
    d = pd.read_csv(BINDINGDB_TSV, sep="\t").rename(
        columns={"ID1": "drug_id", "X1": "smiles", "ID2": "uniprot_id",
                 "X2": "target_seq", "Y": "kd_nM"})
    n0 = len(d)
    caps = set(CENSOR_CAPS_NM)
    d = d[d["kd_nM"].gt(0) & ~d["kd_nM"].isin(caps)
          & d["smiles"].notna() & d["target_seq"].notna()].copy()
    log(f"  {n0} -> {len(d)} rows after dropping non-positive and censor-cap (>X) Kd values")

    d["target_seq"] = d["target_seq"].map(sanitize_seq)
    d["seq_len"] = d["target_seq"].str.len()
    d = d[d["seq_len"].between(MIN_CHAIN_LEN, MAX_CHAIN_LEN)]
    log(f"  {len(d)} rows after chain-length filter [{MIN_CHAIN_LEN}, {MAX_CHAIN_LEN}]")

    d["neg_log10_affinity_M"] = pkd_from_kd_nm(d["kd_nM"]).round(3)
    n_imp = d["neg_log10_affinity_M"].gt(MAX_PKD).sum()
    d = d[d["neg_log10_affinity_M"].le(MAX_PKD)]
    log(f"  dropped {n_imp} rows with implausible pKd > {MAX_PKD} (source unit errors)")
    log(f"  pKd: min={d.neg_log10_affinity_M.min():.2f} "
        f"median={d.neg_log10_affinity_M.median():.2f} max={d.neg_log10_affinity_M.max():.2f}")
    log(f"  unique targets={d.uniprot_id.nunique()} unique ligands={d.smiles.nunique()}")
    return d


# --------------------------------------------------------------------------- #
def ligand_gate(d: pd.DataFrame) -> pd.DataFrame:
    """Drop ligands identical or ECFP4-similar to any training ligand."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    train_smiles = pd.read_csv(TRAIN_CSV, usecols=["smiles"])["smiles"].dropna().unique()

    can_keys, skel_keys, fps = set(), set(), []
    for s in train_smiles:
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
    log(f"training ligands indexed: {len(fps)} fingerprints, {len(can_keys)} canonical SMILES")

    rows = {}
    uniq = d["smiles"].unique()
    for i, s in enumerate(uniq):
        m = Chem.MolFromSmiles(s)
        if m is None:
            rows[s] = (True, None, None, None)          # unparsable -> drop
            continue
        can = Chem.MolToSmiles(m)
        try:
            ik = Chem.MolToInchiKey(m)
            skel = ik.split("-")[0] if ik else None
        except Exception:
            skel = None
        if can in can_keys or (skel and skel in skel_keys):
            rows[s] = (True, can, skel, 1.0)
            continue
        mt = float(max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), fps)))
        rows[s] = (mt >= MAX_LIGAND_TANIMOTO, can, skel, round(mt, 3))
        if (i + 1) % 2000 == 0:
            log(f"  ligand scan {i + 1}/{len(uniq)}")

    d["_drop"] = d["smiles"].map(lambda s: rows[s][0])
    d["canonical_smiles"] = d["smiles"].map(lambda s: rows[s][1])
    d["inchikey_skeleton"] = d["smiles"].map(lambda s: rows[s][2])
    d["max_train_ligand_tanimoto"] = d["smiles"].map(lambda s: rows[s][3])

    kept = d[~d["_drop"]].drop(columns="_drop").copy()
    log(f"ligand gate: {len(d)} -> {len(kept)} rows "
        f"({d['smiles'].nunique()} -> {kept['smiles'].nunique()} unique ligands, "
        f"Tanimoto < {MAX_LIGAND_TANIMOTO})")
    return kept


# --------------------------------------------------------------------------- #
def same_protein_flags(target_seqs: list[str]) -> pd.DataFrame:
    """Best 8-mer containment of each target against the training chains."""
    tp = pd.read_csv(TRAIN_PROTEINS)
    log(f"training proteins: {len(tp)} unique chains")

    def kmers(s):
        s = sanitize_seq(s)
        return {s[i:i + K] for i in range(len(s) - K + 1)}

    index = defaultdict(set)
    train_sets = []
    for i, s in enumerate(tp["sequence"]):
        ks = kmers(s)
        train_sets.append(ks)
        for k in ks:
            index[k].add(i)

    out = []
    for n, seq in enumerate(target_seqs):
        qs = kmers(seq)
        hits = defaultdict(int)
        for k in qs:
            for j in index.get(k, ()):
                hits[j] += 1
        best_c, best_j = 0.0, None
        for j, c in hits.items():
            cont = c / max(1, min(len(qs), len(train_sets[j])))
            if cont > best_c:
                best_c, best_j = cont, j
        out.append({
            "target_seq": seq,
            "max_train_kmer_containment": round(best_c, 3),
            "closest_train_protein": tp["protein_id"].iloc[best_j] if best_j is not None else None,
        })
        if (n + 1) % 200 == 0:
            log(f"  protein scan {n + 1}/{len(target_seqs)}")
    return pd.DataFrame(out)


def training_uniprots() -> set[str]:
    tax = pd.read_csv(TRAIN_PROT_TAX, usecols=["uniprot_id"])
    u = {str(x).strip().upper() for x in tax["uniprot_id"].dropna() if str(x).strip()}
    log(f"training UniProt accessions: {len(u)}")
    return u


# --------------------------------------------------------------------------- #
def main() -> None:
    ensure_dirs()
    d = load_bindingdb()
    d = ligand_gate(d)

    targets = d["target_seq"].unique().tolist()
    log(f"scanning {len(targets)} unique surviving targets against training chains")
    prot = same_protein_flags(targets)
    d = d.merge(prot, on="target_seq", how="left")

    tu = training_uniprots()
    d["uniprot_in_training"] = d["uniprot_id"].astype(str).str.upper().str.split("-").str[0].isin(tu)
    d["is_same_protein"] = d["uniprot_in_training"] | d["max_train_kmer_containment"].ge(0.90)

    # one row per (target, ligand); keep the tightest Kd where a pair repeats
    d = d.sort_values("kd_nM").drop_duplicates(subset=["uniprot_id", "canonical_smiles"])
    log(f"unique (target, ligand) candidate pairs: {len(d)}")
    log(f"  same_protein pairs: {d.is_same_protein.sum()} "
        f"({d.loc[d.is_same_protein, 'uniprot_id'].nunique()} targets)")
    log(f"  novel-protein pairs: {(~d.is_same_protein).sum()} "
        f"({d.loc[~d.is_same_protein, 'uniprot_id'].nunique()} targets)")

    d.to_csv(CANDIDATES, index=False)
    log(f"wrote {CANDIDATES} ({len(d)} rows)")

    # FASTA of candidate targets so the tiers can be confirmed with mmseqs on the HPC
    fa = os.path.join(WORK, "candidate_targets.fasta")
    with open(fa, "w") as f:
        for u, s in d.drop_duplicates("uniprot_id")[["uniprot_id", "target_seq"]].values:
            f.write(f">{u}\n{s}\n")
    log(f"wrote {fa}")


if __name__ == "__main__":
    main()

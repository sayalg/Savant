#!/usr/bin/env python
"""
Step 03 — Build the fold-disjoint / ligand-disjoint BindingDB Kd benchmark.

A BindingDB (target, ligand, Kd) pair is kept only if:
  (A) PROTEIN sequence-similarity: no training chain matches the target at
      >= min_seq_id identity over >= coverage of both sequences  (mmseqs, cov-mode 0), AND
  (B) LIGAND SMILES-similarity: the ligand is neither identical (canonical SMILES /
      InChIKey skeleton) nor ECFP4-Tanimoto >= max_tanimoto to any training ligand.

Then ~n_pairs pairs are selected, stratified across pKd and capped per UniProt target.

Inputs : <training_with_seq_csv> or <training_fasta> (step 01), <bindingdb_tsv> (step 02),
         <training_csv> (ligand SMILES).
Output : <output_dir>/bindingdb_kd_benchmark_<N>.csv  + a summary markdown.
"""
import argparse
import os
import subprocess
import shutil
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator

from common import load_config, ensure_dirs, sanitize_seq, split_chains, write_fasta, log

RDLogger.DisableLog("rdApp.*")


# --------------------------------------------------------------------------- #
# Ligand identity / similarity
# --------------------------------------------------------------------------- #
def mol_keys(smiles):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None, None, None
    can = Chem.MolToSmiles(m)
    try:
        ik = Chem.MolToInchiKey(m)
        skel = ik.split("-")[0] if ik else None
    except Exception:
        skel = None
    return m, can, skel


def build_training_ligands(cfg):
    p, fi = cfg["paths"], cfg["filters"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=fi["morgan_radius"], fpSize=fi["morgan_bits"])
    smis = pd.read_csv(p["training_csv"], usecols=["smiles"])["smiles"].dropna().unique()
    can_keys, skel_keys, fps = set(), set(), []
    for s in smis:
        m, can, skel = mol_keys(s)
        if m is None:
            continue
        can_keys.add(can)
        if skel:
            skel_keys.add(skel)
        fps.append(gen.GetFingerprint(m))
    log(f"[03] training ligands: {len(smis)} smiles -> {len(can_keys)} canonical, {len(fps)} fingerprints")
    return gen, can_keys, skel_keys, fps


def ligand_is_overlapping(smiles, gen, can_keys, skel_keys, train_fps, max_tan):
    """Return (overlap: bool, canonical_smiles, max_tanimoto)."""
    m, can, skel = mol_keys(smiles)
    if m is None:
        return True, None, None                 # unparsable -> drop
    if can in can_keys:
        return True, can, 1.0
    if skel and skel in skel_keys:
        return True, can, 1.0
    fp = gen.GetFingerprint(m)
    max_tan_val = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) if train_fps else 0.0
    return (max_tan_val >= max_tan), can, round(float(max_tan_val), 3)


# --------------------------------------------------------------------------- #
# Protein sequence similarity
# --------------------------------------------------------------------------- #
def protein_overlap_mmseqs(cfg, target_seqs):
    """Return set of target ids (b{i}) that MATCH a training chain above threshold."""
    p, fi, rt = cfg["paths"], cfg["filters"], cfg["runtime"]
    work = p["work_dir"]
    qfa = os.path.join(work, "bdb_targets.fasta")
    write_fasta(((f"b{i}", s) for i, s in enumerate(target_seqs)), qfa)

    out_m8 = os.path.join(work, "bdb_vs_train.m8")
    tmp = os.path.join(work, "mmseqs_tmp")
    cmd = [
        "mmseqs", "easy-search", qfa, p["training_fasta"], out_m8, tmp,
        "--min-seq-id", str(fi["min_seq_id"]),
        "-c", str(fi["coverage"]), "--cov-mode", "0",
        "-s", str(fi["mmseqs_sensitivity"]),
        "--alignment-mode", "3",
        "--threads", str(rt["threads"]), "-v", "1",
    ]
    log(f"[03] mmseqs: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    overlapping = set()
    if os.path.exists(out_m8):
        with open(out_m8) as f:
            for line in f:
                overlapping.add(line.split("\t")[0])
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"[03] mmseqs: {len(overlapping)}/{len(target_seqs)} targets matched a training chain (excluded)")
    return overlapping


def _aligner():
    from Bio.Align import PairwiseAligner, substitution_matrices
    a = PairwiseAligner()
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score, a.extend_gap_score, a.mode = -11, -1, "local"
    return a


def _id_cov(aln, q, t):
    a = aln.align(q, t)[0]
    qb, tb = a.aligned
    nid = alen = 0
    for (qs, qe), (ts, te) in zip(qb, tb):
        seg = qe - qs
        alen += seg
        nid += sum(1 for i in range(seg) if q[qs + i] == t[ts + i])
    if not alen:
        return 0.0, 0.0, 0.0
    return nid / alen, (qb[-1][1] - qb[0][0]) / len(q), (tb[-1][1] - tb[0][0]) / len(t)


def protein_overlap_biopython(cfg, target_seqs):
    """Portable fallback: banded Smith-Waterman vs the training FASTA."""
    fi = cfg["filters"]
    train = [l.strip() for l in open(cfg["paths"]["training_fasta"]) if l and not l.startswith(">")]
    tl = np.array([len(c) for c in train])
    aln = _aligner()
    overlapping = set()
    for i, s in enumerate(target_seqs):
        q = sanitize_seq(s)
        L = len(q)
        idx = np.where((tl >= fi["coverage"] * L) & (tl <= L / fi["coverage"]))[0]
        for j in idx:
            iden, cq, ct = _id_cov(aln, q, train[j])
            if iden >= fi["min_seq_id"] and cq >= fi["coverage"] and ct >= fi["coverage"]:
                overlapping.add(f"b{i}")
                break
        if (i + 1) % 200 == 0:
            log(f"[03]   biopython protein scan {i + 1}/{len(target_seqs)}")
    log(f"[03] biopython: {len(overlapping)}/{len(target_seqs)} targets excluded")
    return overlapping


def best_train_identity(cfg, seqs):
    """Best identity of each seq to any training chain (for annotating the selected set).

    Cached per unique sequence so repeated targets (many ligands share one target)
    are aligned only once.
    """
    fi = cfg["filters"]
    train = [l.strip() for l in open(cfg["paths"]["training_fasta"]) if l and not l.startswith(">")]
    tl = np.array([len(c) for c in train])
    aln = _aligner()
    cache = {}
    for s in set(seqs):
        q = sanitize_seq(s)
        L = len(q)
        idx = np.where((tl >= 0.5 * L) & (tl <= 2.0 * L))[0]   # wider band, just for reporting
        best = 0.0
        for j in idx:
            iden, _, _ = _id_cov(aln, q, train[j])
            best = max(best, iden)
        cache[s] = round(best, 3)
    return [cache[s] for s in seqs]


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def stratified_select(df, cfg):
    sel_cfg = cfg["selection"]
    rng = np.random.default_rng(sel_cfg["random_seed"])
    edges = sel_cfg["pkd_bins"]
    labels = [f"bin{i}" for i in range(len(edges) - 1)]
    df = df.copy()
    df["pkd_bin"] = pd.cut(df["neg_log10_affinity_M"], bins=edges, labels=labels)
    N = sel_cfg["n_pairs"]
    per_bin = {b: N // len(labels) for b in labels}
    for b in labels[: N % len(labels)]:
        per_bin[b] += 1

    chosen_idx, used_per_uniprot = [], {}
    for b in labels:
        sub = df[df["pkd_bin"] == b].sample(frac=1.0, random_state=int(rng.integers(1e9)))
        # round-robin across UniProt targets, respecting the per-target cap
        groups = {u: list(idx) for u, idx in sub.groupby("uniprot_id").groups.items()}
        order = sorted(groups, key=lambda u: -len(groups[u]))
        picked = 0
        while picked < per_bin[b] and any(groups.values()):
            for u in order:
                if not groups[u]:
                    continue
                if used_per_uniprot.get(u, 0) >= sel_cfg["max_per_uniprot"]:
                    groups[u] = []
                    continue
                chosen_idx.append(groups[u].pop(0))
                used_per_uniprot[u] = used_per_uniprot.get(u, 0) + 1
                picked += 1
                if picked >= per_bin[b]:
                    break
    sel = df.loc[chosen_idx]
    if len(sel) < N:   # top up from anything remaining (still respecting the cap)
        rest = df.drop(index=sel.index)
        for idx, row in rest.iterrows():
            if used_per_uniprot.get(row["uniprot_id"], 0) < sel_cfg["max_per_uniprot"]:
                sel = pd.concat([sel, rest.loc[[idx]]])
                used_per_uniprot[row["uniprot_id"]] = used_per_uniprot.get(row["uniprot_id"], 0) + 1
            if len(sel) >= N:
                break
    return sel.head(N)


# --------------------------------------------------------------------------- #
def main(config_path):
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    p, fi = cfg["paths"], cfg["filters"]

    # ---- load + clean BindingDB Kd ----
    log(f"[03] loading BindingDB: {p['bindingdb_tsv']}")
    bdb = pd.read_csv(p["bindingdb_tsv"], sep="\t")
    bdb = bdb.rename(columns={"ID1": "drug_id", "X1": "smiles", "ID2": "uniprot_id", "X2": "target_seq", "Y": "kd_nM"})
    n0 = len(bdb)
    caps = set(fi["drop_censor_caps_nM"])
    bdb = bdb[(bdb["kd_nM"] > 0) & (~bdb["kd_nM"].isin(caps)) & bdb["smiles"].notna() & bdb["target_seq"].notna()]
    bdb["neg_log10_affinity_M"] = (9 - np.log10(bdb["kd_nM"])).round(3)
    log(f"[03] Kd rows {n0} -> {len(bdb)} after dropping non-positive / censor-cap values")
    log(f"[03] unique targets={bdb['target_seq'].nunique()} unique ligands={bdb['smiles'].nunique()}")

    # ---- (A) protein sequence-similarity filter over UNIQUE targets ----
    targets = bdb["target_seq"].dropna().unique().tolist()
    tgt_id = {s: f"b{i}" for i, s in enumerate(targets)}
    target_sane = [sanitize_seq(t) for t in targets]
    method = fi["protein_method"]
    if method == "mmseqs" and shutil.which("mmseqs") is None:
        log("[03] WARNING: mmseqs not found on PATH; falling back to biopython")
        method = "biopython"
    overlapping = (protein_overlap_mmseqs(cfg, target_sane) if method == "mmseqs"
                   else protein_overlap_biopython(cfg, target_sane))
    disjoint_targets = {s for s in targets if tgt_id[s] not in overlapping}
    log(f"[03] fold-disjoint targets: {len(disjoint_targets)}/{len(targets)}")
    bdb = bdb[bdb["target_seq"].isin(disjoint_targets)]
    log(f"[03] pairs on fold-disjoint targets: {len(bdb)}")
    if bdb.empty:
        log("[03] No fold-disjoint targets remain. Nothing to build."); return

    # ---- (B) ligand SMILES-similarity filter over UNIQUE candidate ligands ----
    gen, can_keys, skel_keys, train_fps = build_training_ligands(cfg)
    cache = {}
    for s in bdb["smiles"].unique():
        cache[s] = ligand_is_overlapping(s, gen, can_keys, skel_keys, train_fps, fi["max_tanimoto"])
    bdb["_ovl"] = bdb["smiles"].map(lambda s: cache[s][0])
    bdb["ligand_smiles"] = bdb["smiles"].map(lambda s: cache[s][1])
    bdb["max_train_ligand_tanimoto"] = bdb["smiles"].map(lambda s: cache[s][2])
    kept = bdb[~bdb["_ovl"]].copy()
    log(f"[03] ligand filter: {len(bdb)} -> {len(kept)} pairs with dissimilar ligands "
        f"(Tanimoto < {fi['max_tanimoto']}, no exact/skeleton match)")
    if kept.empty:
        log("[03] No ligand-dissimilar pairs remain. Nothing to build."); return

    # one row per (target, ligand); keep the most precise / representative
    kept = kept.drop_duplicates(subset=["uniprot_id", "ligand_smiles"])
    log(f"[03] unique (target, ligand) candidate pairs: {len(kept)}")

    # ---- select ----
    sel = stratified_select(kept, cfg).reset_index(drop=True)
    log(f"[03] selected {len(sel)} pairs")

    # ---- annotate selected set with best training-protein identity (biopython, cheap) ----
    sel["max_train_protein_identity"] = best_train_identity(cfg, sel["target_seq"].tolist())

    out = pd.DataFrame({
        "uniprot_id": sel["uniprot_id"],
        "ligand_smiles": sel["ligand_smiles"],
        "target_sequence": sel["target_seq"],
        "kd_nM": sel["kd_nM"],
        "neg_log10_affinity_M": sel["neg_log10_affinity_M"],
        "deltaG_kcal_per_mol": (-1.364 * sel["neg_log10_affinity_M"]).round(3),
        "max_train_protein_identity": sel["max_train_protein_identity"],
        "max_train_ligand_tanimoto": sel["max_train_ligand_tanimoto"],
    }).sort_values("neg_log10_affinity_M").reset_index(drop=True)

    N = len(out)
    dest = os.path.join(p["output_dir"], f"bindingdb_kd_benchmark_{N}.csv")
    out.to_csv(dest, index=False)
    log(f"[03] wrote {dest} ({N} rows)")

    # summary
    summ = os.path.join(p["output_dir"], f"bindingdb_kd_benchmark_{N}_summary.md")
    with open(summ, "w") as f:
        f.write(f"# BindingDB Kd benchmark ({N} pairs)\n\n")
        f.write(f"- unique targets: {out['uniprot_id'].nunique()}\n")
        f.write(f"- pKd range: {out['neg_log10_affinity_M'].min()}–{out['neg_log10_affinity_M'].max()} "
                f"(median {out['neg_log10_affinity_M'].median()})\n")
        f.write(f"- max training-protein identity (should be < {fi['min_seq_id']}): "
                f"max={out['max_train_protein_identity'].max()}, median={out['max_train_protein_identity'].median()}\n")
        f.write(f"- max training-ligand Tanimoto (should be < {fi['max_tanimoto']}): "
                f"max={out['max_train_ligand_tanimoto'].max()}, median={out['max_train_ligand_tanimoto'].median()}\n")
        f.write("\nFilters: protein seq-identity "
                f">= {fi['min_seq_id']} over >= {fi['coverage']} coverage (excluded); "
                f"ligand ECFP{2*fi['morgan_radius']} Tanimoto >= {fi['max_tanimoto']} (excluded).\n")
    log(f"[03] wrote {summ}")
    log("[03] DONE.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)

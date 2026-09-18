#!/usr/bin/env python
"""Does the new ChEMBL/BindingDB data break the v3 benchmark's ligand-novelty gate?

    python audit_expanded_vs_benchmark.py

The v3 benchmark's whole validity rests on one property: every benchmark ligand is at
ECFP4 Tanimoto <= 0.50 from anything in training. That was verified once for input.csv's
991,876 ligands (max observed 0.494). It is NOT automatically true for the 245,006 NEW
ligands added by get_more_data.py - that script only excluded exact (seq, smiles) pair
matches, which is the weakest possible gate:

  * exact string matching misses canonicalisation variants, salts and stereochemistry
  * the 0.50 Tanimoto gate was never applied to the new ligands at all
  * the benchmark is TEMPORAL by design (2025+ measurements, absent from ChEMBL v34) and
    the ChEMBL crawl had no date filter, so a current release may contain the very
    measurements the benchmark tests on

Three checks, in increasing severity:

  1. ligand skeleton present    - benchmark ligand appears anywhere in the new data
  2. max ECFP4 Tanimoto >= 0.50 - the actual gate the benchmark is built on, violated
  3. PAIR present               - same protein AND same ligand skeleton, i.e. the model
                                  was trained on the exact measurement being tested

Check 3 is the one that would invalidate a headline number outright.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import inchi, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
EXPANDED = ROOT / "data" / "expanded_kd_ki.csv"
BENCH = REPO / "ML_model" / "benchmark" / "eval" / "output" / "bench_scored.csv"
OUT = ROOT / "work"
GATE = 0.50

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def prep(smiles, want_fp=True):
    keys, fps, keep = [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        keep.append(s)
        try:
            keys.append(inchi.MolToInchiKey(m)[:14])
        except Exception:
            keys.append(None)
        if want_fp:
            fps.append(gen.GetFingerprint(m))
    return keys, fps, keep


def main():
    bm = pd.read_csv(BENCH, usecols=["target_sequence", "ligand_smiles", "uniprot_id"]).dropna()
    b_lig = sorted(bm.ligand_smiles.unique())
    log(f"benchmark: {len(bm)} pairs, {bm.uniprot_id.nunique()} targets, {len(b_lig)} ligands")

    exp = pd.read_csv(EXPANDED, usecols=["seq", "smiles"])
    e_lig = sorted(exp.smiles.unique())
    log(f"expanded:  {len(exp):,} rows, {len(e_lig):,} unique ligands, "
        f"{exp.seq.nunique():,} proteins")

    # --- protein overlap (expected and fine - benchmark is protein-overlapping by design)
    b_seqs = set(bm.target_sequence)
    shared_prot = b_seqs & set(exp.seq.unique())
    log(f"benchmark targets present in the new data: {len(shared_prot)}/{len(b_seqs)}")

    log("fingerprinting benchmark ligands")
    b_keys, b_fps, b_keep = prep(b_lig)
    log("fingerprinting new ligands (slow step)")
    e_keys, e_fps, e_keep = prep(e_lig)
    log(f"  {len(e_fps):,} fingerprinted")

    e_key_set = {k for k in e_keys if k}
    # skeleton -> set of protein sequences it was measured against, for the pair check
    key_of_smiles = dict(zip(e_keep, e_keys))
    pair_index = {}
    for seq, smi in zip(exp.seq.to_numpy(), exp.smiles.to_numpy()):
        k = key_of_smiles.get(smi)
        if k:
            pair_index.setdefault(k, set()).add(seq)

    log(f"computing max Tanimoto for {len(b_fps)} benchmark ligands vs {len(e_fps):,} new")
    mx = np.zeros(len(b_fps), np.float32)
    for i, fp in enumerate(b_fps):
        mx[i] = max(DataStructs.BulkTanimotoSimilarity(fp, e_fps))
        if (i + 1) % 50 == 0:
            log(f"  {i+1}/{len(b_fps)}")

    # benchmark pairs, for the pair-level check
    bench_pairs = set(zip(bm.target_sequence, bm.ligand_smiles))
    lig_to_seqs = {}
    for seq, smi in bench_pairs:
        lig_to_seqs.setdefault(smi, set()).add(seq)

    rows = []
    for smi, key, m in zip(b_keep, b_keys, mx):
        trained_seqs = pair_index.get(key, set()) if key else set()
        bench_seqs = lig_to_seqs.get(smi, set())
        rows.append({"ligand_smiles": smi, "skeleton_in_new": bool(key and key in e_key_set),
                     "max_tanimoto_to_new": float(m),
                     "gate_violated": bool(m >= GATE),
                     "pair_in_new": bool(trained_seqs & bench_seqs)})
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "audit_expanded_vs_benchmark.csv", index=False)

    n = len(res)
    print("\n================ RESULT ================")
    print(f"benchmark ligands audited: {n}")
    print(f"  1. skeleton present in new data : {res.skeleton_in_new.sum():>4} "
          f"({100*res.skeleton_in_new.mean():.1f}%)")
    print(f"  2. max Tanimoto >= {GATE:.2f} (GATE)   : {res.gate_violated.sum():>4} "
          f"({100*res.gate_violated.mean():.1f}%)")
    print(f"  3. SAME PAIR in new data        : {res.pair_in_new.sum():>4} "
          f"({100*res.pair_in_new.mean():.1f}%)   <- invalidating if non-zero")
    print(f"\nmax-Tanimoto distribution: median {np.median(mx):.3f}  "
          f"mean {mx.mean():.3f}  p95 {np.percentile(mx,95):.3f}  max {mx.max():.3f}")
    for t in (0.9, 0.7, 0.5, 0.4):
        print(f"  >= {t:.1f}: {int((mx>=t).sum()):>4}")

    if res.pair_in_new.any() or res.gate_violated.any():
        print("\nThe 0.4297 benchmark number is NOT safe to quote as-is. Options:")
        print("  a) drop the offending ligands from the training corpus and retrain")
        print("  b) report the benchmark on the uncontaminated subset only")
        print("  c) rebuild the benchmark against the new corpus")
    else:
        print("\nGate holds. The benchmark remains valid against the expanded corpus.")
    print(f"\nper-ligand detail -> {OUT/'audit_expanded_vs_benchmark.csv'}")


if __name__ == "__main__":
    main()

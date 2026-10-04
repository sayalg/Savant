#!/usr/bin/env python
"""How much of PLABench's filtered ChEMBL35 set is novel to SAVANT?

    python scripts/02_audit_contamination.py

PLABench removed test ligands within ECFP4 Tanimoto 0.85 of any of 14 training
corpora. That protects their 9 models. It does not protect ours, for 2 reasons:

  * Savant's corpus (jglaser/binding_affinity: BindingDB, PDBbind-CN, BioLiP,
    Binding MOAD) is not one of the 14. It overlaps several of them, so some
    protection carries over, but the overlap is partial and unquantified.
  * Their gate is 0.85, which removes near-duplicates. Ours is 0.50, which is
    the standard our own benchmark is built on. A ligand at Tanimoto 0.7 to a
    training compound passes their filter and fails ours.

So this script computes, for every PLABench ligand, the maximum ECFP4 Tanimoto
to any ligand in Savant's training corpus, and flags the ones that would not
have survived our own gate. 04_evaluate.py reports the benchmark twice: once on
everything, and once on the subset that clears 0.50. The second is the number
that can be compared to our own benchmark results without an asterisk.

Cost: ~3.6k test ligands against ~992k training ligands. Fingerprints for the
training side are cached, because that is the slow half and it does not change.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import inchi, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[2]
TRAIN = REPO / "input" / "input.csv"
PAIRS = ROOT / "data" / "chembl35" / "chembl35_filtered_input.csv"
WORK = ROOT / "work"
OUT = ROOT / "output"

GATE = 0.50
RADIUS, BITS = 2, 2048
gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=BITS)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fps_and_keys(smiles, want_keys=False):
    """ECFP4 fingerprints, plus InChIKey skeletons, skipping unparseable SMILES."""
    fps, keys, kept = [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(gen.GetFingerprint(m))
        kept.append(s)
        if want_keys:
            try:
                keys.append(inchi.MolToInchiKey(m)[:14])
            except Exception:
                keys.append(None)
    return fps, keys, kept


def training_fingerprints(cache):
    if cache.exists():
        import pickle
        with open(cache, "rb") as fh:
            payload = pickle.load(fh)
        log(f"reusing {cache.name}: {len(payload['fps']):,} training fingerprints")
        return payload["fps"], payload["keys"]

    log(f"fingerprinting the training corpus from {TRAIN.name} (slow, cached after)")
    smiles = set()
    for ch in pd.read_csv(TRAIN, usecols=["smiles"], chunksize=400_000):
        smiles.update(ch.smiles.astype(str))
    smiles = sorted(smiles)
    log(f"  {len(smiles):,} unique training ligands")
    fps, keys, _ = fps_and_keys(smiles, want_keys=True)
    log(f"  {len(fps):,} fingerprinted")

    import pickle
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump({"fps": fps, "keys": keys}, fh, protocol=4)
    return fps, keys


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", type=float, default=GATE)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(PAIRS)
    test_smiles = sorted(d.compound_iso_smiles.unique())
    log(f"PLABench filtered set: {len(d):,} pairs, {len(test_smiles):,} unique ligands")

    t_fps, t_keys, t_kept = fps_and_keys(test_smiles, want_keys=True)
    log(f"  {len(t_fps):,} test ligands fingerprinted")

    r_fps, r_keys = training_fingerprints(WORK / "train_fps.pkl")
    r_keyset = {k for k in r_keys if k}

    ## Running max over chunks of the training side: the full cross product is
    ## ~3.6e9 comparisons and materialising it would not fit in memory.
    log(f"computing max Tanimoto for {len(t_fps):,} test vs {len(r_fps):,} training")
    mx = np.zeros(len(t_fps), np.float32)
    CH = 20_000
    t0 = time.time()
    for st in range(0, len(r_fps), CH):
        block = r_fps[st:st + CH]
        for i, fp in enumerate(t_fps):
            s = DataStructs.BulkTanimotoSimilarity(fp, block)
            m = max(s)
            if m > mx[i]:
                mx[i] = m
        if (st // CH) % 5 == 0:
            frac = min(st + CH, len(r_fps)) / len(r_fps)
            log(f"  {100*frac:5.1f}%  ({time.time()-t0:.0f}s)")

    skel = np.array([bool(k and k in r_keyset) for k in t_keys])
    res = pd.DataFrame({"compound_iso_smiles": t_kept,
                        "max_tanimoto_to_savant_training": mx,
                        "skeleton_in_savant_training": skel})
    res["fails_savant_gate"] = (res.max_tanimoto_to_savant_training >= args.gate) | res.skeleton_in_savant_training
    res.to_csv(OUT / "contamination_audit.csv", index=False)

    merged = d.merge(res, on="compound_iso_smiles", how="left")
    merged.to_csv(OUT / "pairs_with_audit.csv", index=False)

    n_lig, n_bad = len(res), int(res.fails_savant_gate.sum())
    pair_bad = int(merged.fails_savant_gate.fillna(False).sum())
    print("\n================ RESULT ================")
    print(f"ligands audited                 : {n_lig:,}")
    print(f"  InChIKey skeleton in training : {int(skel.sum()):,} ({100*skel.mean():.1f}%)")
    print(f"  max Tanimoto >= {args.gate:.2f}          : {int((mx>=args.gate).sum()):,} "
          f"({100*(mx>=args.gate).mean():.1f}%)")
    print(f"  fails Savant's gate           : {n_bad:,} ({100*n_bad/n_lig:.1f}%)")
    print(f"pairs affected                  : {pair_bad:,} of {len(merged):,} "
          f"({100*pair_bad/len(merged):.1f}%)")
    print(f"\nmax-Tanimoto distribution: median {np.median(mx):.3f}  mean {mx.mean():.3f}  "
          f"p95 {np.percentile(mx,95):.3f}  max {mx.max():.3f}")
    for t in (0.9, 0.85, 0.7, 0.5, 0.4):
        print(f"  >= {t:.2f}: {int((mx>=t).sum()):>5}")
    print(f"\nnote: PLABench's own gate is 0.85; anything at or above that in the "
          f"rows above\nis a ligand their filter should have removed for their 14 "
          f"corpora but which\nremains close to OURS.")
    print(f"\nper-ligand detail -> {OUT/'contamination_audit.csv'}")


if __name__ == "__main__":
    main()

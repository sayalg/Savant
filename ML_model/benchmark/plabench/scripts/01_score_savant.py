#!/usr/bin/env python
"""Score Savant on PLABench's leakage-filtered ChEMBL35 set.

    python scripts/01_score_savant.py                 # all 71 targets
    python scripts/01_score_savant.py --limit 3       # smoke test

PLABench (Wang & Cheng 2026, Zenodo 10.5281/zenodo.22716174) is an independent
benchmark we had no hand in building. Scoring Savant on it tests the method
against 71 targets instead of the 20 our own benchmark spans, and against 9
other methods whose numbers were produced by someone else.

What this script does NOT establish on its own: PLABench filtered its test set
against 14 training corpora, but Savant's corpus is not one of them, and its
gate is ECFP4 Tanimoto 0.85 against our 0.50. Run 02_audit_contamination.py
before quoting any number from here -- the decontaminated subset is the honest
comparison, and the full-set number is reported only beside it.

Scoring is per target rather than over the flat pair list, because the protein
encoder is the only expensive step and scoring a target's ligands together
encodes its sequence once. That is the same cost structure the method claims.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

PAIRS = ROOT / "data" / "chembl35" / "chembl35_filtered_input.csv"
OUT = ROOT / "output"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N targets (smoke test)")
    ap.add_argument("--model", default="esm650")
    ap.add_argument("--out", default=str(OUT / "savant_chembl35_filtered.csv"))
    args = ap.parse_args()

    from savant.model import Savant

    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(PAIRS)
    d["uniprot"] = d.compound_id.str.split("_").str[0]
    log(f"{len(d):,} pairs | {d.uniprot.nunique()} targets | "
        f"{d.compound_iso_smiles.nunique():,} ligands")

    ## One sequence per target is an assumption the grouping below relies on.
    ## PLABench names targets by UniProt accession, but nothing guarantees a
    ## single sequence per accession, and silently scoring 2 sequences as 1
    ## target would corrupt every per-target metric downstream.
    per_target_seqs = d.groupby("uniprot").target_sequence.nunique()
    bad = per_target_seqs[per_target_seqs > 1]
    if len(bad):
        sys.exit(f"{len(bad)} targets carry more than 1 sequence: {list(bad.index)[:5]}")

    sv = Savant(model=args.model)
    log(f"loaded {sv!r}")

    targets = sorted(d.uniprot.unique())
    if args.limit:
        targets = targets[:args.limit]
        log(f"--limit {args.limit}: scoring {len(targets)} targets only")

    rows, t0 = [], time.time()
    for i, uid in enumerate(targets, 1):
        g = d[d.uniprot == uid]
        seq = g.target_sequence.iloc[0]
        pk = sv.predict([seq] * len(g), list(g.compound_iso_smiles)).pKd.to_numpy()
        out = g[["compound_id", "uniprot", "compound_iso_smiles", "affinity"]].copy()
        out["savant_pk"] = pk
        rows.append(out)
        if i % 10 == 0 or i == len(targets):
            done = sum(len(r) for r in rows)
            log(f"  {i}/{len(targets)} targets, {done:,} pairs, "
                f"{time.time()-t0:.0f}s elapsed")

    res = pd.concat(rows, ignore_index=True)
    res.to_csv(args.out, index=False)
    el = time.time() - t0
    log(f"wrote {args.out}  ({len(res):,} rows)")
    log(f"{el:.0f}s total, {el/max(len(res),1)*1000:.1f} ms per pair "
        f"({len(targets)} protein encodings amortised across them)")

    y, p = res.affinity.to_numpy(), res.savant_pk.to_numpy()
    log(f"quick look: pooled r {np.corrcoef(y, p)[0,1]:.4f}  "
        f"RMSE {np.sqrt(((y-p)**2).mean()):.4f}  "
        f"pred sd/true sd {p.std()/y.std():.3f}")


if __name__ == "__main__":
    main()

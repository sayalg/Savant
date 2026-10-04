#!/usr/bin/env python
"""Savant against the 9 PLABench methods, in their protocol and in ours.

    python scripts/03_evaluate.py

Reports 3 things, in increasing order of how much they can be relied on.

1. PLABench's own protocol, so the number sits in their published table without
   an adjustment. Their headline "Pearson" is the N-weighted mean of per-target
   Pearson, which this script reproduces exactly for their models from their own
   per-target file before applying the same computation to Savant. That
   reproduction is the check that we are scoring the same quantity they are.

2. The between-target / within-target decomposition, which their metrics do not
   compute. Their weighted per-target mean is already a within-target quantity,
   so the interesting addition is the between-target component and the pooled
   figure that blends the 2.

3. Both of the above restricted to the ligands that clear Savant's own ECFP4
   Tanimoto 0.50 gate against its training corpus. PLABench filtered at 0.85
   against 14 corpora, none of which is ours, so the full-set number is the
   optimistic one and this is the number to quote.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCORES = ROOT / "output" / "savant_chembl35_filtered.csv"
AUDIT = ROOT / "output" / "contamination_audit.csv"
THEIRS = ROOT / "data" / "plabench_results" / "chembl35"
OUT = ROOT / "output"

MIN_PAIRS = 2


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def per_target(df, truth="affinity", pred="savant_pk", group="uniprot"):
    rows = []
    for uid, g in df.groupby(group):
        if len(g) < MIN_PAIRS or g[pred].std() == 0 or g[truth].std() == 0:
            continue
        rows.append(dict(Target=uid, N=len(g),
                         Pearson=np.corrcoef(g[truth], g[pred])[0, 1],
                         Spearman=spearmanr(g[truth], g[pred]).correlation,
                         RMSE=float(np.sqrt(((g[truth] - g[pred]) ** 2).mean()))))
    return pd.DataFrame(rows)


def weighted(pt, col):
    s = pt.dropna(subset=[col])
    return float(np.average(s[col], weights=s.N)) if len(s) else np.nan


def decompose(y, p, g):
    """Between-target and within-target components of the pooled correlation."""
    d = pd.DataFrame({"y": y, "p": p, "g": g})
    m = d.groupby("g")[["y", "p"]].mean()
    between = float(np.corrcoef(m.y, m.p)[0, 1]) if len(m) > 2 and m.p.std() > 0 else np.nan
    c = d.join(m, on="g", rsuffix="_m")
    cy, cp = c.y - c.y_m, c.p - c.p_m
    within = float(np.corrcoef(cy, cp)[0, 1]) if cy.std() > 1e-9 and cp.std() > 1e-9 else np.nan
    return between, within


def savant_row(df, label):
    pt = per_target(df)
    y, p, g = df.affinity.to_numpy(), df.savant_pk.to_numpy(), df.uniprot.to_numpy()
    b, w = decompose(y, p, g)
    return dict(Model=f"Savant ({label})", N_total=len(df), N_targets=pt.Target.nunique(),
                RMSE=weighted(pt, "RMSE"), Pearson=weighted(pt, "Pearson"),
                Spearman=weighted(pt, "Spearman"),
                Between_r=b, Within_r=w,
                Pooled_r=float(np.corrcoef(y, p)[0, 1]),
                Pred_spread=float(p.std() / y.std())), pt


def main():
    d = pd.read_csv(SCORES)
    log(f"Savant predictions: {len(d):,} pairs, {d.uniprot.nunique()} targets")

    ## --- 1. reproduce their published numbers from their own per-target file
    ptf = pd.read_csv(THEIRS / "per_target_evaluation_chembl35_filtered.csv")
    pub = pd.read_csv(THEIRS / "weighted_summary_chembl35_filtered.csv").set_index("Model")
    log("reproducing PLABench's published weighted means from their per-target file")
    worst = 0.0
    for m, g in ptf.groupby("Model"):
        got = weighted(g.rename(columns={"N": "N"}), "Pearson")
        want = float(pub.loc[m, "Pearson"])
        worst = max(worst, abs(got - want))
        flag = "ok " if abs(got - want) < 5e-4 else "MISMATCH"
        log(f"  {flag} {m:11s} recomputed {got:.4f}  published {want:.4f}")
    if worst >= 5e-4:
        raise SystemExit(f"could not reproduce PLABench's protocol (max diff {worst:.4f}); "
                         "do not compare Savant to their table until this agrees")
    log(f"protocol reproduced, max difference {worst:.2e}")

    ## --- 2. Savant, full filtered set and the decontaminated subset
    rows, pts = [], {}
    r, pt = savant_row(d, "all")
    rows.append(r); pts["all"] = pt

    if AUDIT.exists():
        a = pd.read_csv(AUDIT)
        d2 = d.merge(a[["compound_iso_smiles", "fails_savant_gate"]],
                     on="compound_iso_smiles", how="left")
        clean = d2[~d2.fails_savant_gate.fillna(False)]
        log(f"Savant gate: {len(d2)-len(clean):,} of {len(d2):,} pairs removed "
            f"({100*(len(d2)-len(clean))/len(d2):.1f}%)")
        r, pt = savant_row(clean, "Savant-gate 0.50")
        rows.append(r); pts["clean"] = pt
    else:
        log("no contamination audit found - run 02_audit_contamination.py; "
            "the full-set number below is optimistic and must not be quoted alone")

    sav = pd.DataFrame(rows)

    ## --- 3. combined table against their published models
    theirs = pub.reset_index()[["Model", "N_total", "N_targets", "RMSE",
                                "Pearson", "Spearman"]]
    comb = pd.concat([theirs, sav[["Model", "N_total", "N_targets", "RMSE",
                                   "Pearson", "Spearman"]]], ignore_index=True)
    comb = comb.sort_values("Pearson", ascending=False)
    comb.to_csv(OUT / "comparison_vs_plabench.csv", index=False)
    sav.to_csv(OUT / "savant_metrics.csv", index=False)
    for k, v in pts.items():
        v.to_csv(OUT / f"savant_per_target_{k}.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=========== PLABench filtered ChEMBL35, their protocol ===========")
    print("(Pearson/Spearman/RMSE are N-weighted means of per-target values)\n")
    print(comb.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=========== decomposition, which their metrics omit ===========\n")
    print(sav[["Model", "Between_r", "Within_r", "Pooled_r", "Pred_spread"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log(f"\nwrote {OUT/'comparison_vs_plabench.csv'}")


if __name__ == "__main__":
    main()

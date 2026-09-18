#!/usr/bin/env python
"""Paired comparison of two encoders from run_2x2_split_test.py's saved predictions.

    python scripts/compare_encoders.py --a esmc-300m --b esmfold2
    python scripts/compare_encoders.py --a esmc-300m --b esmfold2 --metric pearson

Why paired, and why this script exists
--------------------------------------
Two encoders are scored on the SAME pairs against the SAME truth, so their correlations
are dependent. Comparing them by eye - or with two independent confidence intervals -
throws away most of the available power: a 0.032 gap in Part A at n=438 proteins came out
p=0.58 unpaired, which says nothing at all. Steiger's (1980) test for dependent
correlations uses the agreement between the two models' own predictions and resolves
differences several times smaller.

Reported per 2x2 cell, for whichever metric is asked for:

  within_r  (default) - ranking ligands WITHIN a target: the screening claim
  pearson             - pooled, mixes within- and between-target placement

The two can and do disagree. Read the one that matches the claim being made.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def centered(y, p, g):
    """Subtract each target's mean from truth and prediction."""
    c, u = pd.factorize(g)
    ng = len(u)
    cnt = np.bincount(c, minlength=ng).astype(np.float64)
    my = np.bincount(c, weights=y, minlength=ng) / np.maximum(cnt, 1)
    mp = np.bincount(c, weights=p, minlength=ng) / np.maximum(cnt, 1)
    return y - my[c], p - mp[c], ng


def steiger(r12, r13, r23, n):
    """Steiger (1980): do two DEPENDENT correlations with a common variable differ?

    r12 = truth vs A, r13 = truth vs B, r23 = A vs B. Returns (t, p, df).
    """
    if not all(np.isfinite([r12, r13, r23])) or n < 5:
        return np.nan, np.nan, n
    R = 1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23
    rm = (r12 + r13) / 2
    den = 2*R*(n - 1)/(n - 3) + rm**2 * (1 - r23)**3
    if R <= 0 or den <= 0:
        return np.nan, np.nan, n
    t = (r12 - r13) * np.sqrt((n - 1) * (1 + r23) / den)
    return float(t), float(2 * student_t.sf(abs(t), n - 3)), n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "work" / "split_test"))
    ap.add_argument("--a", required=True, help="first encoder name (the baseline)")
    ap.add_argument("--b", required=True, help="second encoder name")
    ap.add_argument("--metric", default="within_r", choices=["within_r", "pearson"])
    args = ap.parse_args()

    d = Path(args.dir)
    f = d / "part_b_predictions.parquet"
    if not f.exists():
        raise SystemExit(
            f"{f} not found.\n"
            "Re-run run_2x2_split_test.py - older runs saved only aggregate metrics, and a "
            "paired test cannot be reconstructed from those.")
    P = pd.read_parquet(f)
    have = sorted(P.encoder.unique())
    for name in (args.a, args.b):
        if name not in have:
            raise SystemExit(f"encoder {name!r} not in {f.name}; present: {have}")

    rows = []
    for cell in P.cell.unique():
        A = P[(P.encoder == args.a) & (P.cell == cell)].sort_values(["pi", "li"])
        B = P[(P.encoder == args.b) & (P.cell == cell)].sort_values(["pi", "li"])
        ## The pairing is the whole basis of the test. If the two encoders were somehow
        ## scored on different rows, a paired test would be invalid in a way that produces
        ## a plausible-looking number rather than an error - so check, do not assume.
        if len(A) != len(B) or not np.array_equal(A.pi.to_numpy(), B.pi.to_numpy()) \
                or not np.array_equal(A.li.to_numpy(), B.li.to_numpy()):
            raise SystemExit(f"{cell}: {args.a} and {args.b} were scored on different "
                             f"rows - cannot pair them")
        y = A.y.to_numpy(np.float64)
        pa, pb = A.p.to_numpy(np.float64), B.p.to_numpy(np.float64)

        if args.metric == "within_r":
            cy, ca, ng = centered(y, pa, A.pi.to_numpy())
            _, cb, _ = centered(y, pb, A.pi.to_numpy())
            n_eff = len(y) - ng          # within-target degrees of freedom, not row count
        else:
            cy, ca, cb, n_eff = y, pa, pb, len(y)

        r12 = float(np.corrcoef(cy, ca)[0, 1])
        r13 = float(np.corrcoef(cy, cb)[0, 1])
        r23 = float(np.corrcoef(ca, cb)[0, 1])
        ## Arguments ordered (B, A) so the t statistic carries the same sign as `delta`
        ## (= B - A). With (A, B) the test is identical and the p-value unchanged, but t
        ## comes back negative whenever delta is positive - a contradiction between two
        ## adjacent columns that is very easy to misread in a write-up.
        t, p, df = steiger(r13, r12, r23, n_eff)
        rows.append({"cell": cell, "n": len(y), "df": df,
                     args.a: r12, args.b: r13, "delta": r13 - r12,
                     "model_agreement": r23, "steiger_t": t, "p_value": p})

    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\n=== {args.metric}: {args.b} vs {args.a} (paired, Steiger) ===")
    print(f"delta > 0 means {args.b} is better\n")
    print(R.round(4).to_string(index=False))
    sig = R[R.p_value < 0.05]
    print(f"\n{len(sig)} of {len(R)} cells differ at p < 0.05"
          + (f": {', '.join(sig.cell)}" if len(sig) else ""))
    out = Path(args.dir) / f"compare_{args.a}_vs_{args.b}_{args.metric}.csv"
    R.to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Step 10 - Paired comparison of within-target correlations (Article Table 5).

    python 10_steiger.py                       # reads output/bench_scored.csv
    python 10_steiger.py --scored /path/to.csv --out /path/to/steiger.csv

Two correlations measured against the same observed values are dependent, so
comparing their separate confidence intervals understates the evidence. Steiger
(1980) gives the paired test for exactly this case, and it is what the Article
reports alongside the cluster bootstrap in bench_B_vs_ref.csv.

Everything here is computed on TARGET-CENTERED values: within each target, the
measured pKd and each method's prediction have that target's own mean subtracted
before correlating. That is the within-target correlation of Table 4 - the
ability to rank ligands against one protein - with the between-target component
removed rather than averaged over.

Each comparison runs on its own intersection: the rows where BOTH methods
returned a prediction. Boltz-2 and PRODIGY-LIG return nothing for 4 of the 500
pairs, so Savant's own within-target r differs slightly between rows of the
output. That is intended; a single pooled n would be comparing methods on
different pairs.

Degrees of freedom are (pairs - targets), not (pairs - 3): centering on 20
target means spends 20 degrees of freedom, not 1.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import OUT, log

## The model of record is Savant esm650 - the ESM2-650M / ChemBERTa-77M-MLM
## production model. The esm8 variant is trained on identical rows and differs
## only in the protein encoder, so it is not a reference method.
MODEL = "savant_esm650_pk"
MODEL_LABEL = "Savant esm650"

## Reference methods, in the order Table 5 lists them.
REFERENCES = [
    ("Boltz-2", "boltz_affinity_pic50"),
    ("PRODIGY-LIG", "prodigy_pkd"),
    ("PLAPT", "plapt_pk"),
]

TRUTH = "neg_log10_affinity_M"
GROUP = "uniprot_id"


def center(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Subtract each target's own mean from every named column."""
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(GROUP)[c].transform("mean")
    return out


def steiger(r_jk: float, r_jh: float, r_kh: float, n: int, df: int) -> tuple[float, float]:
    """Steiger's t for r_jk vs r_jh, two correlations sharing variable j.

    r_kh is the correlation between the two predictors - the agreement term
    that makes this test more sensitive than comparing independent intervals.
    Follows Steiger (1980) eq. 11, with the degrees of freedom passed in
    because target-centering does not spend the usual 3.
    """
    det = (1 - r_jk ** 2 - r_jh ** 2 - r_kh ** 2) + 2 * r_jk * r_jh * r_kh
    rbar = (r_jk + r_jh) / 2
    num = (r_jk - r_jh) * np.sqrt((n - 1) * (1 + r_kh))
    den = np.sqrt(2 * ((n - 1) / df) * det + (rbar ** 2) * ((1 - r_kh) ** 3))
    t = num / den
    return float(t), float(2 * stats.t.sf(abs(t), df))


def compare(scored: pd.DataFrame, ref_label: str, ref_col: str) -> dict | None:
    cols = [TRUTH, MODEL, ref_col]
    rows = scored.dropna(subset=cols)
    if rows.empty:
        log(f"  {ref_label}: no rows with both predictions, skipped")
        return None

    ## Centering must happen AFTER the intersection is taken: a target mean
    ## computed over rows the reference could not score is a different mean.
    c = center(rows, cols)
    r_sav = c[TRUTH].corr(c[MODEL])
    r_ref = c[TRUTH].corr(c[ref_col])
    r_agree = c[MODEL].corr(c[ref_col])

    n = len(rows)
    df = n - rows[GROUP].nunique()
    t, p = steiger(r_sav, r_ref, r_agree, n, df)
    return {
        "vs": ref_label, "n": n, "df": df,
        "within_r_savant": r_sav, "within_r_ref": r_ref,
        "delta": r_sav - r_ref, "model_agreement_r": r_agree,
        "steiger_t": t, "p_value": p,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", default=os.path.join(OUT, "bench_scored.csv"),
                    help="scored benchmark written by score_savant_benchmark.ipynb")
    ap.add_argument("--out", default=os.path.join(OUT, "bench_steiger.csv"))
    args = ap.parse_args()

    scored = pd.read_csv(args.scored)
    missing = [c for c in [TRUTH, GROUP, MODEL] if c not in scored.columns]
    if missing:
        log(f"FAIL: {args.scored} is missing {missing}")
        return 1

    log(f"{len(scored)} pairs, {scored[GROUP].nunique()} targets, model = {MODEL_LABEL}")
    rows = [r for label, col in REFERENCES
            if col in scored.columns
            for r in [compare(scored, label, col)] if r]

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    log(f"wrote {args.out}")
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

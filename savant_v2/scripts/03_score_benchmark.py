#!/usr/bin/env python
"""
Step 03 - Score the 1000-pair benchmark and compare against the other tools.

Reads the v2 features for the benchmark structures, applies the trained bundle,
and reports next to the Boltz-2 and PRODIGY-LIG numbers already in
ML_model/benchmark/eval/output/savant_benchmark_1000_scored.csv.

Every comparison is computed on the rows all named tools scored, because the
tools cover different subsets and comparing them on their own subsets is how v1
looked better than it was.

    python 03_score_benchmark.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from savant2.models import SavantBundle       # noqa: E402
from savant2.metrics import METRIC_ORDER, regression_metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
WORK = ROOT / "work"
OUT = ROOT / "output"

SCORED = REPO / "ML_model" / "benchmark" / "eval" / "output" / "savant_benchmark_1000_scored.csv"
LABEL = "neg_log10_affinity_M"
NON_FEATURE = {"id", "ligand_typed", "n_ligand_atoms", "n_protein_atoms"}

COMPARATORS = {
    "Boltz-2 affinity": "boltz_affinity_pic50",
    "PRODIGY-LIG": "prodigy_pkd",
    "PRODIGY-LIG (+elec)": "prodigy_pkd_elec",
    "Savant v1": "savant_pk",
}


def metrics(y: pd.Series, p: pd.Series) -> dict:
    m = y.notna() & p.notna()
    if m.sum() < 5:
        return {"n": int(m.sum())}
    a, b = y[m].astype(float), p[m].astype(float)
    err = (a - b).abs()
    return {
        "n": int(m.sum()),
        "pearson_r": float(a.corr(b)),
        "spearman_rho": float(a.corr(b, method="spearman")),
        "rmse": float(np.sqrt(((a - b) ** 2).mean())),
        "median_fold_error": float(10 ** err.median()),
        "pct_within_10x": float(100 * (err <= 1).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(WORK / "features_benchmark"))
    ap.add_argument("--model-dir", default=str(ROOT / "models"))
    ap.add_argument("--out", default=str(OUT / "benchmark_v2_scored.csv"))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    path = Path(args.features)
    shards = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not shards:
        raise SystemExit(f"no features in {path} - run 01_extract_features.py "
                         "with --manifest work/manifest_benchmark.csv")
    feats = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)

    model = SavantBundle.load(args.model_dir)
    X = feats[model.feature_names].to_numpy(np.float32)
    pred = model.predict(X)

    v2 = pd.DataFrame({
        "benchmark_id": feats.id,
        "savant2_p_binder": pred["p_binder"],
        "savant2_is_binder": pred["is_binder"],
        "savant2_pk": pred["pk"],
        "savant2_pk_lower": pred["pk_lower"],
        "savant2_pk_upper": pred["pk_upper"],
        "savant2_interval_width": pred["pk_upper"] - pred["pk_lower"],
        "savant2_kd_nM": 10 ** (9 - pred["pk"]),
    })

    base = pd.read_csv(SCORED)
    merged = base.merge(v2, on="benchmark_id", how="left")
    merged.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(merged)} rows)")

    # --- headline comparison, on rows every tool scored ---------------------
    cols = [c for c in COMPARATORS.values() if c in merged.columns] + ["savant2_pk"]
    common = merged[merged[cols].notna().all(axis=1) & merged[LABEL].notna()]
    print(f"\nHead-to-head on the {len(common)} pairs every tool scored:")
    rows = []
    for name, col in {**COMPARATORS, "Savant v2": "savant2_pk"}.items():
        if col in common:
            rows.append({"tool": name, **metrics(common[LABEL], common[col])})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- coverage: what each stage actually returned ------------------------
    print("\nCoverage of the full 1000:")
    for name, col in {**COMPARATORS, "Savant v2": "savant2_pk"}.items():
        if col in merged:
            print(f"  {name:22} {int(merged[col].notna().sum()):4d}")
    if "savant2_is_binder" in merged:
        gate = merged.savant2_is_binder.fillna(False)
        print(f"  {'Savant v2 gate kept':22} {int(gate.sum()):4d}"
              f"   (v1 kept {int(merged.savant_pk.notna().sum())})")

    # --- conformal coverage, measured not assumed --------------------------
    ok = merged.savant2_pk.notna() & merged[LABEL].notna()
    if ok.any():
        inside = ((merged.loc[ok, LABEL] >= merged.loc[ok, "savant2_pk_lower"]) &
                  (merged.loc[ok, LABEL] <= merged.loc[ok, "savant2_pk_upper"]))
        print(f"\nConformal intervals: nominal {100*(1-model.conformal.alpha):.0f}%, "
              f"empirical {100*inside.mean():.1f}%, "
              f"median width {merged.loc[ok,'savant2_interval_width'].median():.2f} log units")
        print("  The benchmark is protein- and ligand-disjoint from training, which breaks")
        print("  the exchangeability conformal coverage assumes. Report this gap, do not")
        print("  hide it: it is the honest measure of what the guarantee is worth on")
        print("  genuinely novel targets.")

    if "novelty_tier" in merged.columns:
        print("\nBy novelty tier (Pearson r):")
        tiers = []
        for tier, grp in common.groupby("novelty_tier"):
            row = {"tier": tier, "n": len(grp)}
            for name, col in {**COMPARATORS, "Savant v2": "savant2_pk"}.items():
                if col in grp:
                    row[name] = metrics(grp[LABEL], grp[col]).get("pearson_r", np.nan)
            tiers.append(row)
        print(pd.DataFrame(tiers).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()

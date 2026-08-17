#!/usr/bin/env python
"""
Step 07 - Reshape the scored benchmark for Tableau.

The scored CSV is wide: one column per tool. Tableau wants a tool DIMENSION, so
every "compare the tools" view becomes a colour/facet instead of hand-built
dual axes. This writes two files:

  *_tableau_long.csv     one row per (pair, tool): predicted, error, abs_error,
                         the same prediction as a dissociation constant in nM
                         and M, plus the label and every grouping field worth
                         faceting on. Rows a tool did not score are kept with a
                         null prediction and scored=False, so coverage is visible.

  *_tableau_metrics.csv  precomputed n / Pearson r / Spearman rho / RMSE / MAE
                         per tool per grouping (overall, novelty tier, protein
                         class, ligand class, affinity range, pose-quality
                         quartile). Correlations are awkward to compute inside
                         Tableau and easy to get wrong across facets; computing
                         them here means a bar chart is just SUM/AVG of a field.

    python 07_tableau_export.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from common import OUT, SCORED_CSV, ensure_dirs, log

LABEL = "neg_log10_affinity_M"

# pKd = 9 - log10(Kd[nM]), so Kd[nM] = 10^(9 - pKd) and Kd[M] = 10^(-pKd).
# Every tool is already on that log scale by the time it reaches this script
# (06_merge_scores.py converts PRODIGY's dG at 298 K and Boltz's log10 IC50),
# so one exponentiation puts them all in concentration units.
#
# CAVEAT worth keeping in front of anyone reading the Kd columns: only Savant
# predicts a Kd. Boltz-2's affinity head is trained on IC50 and PRODIGY-LIG
# predicts a binding free energy. Converting them to nM changes the units, not
# what was measured - they stay comparable in rank and order of magnitude, not
# as like-for-like dissociation constants.
PKD_OFFSET_NM = 9

# tool -> (prediction column, what the number is)
TOOLS = {
    "Boltz-2 affinity": ("boltz_affinity_pic50", "6 - log10(IC50 uM), IC50-trained",
                         "IC50 reported as Kd-equivalent"),
    "Savant pK": ("savant_pk", "pKd, trained on this target", "true Kd prediction"),
    "PRODIGY-LIG": ("prodigy_pkd", "-dG/1.364, contacts only", "from dG at 298 K"),
    "PRODIGY-LIG (+elec)": ("prodigy_pkd_elec", "-dG/1.364, with HADDOCK elec",
                            "from dG at 298 K"),
}

# Carried onto every row so any of them can be a Tableau facet, colour or filter.
DIMENSIONS = [
    "benchmark_id", "uniprot_id", "gene", "protein_name", "organism",
    "novelty_tier", "protein_class", "chembl_l1", "chembl_l2", "family_l1",
    "cf_superclass", "cf_class", "drug_class_atc", "atc_l1", "max_phase",
    "murcko_scaffold", "inchikey",
]
MEASURES = [
    "kd_nM", "molecular_weight", "logP", "tpsa", "num_heavy_atoms",
    "num_rotatable_bonds", "num_aromatic_rings", "formal_charge",
    "boltz_confidence_score", "boltz_iptm", "boltz_ligand_iptm",
    "boltz_complex_plddt", "boltz_affinity_probability_binary",
    "savant_p_binder", "savant_classification",
    "haddock_score", "haddock_vdw", "haddock_elec", "haddock_bsa",
    "haddock_num_interacting_residues", "was_desalted", "affinity_requested",
]


FIELDS = ["n", "pearson_r", "spearman_rho", "rmse", "mae", "median_abs_error",
          "median_fold_error", "pct_within_10x", "pct_within_100x"]


def metrics(t: pd.Series, p: pd.Series) -> dict:
    mask = t.notna() & p.notna()
    n = int(mask.sum())
    if n < 5:
        return {**{f: np.nan for f in FIELDS}, "n": n}
    t, p = t[mask].astype(float), p[mask].astype(float)
    abs_error = (t - p).abs()
    return {
        "n": n,
        "pearson_r": t.corr(p),
        "spearman_rho": t.corr(p, method="spearman"),
        "rmse": float(np.sqrt(((t - p) ** 2).mean())),
        "mae": float(abs_error.mean()),
        "median_abs_error": float(abs_error.median()),
        # The log-unit errors above, restated as a fold-difference in Kd.
        "median_fold_error": float(10.0 ** abs_error.median()),
        "pct_within_10x": float(100.0 * (abs_error <= 1.0).mean()),
        "pct_within_100x": float(100.0 * (abs_error <= 2.0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=SCORED_CSV)
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()

    ensure_dirs(args.out_dir)
    df = pd.read_csv(args.csv)
    log(f"{len(df)} scored rows from {args.csv}")

    # Binned versions of two continuous fields, so they can be used as facets.
    df["pkd_bin"] = pd.cut(df[LABEL], [0, 5, 6, 7, 8, 9, 20],
                           labels=["<5", "5-6", "6-7", "7-8", "8-9", ">9"])
    # Labels are ordered by their own text so Tableau sorts them correctly
    # without needing a manual sort on the dimension.
    posed = df.boltz_ligand_iptm.notna()
    pose_quality = pd.Series(np.nan, index=df.index, dtype=object)
    pose_quality.loc[posed] = pd.qcut(
        df.loc[posed, "boltz_ligand_iptm"], 4,
        labels=["Q1 worst pose", "Q2", "Q3", "Q4 best pose"],
    ).astype(str)
    df["pose_quality"] = pose_quality

    keep_dims = [c for c in DIMENSIONS if c in df.columns]
    keep_meas = [c for c in MEASURES if c in df.columns] + ["pkd_bin", "pose_quality"]

    frames = []
    for tool, (column, note, kd_note) in TOOLS.items():
        if column not in df.columns:
            log(f"WARNING: {column} missing, skipping {tool}")
            continue
        block = df[keep_dims + keep_meas].copy()
        block["tool"] = tool
        block["prediction_note"] = note
        block["kd_note"] = kd_note
        block["measured_pkd"] = df[LABEL]
        block["predicted_pkd"] = df[column]
        block["scored"] = df[column].notna()
        # Signed error keeps over/under-prediction visible; Tableau can ABS() it
        # but not recover the sign.
        block["error"] = df[column] - df[LABEL]
        block["abs_error"] = block["error"].abs()

        # Concentration units. Plot these on a LOG axis in Tableau: they span
        # ~12 orders of magnitude, so a linear axis shows one weak binder and
        # nothing else.
        block["measured_kd_nM"] = 10.0 ** (PKD_OFFSET_NM - df[LABEL])
        block["predicted_kd_nM"] = 10.0 ** (PKD_OFFSET_NM - df[column])
        block["measured_kd_M"] = 10.0 ** (-df[LABEL])
        block["predicted_kd_M"] = 10.0 ** (-df[column])
        # An error of 1 log unit is a 10-fold miss; this is that error read as a
        # multiplier, which is how medicinal chemists judge a prediction.
        block["fold_error"] = 10.0 ** block["abs_error"]
        block["within_10x"] = block["abs_error"] <= 1.0
        block["within_100x"] = block["abs_error"] <= 2.0
        frames.append(block)

    long = pd.concat(frames, ignore_index=True)
    long_path = os.path.join(args.out_dir, "savant_benchmark_1000_tableau_long.csv")
    long.to_csv(long_path, index=False)
    log(f"wrote {long_path} ({len(long)} rows = {len(df)} pairs x {len(frames)} tools)")

    # uniprot_id/gene give per-target rows. Most targets carry 1-3 pairs, so
    # metrics() returns NaN for them (n < 5) and only the 58 targets with enough
    # ligands report a value - which is the population that can answer "does the
    # tool rank analogs against a single target correctly", the lead-optimisation
    # question. Filter on n in Tableau.
    groupings = ["novelty_tier", "protein_class", "chembl_l1", "cf_superclass",
                 "pkd_bin", "pose_quality", "uniprot_id", "gene"]
    rows = []
    for tool, (column, _, _kd) in TOOLS.items():
        if column not in df.columns:
            continue
        rows.append({"grouping": "overall", "group": "all pairs", "tool": tool,
                     **metrics(df[LABEL], df[column])})
        for grouping in groupings:
            if grouping not in df.columns:
                continue
            for value, grp in df.groupby(grouping, observed=True):
                rows.append({"grouping": grouping, "group": str(value), "tool": tool,
                             **metrics(grp[LABEL], grp[column])})

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(args.out_dir, "savant_benchmark_1000_tableau_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    log(f"wrote {metrics_path} ({len(metrics_df)} tool x group rows)")

    overall = metrics_df[metrics_df.grouping == "overall"]
    log("overall (each tool on every pair it scored):")
    for r in overall.itertuples(index=False):
        log(f"  {r.tool:22} n={r.n:4d}  r={r.pearson_r:+.3f}  rho={r.spearman_rho:+.3f}  "
            f"rmse={r.rmse:.2f}  median fold-error={r.median_fold_error:6.1f}x  "
            f"within 10x={r.pct_within_10x:4.1f}%")


if __name__ == "__main__":
    main()

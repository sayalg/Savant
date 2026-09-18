#!/usr/bin/env python
"""
Step 06 - Merge every tool's prediction back onto the benchmark rows.

Produces output/<benchmark stem>_scored.csv - by default
output/savant_benchmark_scored.csv: the original 500-row benchmark with its
labels and annotations untouched, plus one block of columns per tool. Nothing is
dropped - pairs a tool failed on keep NaN in that tool's columns and are counted
in the summary, so coverage stays visible instead of silently biasing whatever is
computed downstream.

Predictions are put on the label's scale wherever the tool allows it:

  boltz_affinity_pic50  6 - log10(IC50[uM])   comparable to pKd, not identical:
                        Boltz-2's affinity head is trained on IC50, the label is Kd
  savant_pk             already pKd (the regressor's training target)
  prodigy_pkd           -dG / 1.364, the benchmark's own deltaG <-> pKd convention

    python 06_merge_scores.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from common import (BENCHMARK_CSV, BOLTZ_SCORES, MANIFEST, OUT, PRODIGY_RESULTS, RESULTS,
                    SAVANT_RESULTS, SCORED_CSV, SCORED_SUMMARY, ensure_dirs, log)

## PLAPT needs no structure, so its results live outside the Boltz-dependent
## tree and its coverage is not capped by whether a pose was produced.
PLAPT_RESULTS = os.path.join(RESULTS, "plapt")

LABEL = "neg_log10_affinity_M"

BOLTZ_COLUMNS = [
    "boltz_status", "boltz_confidence_score", "boltz_ptm", "boltz_iptm",
    "boltz_ligand_iptm", "boltz_complex_plddt", "boltz_affinity_pred_value",
    "boltz_affinity_probability_binary", "boltz_affinity_pic50",
    "chains_remapped", "structure_path", "boltz_error",
]

# Savant's 7 structure-derived features, kept alongside its prediction: they are
# the only part of its feature vector that depends on the Boltz pose, so they are
# what to look at when a prediction is off.
SAVANT_FEATURES = ["score", "total", "vdw", "elec", "desolv", "bsa", "num_interacting_residues"]

PREDICTION_COLUMNS = {
    "boltz_affinity_pic50": "Boltz-2 affinity",
    "savant_pk": "Savant pK",
    "prodigy_pkd": "PRODIGY-LIG",
    "prodigy_pkd_elec": "PRODIGY-LIG (+elec)",
    "plapt_pk": "PLAPT",
}


def load_savant(results_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        bid = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path) as fh:
                data = json.load(fh)
        except json.JSONDecodeError:
            log(f"WARNING: unreadable Savant JSON, ignoring: {path}")
            continue
        probabilities = data.get("classification_probabilities") or {}
        features = data.get("features") or {}
        row = {
            "benchmark_id": bid,
            "savant_classification": data.get("classification"),
            "savant_p_binder": probabilities.get("binder"),
            "savant_pk": data.get("pK"),
            "savant_ligand_smiles_source": data.get("ligand_smiles_source"),
        }
        row.update({f"haddock_{k}": features.get(k) for k in SAVANT_FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)


def load_plapt(results_dir: str) -> pd.DataFrame:
    """One row per scored pair from 09_run_plapt.py.

    `plapt_pk` is already on the pKd scale - PLAPT's own key for it is
    `neg_log10_affinity_M`, which is the benchmark's ground-truth column name, so
    step 09 renames it on write rather than risking a silent collision here.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except json.JSONDecodeError:
            log(f"WARNING: unreadable PLAPT JSON, ignoring: {path}")
            continue
        rows.append({
            "benchmark_id": data.get("benchmark_id"),
            "plapt_pk": data.get("plapt_pk"),
            "plapt_affinity_uM": data.get("plapt_affinity_uM"),
        })
    return pd.DataFrame(rows)


def load_prodigy(results_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as fh:
                rows.append(json.load(fh))
        except json.JSONDecodeError:
            log(f"WARNING: unreadable PRODIGY JSON, ignoring: {path}")
    frame = pd.DataFrame(rows)
    return frame.drop(columns=["haddock_elec"], errors="ignore")  # already from Savant


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    mask = y_true.notna() & y_pred.notna()
    n = int(mask.sum())
    if n < 3:
        return {"n": n, "pearson_r": np.nan, "spearman_rho": np.nan,
                "rmse": np.nan, "mae": np.nan}
    t, p = y_true[mask].astype(float), y_pred[mask].astype(float)
    return {
        "n": n,
        "pearson_r": t.corr(p),
        "spearman_rho": t.corr(p, method="spearman"),
        "rmse": float(np.sqrt(((t - p) ** 2).mean())),
        "mae": float((t - p).abs().mean()),
    }


def table(rows: list[dict], columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) and not np.isnan(v)
                         else ("n/a" if isinstance(v, float) else str(v)))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule] + body)


def write_summary(df: pd.DataFrame, path: str) -> None:
    total = len(df)
    lines = [
        "# Benchmark scoring run - Boltz-2, Savant, PRODIGY-LIG, PLAPT",
        "",
        f"{total} pairs from `{os.path.basename(BENCHMARK_CSV)}`. "
        f"Label: `{LABEL}` (pKd), median {df[LABEL].median():.2f}.",
        "",
        "## Coverage",
        "",
    ]
    status = df["boltz_status"] if "boltz_status" in df else pd.Series(dtype=object)
    coverage = [
        {"stage": "Boltz-2 structure", "n": int((status == "ok").sum())},
        {"stage": "Boltz-2 affinity",
         "n": int(df.get("boltz_affinity_pic50", pd.Series(dtype=float)).notna().sum())},
        {"stage": "Savant pK",
         "n": int(df.get("savant_pk", pd.Series(dtype=float)).notna().sum())},
        {"stage": "Savant classified",
         "n": int(df.get("savant_classification", pd.Series(dtype=object)).notna().sum())},
        {"stage": "PRODIGY-LIG",
         "n": int(df.get("prodigy_pkd", pd.Series(dtype=float)).notna().sum())},
        {"stage": "PRODIGY-LIG (+elec)",
         "n": int(df.get("prodigy_pkd_elec", pd.Series(dtype=float)).notna().sum())},
        ## structure-free, so not capped by Boltz coverage
        {"stage": "PLAPT",
         "n": int(df.get("plapt_pk", pd.Series(dtype=float)).notna().sum())},
    ]
    for row in coverage:
        row["% of pairs"] = f"{100.0 * row['n'] / total:.1f}" if total else "0.0"
    lines += [table(coverage, ["stage", "n", "% of pairs"]), ""]

    if "savant_classification" in df:
        classified = df.savant_classification.dropna()
        if len(classified):
            n_binder = int((classified == "binder").sum())
            lines += [
                f"Every benchmark pair has a measured Kd, so all {total} are true binders: "
                f"Savant's classifier called {n_binder}/{len(classified)} "
                f"({100.0 * n_binder / len(classified):.1f}%) binder. Only those get a pK.",
                "",
            ]

    lines += ["## Agreement with measured pKd", "",
              "Correlation is the meaningful column; RMSE/MAE only mean something for "
              "predictions already on the pKd scale.", ""]
    rows = []
    for column, name in PREDICTION_COLUMNS.items():
        if column in df:
            rows.append({"prediction": name, **metrics(df[LABEL], df[column])})
    lines += [table(rows, ["prediction", "n", "pearson_r", "spearman_rho", "rmse", "mae"]), ""]

    if "novelty_tier" in df:
        lines += ["## By novelty tier", ""]
        rows = []
        for tier, group in df.groupby("novelty_tier"):
            for column, name in PREDICTION_COLUMNS.items():
                if column in group:
                    rows.append({"tier": tier, "prediction": name,
                                 **metrics(group[LABEL], group[column])})
        lines += [table(rows, ["tier", "prediction", "n", "pearson_r", "spearman_rho", "rmse"]), ""]

    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=BENCHMARK_CSV)
    ap.add_argument("--out", default=SCORED_CSV)
    ap.add_argument("--summary", default=SCORED_SUMMARY)
    args = ap.parse_args()

    ensure_dirs(OUT)
    df = pd.read_csv(args.csv)
    log(f"{len(df)} benchmark rows")

    if os.path.exists(MANIFEST):
        manifest = pd.read_csv(MANIFEST)[
            ["benchmark_id", "docked_smiles", "was_desalted", "affinity_requested"]
        ]
        df = df.merge(manifest, on="benchmark_id", how="left")

    if os.path.exists(BOLTZ_SCORES):
        boltz = pd.read_csv(BOLTZ_SCORES).rename(
            columns={"status": "boltz_status", "error": "boltz_error"})
        keep = ["benchmark_id"] + [c for c in BOLTZ_COLUMNS if c in boltz.columns]
        df = df.merge(boltz[keep], on="benchmark_id", how="left")
        log(f"merged Boltz scores for {int((boltz.boltz_status == 'ok').sum())} usable structures")
    else:
        log(f"WARNING: {BOLTZ_SCORES} not found - run 03_collect_boltz.py")

    savant = load_savant(SAVANT_RESULTS)
    if len(savant):
        df = df.merge(savant, on="benchmark_id", how="left")
        log(f"merged Savant results for {len(savant)} pairs")
    else:
        log(f"WARNING: no Savant JSONs in {SAVANT_RESULTS}")

    prodigy = load_prodigy(PRODIGY_RESULTS)
    if len(prodigy):
        df = df.merge(prodigy, on="benchmark_id", how="left")
        log(f"merged PRODIGY results for {len(prodigy)} pairs")
    else:
        log(f"WARNING: no PRODIGY JSONs in {PRODIGY_RESULTS}")

    plapt = load_plapt(PLAPT_RESULTS)
    if len(plapt):
        df = df.merge(plapt, on="benchmark_id", how="left")
        log(f"merged PLAPT results for {len(plapt)} pairs")
    else:
        log(f"no PLAPT JSONs in {PLAPT_RESULTS} (run 09_run_plapt.py to add it)")

    df.to_csv(args.out, index=False)
    log(f"wrote {args.out} ({len(df)} rows x {len(df.columns)} columns)")

    write_summary(df, args.summary)
    log(f"wrote {args.summary}")

    failure_logs = sorted(glob.glob(os.path.join(RESULTS, "*_failures_shard*.csv")))
    if failure_logs:
        failures = pd.concat([pd.read_csv(p) for p in failure_logs], ignore_index=True)
        path = os.path.join(OUT, "failures.csv")
        failures.to_csv(path, index=False)
        log(f"{len(failures)} tool failures collated -> {path}")


if __name__ == "__main__":
    main()

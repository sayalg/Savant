#!/usr/bin/env python
"""Every method, rescored on the subset that clears Savant's own 0.50 gate.

    python scripts/04_compare_decontaminated.py

03_evaluate.py puts Savant into PLABench's published table, but that table
scores their 9 methods on all 4,862 pairs while Savant's honest number is on the
2,029 that clear our stricter gate. Comparing those 2 directly would be wrong in
both directions, so this script recomputes every method on the same rows, from
the per-pair predictions PLABench ships.

The protocol is theirs throughout: per-target Pearson, Spearman and RMSE,
averaged across targets weighted by the number of pairs. The script first
reproduces their published full-set numbers from the same prediction files, so
that any disagreement shows up as a reproduction failure rather than as a
difference in Savant's favour.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRED = ROOT / "data" / "plabench_predictions"
THEIRS = ROOT / "data" / "plabench_results" / "chembl35"
PAIRS = ROOT / "data" / "chembl35" / "chembl35_filtered_input.csv"
SCORES = ROOT / "output" / "savant_chembl35_filtered.csv"
AUDIT = ROOT / "output" / "contamination_audit.csv"
OUT = ROOT / "output"

## The run directory PLABench's own filtered summary is built from, per model.
RUNS = {
    "bapred": "af3_chembl35_full", "boltz2": "boltz2_chembl35_full",
    "deepdta": "sequence_chembl35_full", "flowdock": "af3_chembl35_full",
    "flowr_root": "flowr_root_af3_chembl35_full", "haiping": "haiping_af3_chembl35_full",
    "llf": "sequence_chembl35_full", "mfe": "mfe_af3_chembl35_full",
    "mixingdta": "sequence_chembl35_full",
}
MIN_PAIRS = 2


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_preds(model):
    f = PRED / model / RUNS[model] / "latest" / "predictions.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, header=None, names=["pred", "compound_id"])
    return d.dropna().drop_duplicates("compound_id")


def weighted_per_target(df, truth, pred, group="uniprot"):
    rows = []
    for uid, g in df.groupby(group):
        if len(g) < MIN_PAIRS or g[pred].std() == 0 or g[truth].std() == 0:
            continue
        rows.append(dict(N=len(g),
                         Pearson=np.corrcoef(g[truth], g[pred])[0, 1],
                         Spearman=spearmanr(g[truth], g[pred]).correlation,
                         RMSE=float(np.sqrt(((g[truth] - g[pred]) ** 2).mean()))))
    pt = pd.DataFrame(rows)
    if pt.empty:
        return dict(N_total=len(df), N_targets=0, Pearson=np.nan,
                    Spearman=np.nan, RMSE=np.nan)
    w = lambda c: float(np.average(pt[c].dropna(), weights=pt.loc[pt[c].notna(), "N"]))
    return dict(N_total=int(pt.N.sum()), N_targets=len(pt),
                Pearson=w("Pearson"), Spearman=w("Spearman"), RMSE=w("RMSE"))


def main():
    base = pd.read_csv(PAIRS)
    base["uniprot"] = base.compound_id.str.split("_").str[0]
    sav = pd.read_csv(SCORES)[["compound_id", "savant_pk"]]
    base = base.merge(sav, on="compound_id", how="left")

    aud = pd.read_csv(AUDIT)[["compound_iso_smiles", "fails_savant_gate"]]
    base = base.merge(aud, on="compound_iso_smiles", how="left")
    base["fails_savant_gate"] = base.fails_savant_gate.fillna(False)
    clean = base[~base.fails_savant_gate]
    log(f"full {len(base):,} pairs / {base.uniprot.nunique()} targets | "
        f"Savant-gate subset {len(clean):,} / {clean.uniprot.nunique()}")

    pub = pd.read_csv(THEIRS / "weighted_summary_chembl35_filtered.csv").set_index("Model")

    rows, repro_fail = [], []
    for model in sorted(RUNS):
        p = load_preds(model)
        if p is None:
            log(f"  {model}: no predictions file, skipped")
            continue
        m_all = base.merge(p, on="compound_id", how="inner")
        m_cln = clean.merge(p, on="compound_id", how="inner")
        a = weighted_per_target(m_all, "affinity", "pred")
        c = weighted_per_target(m_cln, "affinity", "pred")

        ## Reproduction check against their published filtered number.
        want = float(pub.loc[model, "Pearson"]) if model in pub.index else np.nan
        ok = np.isfinite(want) and abs(a["Pearson"] - want) < 0.01
        if np.isfinite(want) and not ok:
            repro_fail.append((model, a["Pearson"], want))
        rows.append(dict(Model=model, published=want, **{f"all_{k}": v for k, v in a.items()},
                         **{f"clean_{k}": v for k, v in c.items()}))
        log(f"  {model:11s} all {a['Pearson']:.4f} (published {want:.4f}) "
            f"{'ok' if ok else 'MISMATCH'} | clean {c['Pearson']:.4f} "
            f"on {c['N_total']:,} pairs")

    a = weighted_per_target(base.dropna(subset=["savant_pk"]), "affinity", "savant_pk")
    c = weighted_per_target(clean.dropna(subset=["savant_pk"]), "affinity", "savant_pk")
    rows.append(dict(Model="Savant", published=np.nan,
                     **{f"all_{k}": v for k, v in a.items()},
                     **{f"clean_{k}": v for k, v in c.items()}))
    log(f"  {'Savant':11s} all {a['Pearson']:.4f} (not in their table)      "
        f" | clean {c['Pearson']:.4f} on {c['N_total']:,} pairs")

    if repro_fail:
        log("\nWARNING: could not reproduce these published numbers from the shipped "
            "predictions:")
        for m, got, want in repro_fail:
            log(f"    {m}: recomputed {got:.4f} vs published {want:.4f}")
        log("Treat the corresponding rows as unverified.")

    res = pd.DataFrame(rows).sort_values("clean_Pearson", ascending=False)
    res.to_csv(OUT / "comparison_decontaminated.csv", index=False)

    pd.set_option("display.width", 220)
    show = res[["Model", "all_N_total", "all_Pearson", "all_Spearman",
                "clean_N_total", "clean_N_targets", "clean_Pearson",
                "clean_Spearman", "clean_RMSE"]]
    print("\n===== PLABench protocol: all pairs vs the Savant-gate 0.50 subset =====\n")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log(f"\nwrote {OUT/'comparison_decontaminated.csv'}")


if __name__ == "__main__":
    main()

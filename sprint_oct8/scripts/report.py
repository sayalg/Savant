#!/usr/bin/env python3
"""Collate every run into `reports/final.md` (PLAN.md Section 8, Wave 4 / Section 14).

Tables are kept separate on purpose and never merged:

  1. HEADLINE      the protein-cluster split, with ligand_only and target_mean on
                   every row (Section 0.3.2). This is the number.
  2. ABLATIONS     the Phase 9 matrix: PLM mode, geometry variant, vdW form, terms.
  3. SPLITS        the same model across protein_cluster / double / scaffold, so the
                   gap between them is visible rather than implied.
  4. DIAGNOSTICS   distogram calibration, pocket recovery, per-term energies,
                   parameter drift (Section 9's physics diagnostics).
  5. COST          inference seconds per complex, for the value case if the accuracy
                   case fails (Section 9).
  6. SEALED        the benchmark, once per model version.

The report states the configuration count from the compute ledger, because Section
7.4 asks for a multiplicity-aware reading and a leaderboard without that number
invites the opposite.

An empty table is explained in place. "No runs yet" on its own sent the previous
project hunting in the wrong place for an afternoon; the usual cause is a Slurm
dependency cancellation, and the report now says so and names the tracer.

    python scripts/report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

## Imported FIRST, before torch / matplotlib / transformers: importing
## pfpl.paths redirects every HOME-bound cache and TMPDIR under the project
## root, and some of those libraries read their cache variable at import time.
from pfpl import paths as _paths  # noqa: E402,F401


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pfpl import ledger, paths  # noqa: E402

EMPTY_RUNS = """_No runs produced output._

This is usually NOT a reporting problem. `slurm/submit_wave2.sh` wires the training
arms with `--dependency=afterok:... --kill-on-invalid-dep=yes`, so a failure upstream
(the embedding array, mmseqs, the feature merge) CANCELS everything downstream. The
report job is wired `afterany`, so it still runs and finds nothing. `squeue` looks
identical either way.

Trace it:

```bash
python scripts/status.py
```
"""


def md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return ("```\n" + df.to_string(index=False)
                + "\n```\n_(install `tabulate` for real markdown tables)_")


def fmt(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in ("pearson", "rho", "rho_lo", "rho_hi", "slope", "rho_ligand_only",
              "rho_target_mean", "pearson_ligand_only", "delta_vs_ligand_only"):
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:+.3f}" if pd.notna(v) and np.isfinite(v) else "-")
    for c in ("rmse", "rmse_const", "ef10", "std_ratio", "coverage80", "ece"):
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:.3f}" if pd.notna(v) and np.isfinite(v) else "-")
    return d[[c for c in cols if c in d.columns]]


def collect_baselines() -> dict:
    """Phase 6 baselines, keyed by split. They are NOT model runs and must not sit
    in the ablation tables - a `baseline` row with NaN everywhere would otherwise
    appear as a level of every axis."""
    out = {}
    for d in sorted(paths.RUNS.glob("baselines_*")):
        cj = d / "reports.json"
        if cj.exists():
            out[d.name.replace("baselines_", "")] = json.loads(cj.read_text())
    return out


def collect() -> pd.DataFrame:
    rows = []
    for d in sorted(paths.RUNS.glob("*")):
        rj, cj = d / "run.json", d / "reports.json"
        if not cj.exists() or d.name.startswith("baselines_"):
            continue
        reps = json.loads(cj.read_text())
        info = json.loads(rj.read_text()) if rj.exists() else {}
        m = reps.get("model", {})
        hist = []
        if (d / "history.json").exists():
            hist = json.loads((d / "history.json").read_text())
        last = hist[-1] if hist else {}
        rows.append({
            "run": d.name, "dir": str(d),
            "variant": info.get("variant", "baseline"),
            "plm_mode": info.get("plm_mode"),
            "split": info.get("split") or reps.get("_extra", {}).get("split"),
            "stage": info.get("stage"), "seed": info.get("seed"),
            "vdw_form": info.get("vdw_form"), "residual": info.get("residual"),
            "terms": ",".join(info.get("terms", [])),
            "preregistered": info.get("preregistered"),
            "pearson": m.get("pearson"),
            "rho": m.get("per_target_spearman_median"),
            "rho_lo": (m.get("per_target_spearman_ci") or [None, None])[0],
            "rho_hi": (m.get("per_target_spearman_ci") or [None, None])[1],
            "rmse": m.get("rmse"), "rmse_const": m.get("rmse_const"),
            "n_targets": m.get("n_targets"), "ef10": m.get("ef_at_10_median"),
            "slope": m.get("offset_slope"), "std_ratio": m.get("pred_std_ratio"),
            "coverage80": m.get("coverage_80"),
            "rho_ligand_only": reps.get("ligand_only", {}).get(
                "per_target_spearman_median"),
            "pearson_ligand_only": reps.get("ligand_only", {}).get("pearson"),
            "rho_target_mean": reps.get("target_mean", {}).get("pearson"),
            "ece": last.get("val_disto/ece"),
            "pocket_auroc": last.get("val_pocket/auroc"),
            "contact_recall": last.get("val_disto/contact_recall"),
            "decomposition": info.get("decomposition", {}),
            "elapsed_s": info.get("elapsed_s"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.REPORTS / "final.md"))
    ap.add_argument("--headline-split", default="protein_cluster")
    a = ap.parse_args()
    paths.ensure()

    runs = collect()
    base = collect_baselines()
    ## Section 0.3.2 requires ligand_only and target_mean on EVERY row. They live in
    ## the Phase 6 run directory, not in each model's reports.json, so join them on
    ## the split rather than leaving the columns blank.
    if not runs.empty and base:
        for col, src, field in (
                ("rho_ligand_only", "ligand_only", "per_target_spearman_median"),
                ("pearson_ligand_only", "ligand_only", "pearson"),
                ("rho_target_mean", "target_mean", "pearson")):
            filled = runs["split"].map(
                lambda s: (base.get(str(s), {}).get(src) or {}).get(field))
            runs[col] = runs[col].where(runs[col].notna(), filled)
    led = ledger.summary()
    L = ["# pfpl - results", "",
         "Pose-free, physics-informed protein-ligand affinity. Target is pKd, trained",
         "on Kd and Ki only; IC50/EC50 are excluded at every loader (PLAN.md Section",
         "0.4 rule 1).", "",
         f"**{led.get('n_runs', 0)} configurations trained**, "
         f"{led.get('gpu_hours_total', 0)} GPU-hours "
         f"({led.get('n_exploratory', 0)} outside the pre-registered matrix).",
         "", "> " + led.get("multiplicity_note", ""), ""]

    if runs.empty:
        L += ["## 1. Headline", "", EMPTY_RUNS]
        Path(a.out).write_text("\n".join(L))
        print("\n".join(L))
        return

    ## ---- 1. headline --------------------------------------------------------
    h = runs[(runs.split == a.headline_split)
             & (runs.stage.isin(["affinity", "finetune", None]))]
    h = h.sort_values("pearson", ascending=False)
    L += [f"## 1. Headline - split `{a.headline_split}`", "",
          "Primary metric is Pearson (Section 9's comparison rule is written in it);",
          "median per-target Spearman sits next to it because a pooled metric over",
          "many proteins is saturated by between-protein variance. `rho_ligand_only`",
          "is the memorisation floor - a model that does not beat it has learned",
          "nothing about proteins.", ""]
    if h.empty:
        L += [f"_No runs on `{a.headline_split}` yet._", ""]
    else:
        d = h.copy()
        d["delta_vs_ligand_only"] = d.pearson - d.pearson_ligand_only
        L += [md(fmt(d, ["run", "variant", "plm_mode", "pearson", "rho", "rho_lo",
                         "rho_hi", "rho_ligand_only", "delta_vs_ligand_only", "rmse",
                         "rmse_const", "n_targets", "slope", "ef10", "coverage80",
                         "preregistered"])), ""]
        ## "Best physics" means the best VARIANT A OR B run. Taking h.iloc[0] would
        ## compare the control to itself whenever the control happens to win, which
        ## is exactly when the comparison matters most.
        phys = h[h.variant.isin(["A", "B"])]
        ctrl = h[h.variant == "C"]
        if ctrl.empty or phys.empty:
            missing = "matched no-physics control (`model=variant_c`)" if ctrl.empty \
                else "physics model (`model=variant_a` / `variant_b`)"
            L += [f"> **No {missing} on this split.** Section 9's Tier 1 rule compares",
                  "> the two directly, so there is no physics claim to make here yet -",
                  "> only a number.", ""]
        else:
            dd = phys.iloc[0].pearson - ctrl.iloc[0].pearson
            L += [f"> Best physics (`{phys.iloc[0].variant}`) vs matched control "
                  f"(`C`): **dPearson {dd:+.4f}** "
                  f"(Tier 1 needs >= {0.02} with a CI excluding zero).",
                  "> Run `python scripts/phase9_ablation.py --gate` for the paired "
                  "bootstrap, which is the number that decides.", ""]
        ## Two different readings, and conflating them sends you to the wrong fix.
        lo = phys.iloc[0].pearson_ligand_only if not phys.empty else None
        if not phys.empty and pd.notna(lo):
            delta = phys.iloc[0].pearson - lo
            if delta < 0:
                L += [f"> **The best model ({phys.iloc[0].pearson:+.3f}) is WORSE than a "
                      f"protein-blind ligand-only control ({lo:+.3f}).**",
                      "> That is not a split problem. It means the protein features are",
                      "> not merely uninformative but actively harmful - the model is",
                      "> splitting on protein structure that does not transfer to held-out",
                      "> clusters. Section 6.7's PLM-mode ablation (`plm=none` first) is",
                      "> the intended response.", ""]
            elif delta < 0.02:
                L += ["> **The best model clears the ligand-only floor by less than 0.02",
                      f"> Pearson** ({phys.iloc[0].pearson:+.3f} vs {lo:+.3f}). Section",
                      "> 5.3.5 reads that as a ligand shortcut in the split, not as a",
                      "> model problem - tighten the split before changing the",
                      "> architecture.", ""]

    ## ---- 2. ablations --------------------------------------------------------
    L += ["## 2. Ablations (Phase 9)", ""]
    for axis, label in (("plm_mode", "PLM mode (Section 6.7)"),
                        ("variant", "Geometry variant (Section 2.2)"),
                        ("vdw_form", "vdW form (Section 6.2)"),
                        ("residual", "Residual head"),
                        ("terms", "Energy terms")):
        sub = runs[runs.split == a.headline_split].dropna(subset=[axis])
        if sub[axis].nunique() < 2:
            L += [f"### {label}", "",
                  f"_Only {sub[axis].nunique()} level present; this axis has not been "
                  "ablated._", ""]
            continue
        g = (sub.groupby(axis)
             .agg(best_pearson=("pearson", "max"), best_rho=("rho", "max"),
                  n_runs=("run", "count")).reset_index()
             .sort_values("best_pearson", ascending=False))
        L += [f"### {label}", "", md(g.round(4)), ""]
    if not runs.empty and runs[runs.plm_mode == "none"].empty:
        L += ["> `plm.mode=none` has not been run. Section 6.7 asks for it explicitly,",
              "> and the previous sprint's ligand-only control beat every ESM model on",
              "> held-out clusters - so this is the cheapest experiment with the",
              "> highest chance of changing the conclusion.", ""]

    ## ---- 3. splits -----------------------------------------------------------
    L += ["## 3. Across splits", "",
          "The gap between `random` and `protein_cluster` / `double` IS the leakage",
          "measurement. Section 0.4.4 forbids headlining the easy ones.", ""]
    s = (runs.dropna(subset=["split"]).groupby(["split", "variant"])
         .agg(best_pearson=("pearson", "max"), best_rho=("rho", "max"),
              n=("run", "count")).reset_index().sort_values("best_pearson",
                                                            ascending=False))
    L += [md(s.round(4)) if len(s) else "_none_", ""]

    ## ---- 4. diagnostics ------------------------------------------------------
    L += ["## 4. Physics diagnostics (Section 9)", "",
          "If distogram calibration degrades while Pearson improves, the physics is",
          "being gamed (Section 0.3.4) - the affinity loss is buying pK by distorting",
          "geometry, and the result is not a physics result.", ""]
    dg = runs[runs.ece.notna()]
    if dg.empty:
        L += ["_No calibration numbers. These come from structure-bearing validation",
              "rows; if `work/geometry/geometry.pkl` is absent, Stage 1 never ran and",
              "there is no distogram anchor - see `scripts/phase2_geometry.py`._", ""]
    else:
        L += [md(fmt(dg, ["run", "variant", "pearson", "ece", "contact_recall",
                          "pocket_auroc"])), ""]
    dec = [(r.run, r.decomposition) for r in runs.itertuples()
           if isinstance(r.decomposition, dict) and r.decomposition]
    if dec:
        L += ["### Per-term energies and parameter drift", "", "```json",
              json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()}
                          for k, v in dec[:6]}, indent=1), "```", "",
              "Attractive terms are negative. A term whose weighted contribution is",
              "~0 has not earned its place (Section 0.3.5).", ""]

    ## ---- 5. cost -------------------------------------------------------------
    L += ["## 5. Cost", "",
          "If accuracy does not beat pose-based pipelines, the value case rests on",
          "this gap - and Section 9 requires it to be measured, including the folding",
          "step for pipelines that need one.", ""]
    cost = []
    for r in runs.itertuples():
        bj = Path(r.dir) / "benchmark" / "reports.json"
        if bj.exists():
            e = json.loads(bj.read_text()).get("_extra", {})
            if "inference_seconds_per_complex" in e:
                cost.append({"run": r.run, "s_per_complex":
                             e["inference_seconds_per_complex"], "folding": "none"})
    L += [md(pd.DataFrame(cost)) if cost
          else "_Not measured yet; `scripts/phase10_benchmark.py` records it._", ""]

    ## ---- 6. sealed benchmark -------------------------------------------------
    L += ["## 6. Sealed benchmark", "",
          "Scored once per model version (Section 0.4.3). Every scoring event is",
          "logged in `reports/benchmark_log.json`.", ""]
    blog = paths.REPORTS / "benchmark_log.json"
    if blog.exists():
        b = pd.DataFrame(json.loads(blog.read_text()))
        cols = [c for c in ("version", "timestamp", "n_rows", "pearson",
                            "per_target_rho", "low_overlap_n", "rescore")
                if c in b.columns]
        L += [md(b[cols]), ""]
        if b.get("rescore", pd.Series(dtype=bool)).any():
            L += ["> **A version was re-scored.** Section 0.4.3 permits this only for "
                  "a bug that invalidated the first scoring, and it must be disclosed "
                  "here.", ""]
    else:
        L += ["_Not scored yet._", ""]

    ## ---- provenance ----------------------------------------------------------
    prov = {}
    for f in ("corpus_meta.json",):
        p = paths.WORK / f
        if p.exists():
            prov[f] = json.loads(p.read_text())
    for f in ("geometry/meta.json", "ligand_phys/meta.json"):
        p = paths.WORK / f
        if p.exists():
            prov[f] = json.loads(p.read_text())
    idx = paths.SPLITS / "index.json"
    if idx.exists():
        prov["splits"] = json.loads(idx.read_text())
    L += ["## Provenance", "", "```json",
          json.dumps(prov, indent=1, default=str)[:6000], "```", "",
          "## Limitations", "",
          "- Distance-only H-bond and electrostatic terms carry no angular or",
          "  polarisation information; residue-level coarse-graining trades",
          "  directionality for tractability (Section 6.1).",
          "- Sequence-only input hides protonation state, ordered water and",
          "  conformational state; the parameter heads absorb averages of these.",
          "- Ki is pooled with Kd by default. `reports/ki_vs_kd_same_pair.csv` has",
          "  this corpus's own measurement of what that costs.",
          "- The noise ceiling in `reports/phase_1.md` is measured from replicate",
          "  measurements here, not borrowed. An RMSE below the implied sd is fitting",
          "  noise, not signal.", ""]

    Path(a.out).write_text("\n".join(L))
    print("\n".join(L[:60]))
    print(f"\n... wrote {a.out}")
    if not runs.empty:
        b = runs.sort_values("pearson", ascending=False).iloc[0]
        print(f"\nbest run: {b.run}  r {b.pearson:+.4f}  rho {b.rho:+.4f}  "
              f"(ligand-only floor r {b.pearson_ligand_only})")


if __name__ == "__main__":
    main()

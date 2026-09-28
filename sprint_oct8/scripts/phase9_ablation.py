#!/usr/bin/env python3
"""Phase 9: the pre-registered ablation matrix, and Gate G2 (Tier 1).

Two modes.

`--emit` expands `configs/preregistration.yaml` into a numbered list of override
strings, one per cell. Slurm array tasks read cell N from that list, so the grid
lives in one version-controlled place and the sbatch file holds no science.

`--gate` reads the finished runs and applies Section 9's rule verbatim:

    the best MVP configuration beats the matched no-physics control on the
    protein-cluster split by dPearson >= 0.02, with a bootstrap CI excluding zero,
    and beats ligand_only and target_mean.

The gate is paired-bootstrapped on identical test rows, because the two models see
the same items and an unpaired interval on a 0.02 difference is far too wide to
decide anything.

If G2 fails, Section 8 prescribes the diagnosis ORDER - PLM mode first, then
geometry, then individual terms, then data - and `--gate` prints it filled in with
this run's numbers rather than as a checklist.

    python scripts/phase9_ablation.py --emit          # the matrix, one cell per line
    python scripts/phase9_ablation.py --emit --count  # how many array tasks
    python scripts/phase9_ablation.py --gate
"""
from __future__ import annotations

import argparse
import itertools
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

from pfpl import conf, ledger, paths  # noqa: E402
from pfpl.eval import metrics as M  # noqa: E402

## The Phase 9 matrix, in the order Section 8 lists it. Each entry is
## (label, [list of override strings]); the product is the grid.
AXES = [
    ("variant", ["model=variant_a", "model=variant_b", "model=variant_c"]),
    ("plm", ["plm=none", "plm=full", "plm=bottleneck", "plm=pocket_window"]),
    ("vdw", ["model.energy.vdw_form=lj", "model.energy.vdw_form=morse"]),
    ("residual", ["model.energy.residual=false", "model.energy.residual=true"]),
]


def expand(prereg, full: bool, seeds: int) -> list[list[str]]:
    """The grid. `--full` takes the product; otherwise one axis varies at a time.

    One-at-a-time is the default because Section 8's Wave-3 rule ("each addition is
    a SINGLE-CHANGE ablation against the G2 winner") is the reading that survives
    multiplicity. The full product exists for the Wave-2 matrix, which is small.
    """
    base = ["model=variant_a", "plm=bottleneck", "model.energy.vdw_form=lj",
            "model.energy.residual=false"]
    cells: list[list[str]] = []
    if full:
        for combo in itertools.product(*[opts for _, opts in AXES]):
            cells.append(list(combo))
    else:
        cells.append(list(base))
        for i, (_, opts) in enumerate(AXES):
            for o in opts:
                if o == base[i]:
                    continue
                c = list(base)
                c[i] = o
                cells.append(c)
    ## Variant C has no energy function, so its vdw_form and residual cells are the
    ## same run four times. Collapse them rather than burning GPU-hours on duplicates.
    seen, uniq = set(), []
    for c in cells:
        if any(x == "model=variant_c" for x in c):
            c = [x for x in c if not x.startswith("model.energy.")]
        k = tuple(sorted(c))
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    out = []
    all_seeds = (prereg.get("seeds") or [1337])[:max(seeds, 1)]
    for c in uniq:
        for s in all_seeds:
            out.append(c + [f"seed={s}"])
    return out


def load_runs() -> pd.DataFrame:
    rows = []
    for d in sorted(paths.RUNS.glob("*")):
        rj, cj = d / "run.json", d / "reports.json"
        if not (rj.exists() and cj.exists()):
            continue
        info = json.loads(rj.read_text())
        reps = json.loads(cj.read_text())
        m = reps.get("model", {})
        rows.append({
            "run": d.name, "dir": str(d),
            "variant": info.get("variant"), "plm_mode": info.get("plm_mode"),
            "split": info.get("split"), "seed": info.get("seed"),
            "stage": info.get("stage"),
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
            "elapsed_s": info.get("elapsed_s"),
        })
    return pd.DataFrame(rows)


def gate(split: str, margin: float) -> int:
    runs = load_runs()
    if runs.empty:
        raise SystemExit(
            f"no completed runs in {paths.RUNS}.\n"
            "    python scripts/status.py     # what stopped, and where")
    r = runs[(runs.split == split) & (runs.stage.isin(["affinity", "finetune"]))]
    if r.empty:
        raise SystemExit(f"no affinity runs on split {split!r}; have "
                         f"{sorted(runs.split.dropna().unique())}")

    phys = r[r.variant.isin(["A", "B"])].sort_values("pearson", ascending=False)
    ctrl = r[r.variant == "C"].sort_values("pearson", ascending=False)
    if phys.empty or ctrl.empty:
        raise SystemExit(
            "Gate G2 needs BOTH a physics run and the matched no-physics control "
            f"(model=variant_c) on split {split!r}. Have physics={len(phys)}, "
            f"control={len(ctrl)}.\n"
            "  Section 9's rule is defined against the control; without it there is "
            "no gate, only a number.")

    best, control = phys.iloc[0], ctrl.iloc[0]
    print("=" * 78)
    print(f"GATE G2 (Tier 1) - split {split}")
    print("=" * 78)
    print(f"  best physics : {best.run}\n                 r {best.pearson:+.4f}  "
          f"rho {best.rho:+.4f}")
    print(f"  control (C)  : {control.run}\n                 r {control.pearson:+.4f}  "
          f"rho {control.rho:+.4f}")

    pa = pd.read_csv(Path(best.dir) / "predictions.csv")
    pb = pd.read_csv(Path(control.dir) / "predictions.csv")
    j = pa.merge(pb, on="pair_idx", suffixes=("_a", "_b"))
    if len(j) < 100:
        raise SystemExit(f"only {len(j)} shared test rows between the two runs - they "
                         "were scored on different folds, so the comparison is invalid")

    bl = paths.RUNS / f"baselines_{split}" / "reports.json"
    if not bl.exists():
        raise SystemExit(f"{bl} missing - run scripts/phase6_baselines.py. Section "
                         "0.3.1: the physics may not be judged against numbers that "
                         "do not exist yet.")
    base = json.loads(bl.read_text())

    def rep(d: dict, name: str) -> M.Report:
        r = M.Report(name=name)
        for k, v in d.items():
            if hasattr(r, k):
                setattr(r, k, v)
        return r

    mrep = rep(json.loads((Path(best.dir) / "reports.json").read_text())["model"], "physics")
    res = M.tier1_gate(mrep, rep(base["fusion_gbdt"] if "fusion_gbdt" in base
                                 else base["ligand_only"], "control-report"),
                       rep(base["ligand_only"], "ligand_only"),
                       rep(base["target_mean"], "target_mean"),
                       j.y_a.to_numpy(), j.pred_a.to_numpy(), j.pred_b.to_numpy(),
                       margin)
    d = res["delta_pearson_vs_control"]
    print(f"\n  paired bootstrap on {len(j):,} shared rows:")
    print(f"    dPearson = {d['delta']:+.4f}  95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]"
          f"  excludes zero: {d['excludes_zero']}")
    print(f"    required margin: >= {margin}")
    for k in ("beats_control_by_margin", "beats_ligand_only_per_target_rho",
              "beats_ligand_only_pearson", "beats_target_mean_pearson"):
        print(f"    {k:<38} {res[k]}")
    print(f"\n  TIER 1: {'PASS' if res['tier1_pass'] else 'FAIL'}")
    print("=" * 78)

    if not res["tier1_pass"]:
        print("\nSection 8's diagnosis order, with this run's numbers:\n")
        by_plm = r[r.variant.isin(["A", "B"])].groupby("plm_mode").pearson.max()
        print("  1. PLM MODE. Section 6.7: a bottleneck or `none` often helps on")
        print("     strict splits, and ESM-2 reduced Pearson on every LP-PDBBind tier.")
        for k, v in by_plm.sort_values(ascending=False).items():
            print(f"       {k:<22} best r {v:+.4f}")
        if len(by_plm) < 3:
            print("       <- fewer than 3 PLM modes run. This is the FIRST thing to")
            print("          vary and the cheapest; run plm=none before anything else.")
        print("\n  2. DISTOGRAM CALIBRATION / GEOMETRY VARIANT.")
        for _, row in phys.head(3).iterrows():
            h = Path(row.dir) / "history.json"
            if h.exists():
                hist = json.loads(h.read_text())
                last = hist[-1] if hist else {}
                print(f"       {row.run[:44]:<46} ece {last.get('val_disto/ece', float('nan')):.3f}"
                      f"  contact_recall {last.get('val_disto/contact_recall', float('nan')):.3f}")
        print("       A rising ECE while Pearson improves means the physics is being")
        print("       gamed (Section 0.3.4), not that the model is learning.")
        print("\n  3. INDIVIDUAL TERMS AND THE RESIDUAL HEAD.")
        for _, row in phys.head(5).iterrows():
            dec = json.loads((Path(row.dir) / "run.json").read_text()).get(
                "decomposition", {})
            if dec:
                print(f"       {row.run[:44]:<46} "
                      + " ".join(f"{k}={v:+.2f}" for k, v in dec.items()
                                 if not k.startswith("drift/"))[:70])
        print("\n  4. LABEL / DATA QUALITY. reports/phase_1.md has the measured noise")
        print("     ceiling; an RMSE below the implied sd is fitting noise.")
        print("\n  Section 8: if it still fails, STOP the physics expansion and deliver")
        print("  the best fusion model with an honest report of the ablations. Do NOT")
        print("  proceed to Wave 3 by default.")
    return 0 if res["tier1_pass"] else 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--full", action="store_true", help="full product, not one-at-a-time")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--cell", type=int, default=None, help="print one cell's overrides")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--split", default="protein_cluster")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--table", action="store_true", help="dump every run as a table")
    a = ap.parse_args()

    if a.emit or a.count or a.cell is not None:
        prereg = conf.load("preregistration")
        cells = expand(prereg, a.full, a.seeds)
        if a.count:
            print(len(cells))
            return
        if a.cell is not None:
            if a.cell >= len(cells):
                sys.exit(f"cell {a.cell} outside the {len(cells)}-cell grid")
            print(" ".join(f"-o {x}" for x in cells[a.cell]))
            return
        for i, c in enumerate(cells):
            print(f"{i}\t" + " ".join(f"-o {x}" for x in c))
        return

    if a.table:
        r = load_runs()
        if r.empty:
            raise SystemExit("no runs yet")
        cols = ["run", "variant", "plm_mode", "split", "vdw_form", "residual", "seed",
                "pearson", "rho", "rmse", "n_targets", "preregistered"]
        print(r[cols].sort_values("pearson", ascending=False).to_string(index=False))
        print(f"\n  {json.dumps(ledger.summary(), indent=1)}")
        return

    if a.gate:
        sys.exit(gate(a.split, a.margin))
    ap.print_help()


if __name__ == "__main__":
    main()

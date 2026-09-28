#!/usr/bin/env python3
"""Phase 6: the baselines, and Gate G1 (PLAN.md Section 8, Phase 6).

"The physics work may not be judged against numbers that do not exist yet."

This is the most important script in Wave 1 and the cheapest. On the previous
corpus a ligand-only model - 20 RDKit descriptors and an ECFP, no protein input of
any kind - reached median per-target Spearman +0.325 on 613 held-out protein
clusters, while the best model with 7,680 ESM dimensions reached 0.158. Every
protein-aware model lost to a protein-blind control by a factor of two. Running
this first is the difference between discovering that in week one and discovering
it in week three.

Produces, on every requested split:

    const         RMSE floor
    target_mean   ligand-blind pooled ceiling
    ligand_only   the memorisation floor
    protein_only  how much is protein identity alone (pooled only by construction)
    fusion_gbdt   concatenated features - the honest description of Savant v1

plus the Section 0.3.2 shortcut check on each, and a cost line.

    python scripts/phase6_baselines.py
    python scripts/phase6_baselines.py --splits protein_cluster double
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

## Imported FIRST, before torch / matplotlib / transformers: importing
## pfpl.paths redirects every HOME-bound cache and TMPDIR under the project
## root, and some of those libraries read their cache variable at import time.
from pfpl import paths as _paths  # noqa: E402,F401


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pfpl import conf, paths  # noqa: E402
from pfpl.baselines import simple  # noqa: E402
from pfpl.data import splits as SP  # noqa: E402
from pfpl.eval import metrics as M  # noqa: E402


def pooled_protein_features(cfg, n_prot: int) -> np.ndarray | None:
    """Mean-pooled per-residue embeddings, for the protein_only and fusion baselines.

    Pooling is wrong for the model and right here: these baselines exist to measure
    how far protein IDENTITY alone gets you, and a pooled vector is the cleanest
    expression of protein identity.
    """
    ptr = paths.WORK / f"plm_{cfg.plm.mode}.path"
    if not ptr.exists():
        print(f"  no embedding cache for plm.mode={cfg.plm.mode}; skipping the "
              "protein_only and fusion baselines")
        return None
    from pfpl.data.datasets import ProteinStore
    store = ProteinStore(Path(ptr.read_text().strip()))
    out = np.zeros((n_prot, store.dim), np.float32)
    for i in range(min(n_prot, len(store))):
        e, _ = store.get(i)
        if len(e):
            out[i] = e.mean(0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    conf.add_cfg_args(ap)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--no-protein", action="store_true",
                    help="ligand-only and the blind baselines; skips the embedding load")
    a = ap.parse_args()
    cfg = conf.from_args(a)
    paths.ensure()

    pairs = pd.read_parquet(paths.WORK / "pairs.parquet").reset_index(drop=True)
    prot = pd.read_parquet(paths.WORK / "proteins.parquet")
    phys_p = paths.WORK / "ligand_phys.parquet"
    phys = None
    if phys_p.exists():
        lp = pd.read_parquet(phys_p)
        phys = (pairs[["ligand_idx"]].merge(lp, on="ligand_idx", how="left")
                .drop(columns=["ligand_idx"]))
        print(f"  ligand physics: {phys.shape[1]} columns")
    else:
        print("  no work/ligand_phys.parquet - ligand_only will use ECFP alone "
              "(run scripts/phase3_ligand_phys.py to add descriptors)")

    pf = None if a.no_protein else pooled_protein_features(cfg, len(prot))
    names = a.splits or list(cfg.eval.report_splits)
    out_all, gate_lines = {}, []

    for name in names:
        f = paths.SPLITS / f"{name}.npz"
        if not f.exists():
            print(f"\n  split {name!r} not built - run scripts/phase5_splits.py")
            continue
        tr, va, te = SP.load(paths.SPLITS, name)
        print("\n" + "=" * 78)
        print(f"SPLIT {name}   train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}")
        print("=" * 78)
        t0 = time.time()
        res = simple.run_all(pairs, pf, tr, va, te, phys, int(cfg.seed))
        reports = {k: r for k, (r, _) in res.items()}
        M.print_table(reports, f"Phase 6 baselines - split: {name}")

        ## Section 0.3.2 / 5.3.5: the shortcut check, reported whether or not it fires.
        lo = reports.get("ligand_only")
        checks = {}
        for k in ("fusion_gbdt",):
            if k in reports and lo is not None:
                checks[k] = M.shortcut_flag(reports[k], lo,
                                            float(cfg.eval.tier1_margin))
                print(f"\n  shortcut check [{k} vs ligand_only]: "
                      f"{json.dumps(checks[k], indent=1)}")

        d = paths.RUNS / f"baselines_{name}"
        d.mkdir(parents=True, exist_ok=True)
        for k, (r, p) in res.items():
            if p is not None:
                p.to_csv(d / f"pred_{k}.csv", index=False)
        M.write(reports, d / "reports.json",
                extra={"split": name, "shortcut_checks": checks,
                       "seconds": round(time.time() - t0, 1),
                       "plm_mode": str(cfg.plm.mode), "config_hash": cfg.hash,
                       "n_train": int(tr.sum()), "n_test": int(te.sum())})
        out_all[name] = {"reports": {k: v.to_dict() for k, v in reports.items()},
                         "shortcut_checks": checks}
        if lo is not None:
            gate_lines.append(
                (name, lo.pearson, lo.per_target_spearman_median,
                 reports.get("fusion_gbdt").pearson if "fusion_gbdt" in reports else float("nan")))

    _report(out_all)

    print("\n" + "=" * 78)
    print("GATE G1 - baselines exist on every split, with CIs")
    print("=" * 78)
    for name, lr, lrho, fr in gate_lines:
        print(f"  {name:<18} ligand_only r {lr:+.3f} rho {lrho:+.3f}   "
              f"fusion r {fr:+.3f}")
    print("\n  Every later claim is relative to these. Section 9's Tier 1 requires")
    print("  beating the MATCHED no-physics control (model=variant_c) by")
    print(f"  dPearson >= {cfg.eval.tier1_margin} AND beating ligand_only and target_mean.")
    print("  If ligand_only is close to fusion here, tighten the split before")
    print("  building anything (Section 5.3.5).")
    print("=" * 78)


def _report(all_res: dict) -> None:
    L = ["# Phase 6 - baselines (Gate G1)", "",
         "Every later result is relative to these. PLAN.md Section 0.3.2 requires",
         "`ligand_only` and `target_mean` in every table.", "",
         "On the previous corpus, ligand-only reached median per-target rho **+0.325**",
         "on 613 held-out protein clusters while the best 7,680-dimension ESM model",
         "reached **0.158**. Read the ligand_only row first.", ""]
    for split, d in all_res.items():
        L += [f"## split: `{split}`", "",
              "| model | pearson | per-target rho | RMSE | RMSE const | n targets | EF@10 |",
              "|---|---|---|---|---|---|---|"]
        for k, r in d["reports"].items():
            def f(v, s="{:+.3f}"):
                return "-" if v is None or not np.isfinite(v) else s.format(v)
            L.append(f"| `{k}` | {f(r['pearson'])} | "
                     f"{f(r['per_target_spearman_median'])} | "
                     f"{f(r['rmse'], '{:.3f}')} | {f(r['rmse_const'], '{:.3f}')} | "
                     f"{r['n_targets']} | {f(r['ef_at_10_median'], '{:.2f}')} |")
        L.append("")
        if d["shortcut_checks"]:
            L += ["Shortcut check (Section 5.3.5):", "", "```json",
                  json.dumps(d["shortcut_checks"], indent=1), "```", ""]
    (paths.REPORTS / "phase_6.md").write_text("\n".join(L))
    print(f"\n  wrote {paths.REPORTS / 'phase_6.md'}")


if __name__ == "__main__":
    main()

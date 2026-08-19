#!/usr/bin/env python
"""
Step 07 - Retrain the gradient-boosted models under a double-disjoint split.

The benchmark run produced a result the protein-grouped split could not have
predicted:

                       protein-grouped held-out   external benchmark
    GBDT regressor              r 0.580                 r 0.327
    attention regressor         r 0.598                 r 0.269

Internally the two are tied with the transformer marginally ahead; externally
the trees win by a wide margin. So the internal split was not just optimistic -
it ranked the models wrongly, and any model selection based on it would have
shipped the worse one.

The suspected cause is that grouping by protein alone leaves the ligand axis
free, while the benchmark gates ligands at ECFP4 Tanimoto <= 0.50. This trains
the same models under a split disjoint on BOTH axes and reports the two side by
side. If the double-disjoint numbers land near the benchmark's, the split is
validated as the one that predicts deployment behaviour - which is a result in
its own right, independent of Savant.

Models from the disjoint split are written to models/disjoint/ and are trained
on the training fold only, so they can be scored against the benchmark without
having seen their own test rows.

    python 07_split_experiment.py
    python 07_split_experiment.py --protein-frac 0.3 --scaffold-frac 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from savant2.metrics import (METRIC_ORDER, classification_metrics,   # noqa: E402
                             regression_metrics)
from savant2.models import BINDER_LABEL, BinderClassifier, PkRegressor  # noqa: E402
from savant2.splits import (describe, double_disjoint_split,          # noqa: E402
                            murcko_scaffolds, protein_split)

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
OUT = ROOT / "output"
NON_FEATURE = {"id", "ligand_typed", "n_ligand_atoms", "n_protein_atoms"}


def load(features_dir: Path, manifest_path: Path):
    shards = sorted(features_dir.glob("*.parquet")) if features_dir.is_dir() else [features_dir]
    if not shards:
        raise SystemExit(f"no feature shards in {features_dir}")
    features = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    manifest = pd.read_csv(manifest_path)
    keep = ["id", "pkd", "binding_level", "protein_group", "smiles"]
    df = features.merge(manifest[keep], on="id", how="inner")
    return df[df.pkd.notna()].reset_index(drop=True), [
        c for c in features.columns if c not in NON_FEATURE
    ]


def evaluate(df, feature_names, train_idx, test_idx, label: str, save_dir=None):
    """Fit both stages on the training fold and score the held-out fold."""
    X = df[feature_names].to_numpy(np.float32)
    y = df.pkd.to_numpy(np.float64)
    is_binder = (df.binding_level == BINDER_LABEL).to_numpy()
    groups = df.protein_group.to_numpy()

    clf = BinderClassifier(feature_names=feature_names).fit(
        X, is_binder, groups, split=(train_idx, test_idx), refit_on_all=False
    )
    clf_stats = classification_metrics(
        is_binder[test_idx], clf.predict(X[test_idx]), clf.predict_proba(X[test_idx])
    )

    # Stage 2 sees binders only, as in every other v2 path.
    tr_b = train_idx[is_binder[train_idx]]
    te_b = test_idx[is_binder[test_idx]]
    reg = PkRegressor(feature_names=feature_names).fit(
        X, y, groups, split=(tr_b, te_b)
    )
    reg_stats = regression_metrics(y[te_b], reg.model.predict(X[te_b]))

    if save_dir:
        save_dir = Path(save_dir)
        clf.save(save_dir / "classifier.pkl")
        reg.save(save_dir / "regressor.pkl")
        print(f"  saved {save_dir}/classifier.pkl and regressor.pkl", flush=True)

    return {"split": label, **reg_stats}, clf_stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(WORK / "features_training"))
    ap.add_argument("--manifest", default=str(WORK / "manifest_training.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "models" / "disjoint"))
    ap.add_argument("--protein-frac", type=float, default=0.30)
    ap.add_argument("--scaffold-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=str(OUT / "split_experiment.csv"))
    ap.add_argument("--no-size-control", action="store_true",
                    help="skip the equal-training-size control run")
    args = ap.parse_args()

    df, feature_names = load(Path(args.features), Path(args.manifest))
    print(f"{len(df)} complexes, {len(feature_names)} features, "
          f"{df.protein_group.nunique()} proteins", flush=True)

    print("computing Bemis-Murcko scaffolds ...", flush=True)
    scaffolds = murcko_scaffolds(df.smiles.tolist())
    print(f"  {len(np.unique(scaffolds))} distinct scaffolds "
          f"({int((scaffolds == 'ACYCLIC').sum())} acyclic ligands pooled)", flush=True)

    rows, classifiers = [], []

    # --- baseline: what v2 shipped with -----------------------------------
    tr, te, stats = protein_split(df.protein_group.to_numpy(), test_size=0.25, seed=args.seed)
    print("\n" + describe(stats), flush=True)
    reg_stats, clf_stats = evaluate(df, feature_names, tr, te, "protein only")
    rows.append({**reg_stats, **{f"split_{k}": v for k, v in stats.items() if k != "mode"}})
    classifiers.append({"split": "protein only", **clf_stats})

    # --- the experiment ----------------------------------------------------
    tr, te, stats = double_disjoint_split(
        df.protein_group.to_numpy(), scaffolds,
        protein_test_frac=args.protein_frac,
        scaffold_test_frac=args.scaffold_frac, seed=args.seed,
    )
    print("\n" + describe(stats), flush=True)
    reg_stats, clf_stats = evaluate(df, feature_names, tr, te,
                                    "protein + scaffold", save_dir=args.out_dir)
    rows.append({**reg_stats, **{f"split_{k}": v for k, v in stats.items() if k != "mode"}})
    classifiers.append({"split": "protein + scaffold", **clf_stats})

    # --- control: the disjoint split also discards ~44% of rows, so part of any
    # drop could be less training data rather than a harder task. Re-run the
    # protein-only split with its training fold subsampled to the same size, and
    # the difference that remains is attributable to the split itself.
    if not args.no_size_control:
        n_target = stats["n_train"]
        tr_p, te_p, stats_p = protein_split(
            df.protein_group.to_numpy(), test_size=0.25, seed=args.seed
        )
        if len(tr_p) > n_target:
            rng = np.random.RandomState(args.seed)
            tr_small = rng.choice(tr_p, size=n_target, replace=False)
            print(f"\ncontrol: protein-only split, training fold subsampled "
                  f"{len(tr_p)} -> {n_target} to match the disjoint run", flush=True)
            reg_stats, clf_stats = evaluate(df, feature_names, tr_small, te_p,
                                            "protein only (size-matched)")
            rows.insert(1, reg_stats)
            classifiers.insert(1, {"split": "protein only (size-matched)", **clf_stats})

    print("\n=== regressor, held out under each split ===")
    table = pd.DataFrame(rows)
    print(table[["split"] + METRIC_ORDER].to_string(index=False,
                                                    float_format=lambda v: f"{v:.3f}"))
    print("\n=== classifier ===")
    cols = ["split", "n", "binder_recall", "binder_precision", "nonbinder_recall", "roc_auc"]
    ct = pd.DataFrame(classifiers)
    print(ct[[c for c in cols if c in ct.columns]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.report, index=False)
    print(f"\nwrote {args.report}")

    # The comparison this experiment exists to make.
    if len(rows) == 2:
        drop = rows[0]["pearson_r"] - rows[1]["pearson_r"]
        print(f"\nr falls {rows[0]['pearson_r']:.3f} -> {rows[1]['pearson_r']:.3f} "
              f"({drop:+.3f}) when the ligand axis is also held out.")
        print("Compare against this model's score on the external benchmark (0.327).")
        print("A double-disjoint number near the benchmark means the split predicts")
        print("deployment behaviour; a number still well above it means some other")
        print("leak remains - pose source and assay heterogeneity are the candidates.")


if __name__ == "__main__":
    main()

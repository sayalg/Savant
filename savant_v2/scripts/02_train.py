#!/usr/bin/env python
"""
Step 02 - Train the v2 model bundle (gate + pK + sigma + conformal).

Reads whatever feature table it is pointed at, joins the labels and protein
groups from the manifest, fits with a protein-grouped internal split, and saves
one pickled bundle.

``--legacy-features`` trains on v1's eighteen HADDOCK-derived columns instead of
the v2 extraction. That path exists so the v1 and v2 feature sets can be
compared on identical model code, splitting and calibration - otherwise a
difference in score could be the features or could be the training, and the
whole point is to know which.

    python 02_train.py --features work/features_training --manifest work/manifest_training.csv
    python 02_train.py --legacy-features            # v1 features, v2 training code
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from savant2.models import BINDER_LABEL, BinderClassifier, PkRegressor  # noqa: E402
from savant2.metrics import METRIC_ORDER, format_table, regression_metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
WORK = ROOT / "work"
MODELS = ROOT / "models"

LEGACY_FEATURES = [
    "molecular_weight", "logP", "num_h_donors", "num_h_acceptors",
    "num_rotatable_bonds", "tpsa", "num_rings", "num_aromatic_rings",
    "num_atoms", "num_heavy_atoms", "formal_charge",
    "score", "total", "vdw", "elec", "desolv", "bsa", "num_interacting_residues",
]
NON_FEATURE = {"id", "ligand_typed", "n_ligand_atoms", "n_protein_atoms"}


def load_features(path: Path) -> pd.DataFrame:
    """Concatenate parquet shards, or read a single feature file."""
    if path.is_dir():
        shards = sorted(path.glob("*.parquet"))
        if not shards:
            raise SystemExit(f"no parquet shards in {path} - run 01_extract_features.py")
        return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(WORK / "features_training"))
    ap.add_argument("--manifest", default=str(WORK / "manifest_training.csv"))
    ap.add_argument("--legacy-features", action="store_true")
    ap.add_argument("--out-dir", default=str(MODELS),
                    help="writes classifier.pkl and regressor.pkl here")
    ap.add_argument("--alpha", type=float, default=0.1, help="1-alpha conformal coverage")
    ap.add_argument("--gate-recall", type=float, default=0.95,
                    help="fraction of true binders the gate must keep")
    ap.add_argument("--no-ecfp", action="store_true", help="drop the ECFP block")
    args = ap.parse_args()

    if args.legacy_features:
        base = pd.read_csv(REPO / "ML_model" / "combined_50k_filtered.csv")
        labels = pd.read_csv(ROOT / "work_labels.csv")
        df = base.assign(id=base.Job_ID, pkd=base.neg_log10_affinity_M)
        df["binding_level"] = labels.binding_level.values
        source = REPO / "Input" / "input.csv"
        seqs = pd.read_csv(source, usecols=["ID", "seq"]).rename(columns={"ID": "id"})
        df = df.merge(seqs, on="id", how="left")
        df["protein_group"] = df.seq.fillna("").map(lambda s: hash(s) & 0xFFFFFFFF)
        feature_names = LEGACY_FEATURES
    else:
        features = load_features(Path(args.features))
        manifest = pd.read_csv(args.manifest)
        df = features.merge(
            manifest[["id", "pkd", "binding_level", "protein_group"]], on="id", how="inner"
        )
        feature_names = [c for c in features.columns if c not in NON_FEATURE]
        if args.no_ecfp:
            feature_names = [c for c in feature_names if not c.startswith("ecfp_")]

    df = df[df.pkd.notna()]
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df.pkd.to_numpy(dtype=np.float64)
    is_binder = (df.binding_level == BINDER_LABEL).to_numpy()
    groups = df.protein_group.to_numpy()

    print(f"training on {len(df)} complexes, {len(feature_names)} features, "
          f"{len(np.unique(groups))} proteins")
    print(f"  binders {int(is_binder.sum())} / nonbinders {int((~is_binder).sum())}")

    from pathlib import Path as _P
    out_dir = _P(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: binder / nonbinder over every complex.
    clf = BinderClassifier(feature_names=feature_names).fit(
        X, is_binder, groups, recall_target=args.gate_recall)
    clf.save(out_dir / "classifier.pkl")
    print("\nclassifier.pkl  (held-out, protein-grouped)")
    for k, v in clf.metadata["held_out"].items():
        print(f"  {k:20} {v}")

    # Stage 2: pKd, fit on binders only - v1's flow, retained.
    reg = PkRegressor(feature_names=feature_names).fit(
        X[is_binder], y[is_binder], groups[is_binder], alpha=args.alpha)
    reg.save(out_dir / "regressor.pkl")
    print("\nregressor.pkl  (held-out, protein-grouped)")
    print(format_table([reg.metadata["held_out"]], METRIC_ORDER))
    cov = reg.metadata["held_out_coverage"]
    print(f"  conformal: nominal {100*cov['target_coverage']:.0f}%"
          f" -> empirical {100*cov['empirical_coverage']:.1f}%,"
          f" median width {cov['median_interval_width']:.2f} log units")
    print(f"\nwrote {out_dir}/classifier.pkl and {out_dir}/regressor.pkl")


if __name__ == "__main__":
    main()

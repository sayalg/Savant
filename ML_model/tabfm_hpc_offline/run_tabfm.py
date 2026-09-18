#!/usr/bin/env python3
"""Train TabFM on a dataset and save the model + logits, fully offline.

This is the compute-node entrypoint. It never touches the network: weights must
already be on disk (see download_weights.py). TabFM is a zero-shot in-context
model, so `fit` just loads the training rows as context; the real work happens
in the single forward pass at predict time.

Example
-------
    export HF_HUB_OFFLINE=1
    python run_tabfm.py \
        --task classification \
        --weights /scratch/$USER/tabfm \
        --train data/train.csv \
        --test  data/test.csv \
        --target label \
        --out results/run1 \
        --device auto

Outputs (in --out): predictions.csv, proba.{npy,csv}, logits.{npy,csv},
model.joblib, backbone_state_dict.pt, metadata.json.
"""
from __future__ import annotations

import argparse
import sys
import time

# Go offline BEFORE importing anything that pulls in huggingface_hub / torch.
from tabfm_hpc.offline import go_offline


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["classification", "regression"], required=True)
    ap.add_argument("--weights", required=True,
                    help="Local weights dir (download_weights.py --dest).")
    ap.add_argument("--train", required=True, help="Training CSV/Parquet.")
    ap.add_argument("--test", required=True, help="Test/inference CSV/Parquet.")
    ap.add_argument("--target", required=True, help="Target column name.")
    ap.add_argument("--out", required=True, help="Output directory for artifacts.")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32", "none"])
    ap.add_argument("--cache-root", default=None,
                    help="Where HF caches go (default $HF_HOME or ./.hf_home).")
    ap.add_argument("--no-save-model", action="store_true",
                    help="Skip writing model.joblib / backbone_state_dict.pt.")
    args = ap.parse_args()

    go_offline(args.cache_root)

    # Imports deferred until after go_offline() so offline env is in force.
    import numpy as np
    from tabfm_hpc.data import load_xy, load_x
    from tabfm_hpc.model import build_estimator
    from tabfm_hpc.save import extract_logits, save_run

    t0 = time.time()
    print(f"[tabfm-hpc] task={args.task} device={args.device} dtype={args.dtype}")

    X_train, y_train = load_xy(args.train, args.target)
    X_test = load_x(args.test, args.target)
    print(f"[tabfm-hpc] train={X_train.shape}  test={X_test.shape}")

    est = build_estimator(args.task, args.weights,
                          device=args.device, dtype=args.dtype)
    print("[tabfm-hpc] loaded backbone; fitting (loading context)...")
    est.fit(X_train, y_train)

    print("[tabfm-hpc] predicting...")
    predictions = est.predict(X_test)

    proba = None
    logits = None
    classes = None
    if args.task == "classification":
        if hasattr(est, "predict_proba"):
            proba = np.asarray(est.predict_proba(X_test), dtype=np.float64)
        classes = getattr(est, "classes_", None)
        logits = extract_logits(est, X_test, proba)

    written = save_run(
        args.out, task=args.task, estimator=est, X_test=X_test,
        predictions=predictions, proba=proba, logits=logits, classes=classes,
        index=getattr(X_test, "index", None),
        save_model=not args.no_save_model,
        extra_meta={"train_shape": list(X_train.shape),
                    "test_shape": list(X_test.shape),
                    "elapsed_sec": round(time.time() - t0, 2)},
    )

    print(f"[tabfm-hpc] done in {time.time() - t0:.1f}s. Wrote:")
    for name, path in written.items():
        print(f"    {name:24s} {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)

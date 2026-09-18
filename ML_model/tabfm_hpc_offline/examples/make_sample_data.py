#!/usr/bin/env python3
"""Generate tiny train/test CSVs to smoke-test the pipeline end to end.

    python examples/make_sample_data.py --out data
    # -> data/train.csv, data/test.csv  with target column 'label'
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--n-features", type=int, default=8)
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def make(n: int) -> pd.DataFrame:
        num = rng.normal(size=(n, args.n_features - 2))
        cat = rng.integers(0, 4, size=(n, 1))  # a categorical column
        # a signal so the model has something to learn
        score = num[:, 0] * 1.5 - num[:, 1] + cat[:, 0] * 0.3
        y = pd.qcut(score, args.n_classes, labels=range(args.n_classes)).astype(int)
        cols = {f"f{i}": num[:, i] for i in range(num.shape[1])}
        cols["cat0"] = cat[:, 0]
        cols["label"] = y
        return pd.DataFrame(cols)

    make(args.n_train).to_csv(out / "train.csv", index=False)
    # keep label in test.csv too; run_tabfm.py drops it automatically
    make(args.n_test).to_csv(out / "test.csv", index=False)
    print(f"wrote {out}/train.csv and {out}/test.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

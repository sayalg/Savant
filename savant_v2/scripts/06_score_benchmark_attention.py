#!/usr/bin/env python
"""
Step 06 - Score the benchmark with the cross-attention models.

Step 03 evaluates the gradient-boosted bundle; this does the same for the
transformer, so the two model families are compared on identical rows against
identical labels. Both stages run in v1's order: the classifier gates, the
regressor scores what it passes.

Needs attention examples for the benchmark, which come from step 04:

    python scripts/04_build_attention_examples.py --manifest work/manifest_benchmark.csv
    python scripts/06_score_benchmark_attention.py

Writes output/benchmark_attention_scored.csv, merging onto whatever step 03
already produced so one file carries every tool.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from savant2.attention import AttentionModel, collate          # noqa: E402
from savant2.metrics import METRIC_ORDER, regression_metrics   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
WORK = ROOT / "work"
OUT = ROOT / "output"

FALLBACK = REPO / "ML_model" / "benchmark" / "eval" / "output" / "savant_benchmark_1000_scored.csv"
LABEL = "neg_log10_affinity_M"

COMPARATORS = {
    "Boltz-2 affinity": "boltz_affinity_pic50",
    "PRODIGY-LIG": "prodigy_pkd",
    "PRODIGY-LIG (+elec)": "prodigy_pkd_elec",
    "Savant v1": "savant_pk",
    "Savant v2 (GBDT)": "savant2_pk",
    "Savant v2 (attention)": "attn_pk",
}


def predict(model_path: Path, examples, esm_dim: int, batch_size: int, device: str):
    import torch

    model = AttentionModel.load(model_path)
    module = model.module().to(device).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            logits, _ = module(collate(chunk, esm_dim, device))
            out.append(logits.squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0), model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", default=str(WORK / "attention_benchmark"))
    ap.add_argument("--model-dir", default=str(ROOT / "models"))
    ap.add_argument("--base", default=str(OUT / "benchmark_v2_scored.csv"),
                    help="step 03 output; falls back to the eval benchmark CSV")
    ap.add_argument("--out", default=str(OUT / "benchmark_attention_scored.csv"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import joblib
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(args.examples)
    shards = sorted(path.glob("*.joblib")) if path.is_dir() else [path]
    if not shards:
        raise SystemExit(
            f"no attention examples in {path}. Build them first:\n"
            "  python scripts/04_build_attention_examples.py "
            "--manifest work/manifest_benchmark.csv"
        )
    examples, esm_dim = [], None
    for shard in shards:
        blob = joblib.load(shard)
        examples.extend(blob["examples"])
        esm_dim = blob["esm_dim"]
    print(f"{len(examples)} benchmark complexes, ESM dim {esm_dim}, device {device}")

    model_dir = Path(args.model_dir)
    reg_path = model_dir / "attention_regressor.pkl"
    clf_path = model_dir / "attention_classifier.pkl"
    if not reg_path.exists():
        raise SystemExit(f"{reg_path} not found - run scripts/05_train_attention.py")

    pk, reg = predict(reg_path, examples, esm_dim, args.batch_size, device)
    frame = pd.DataFrame({"benchmark_id": [e["id"] for e in examples], "attn_pk": pk})

    if clf_path.exists():
        logits, clf = predict(clf_path, examples, esm_dim, args.batch_size, device)
        frame["attn_p_binder"] = 1.0 / (1.0 + np.exp(-logits))
        # Same staging as everywhere else: the gate decides, the regressor scores
        # what it passes. 0.5 on a pos_weighted logit is the natural threshold.
        frame["attn_is_binder"] = frame.attn_p_binder >= 0.5
    frame["attn_kd_nM"] = 10 ** (9 - frame.attn_pk)

    base_path = Path(args.base) if Path(args.base).exists() else FALLBACK
    print(f"merging onto {base_path.name}")
    merged = pd.read_csv(base_path).merge(frame, on="benchmark_id", how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(merged)} rows)")

    cols = [c for c in COMPARATORS.values() if c in merged.columns]
    common = merged[merged[cols].notna().all(axis=1) & merged[LABEL].notna()]
    print(f"\nHead-to-head on the {len(common)} pairs every tool scored:")
    rows = [{"tool": k, **regression_metrics(common[LABEL], common[v])}
            for k, v in COMPARATORS.items() if v in common]
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nCoverage of the full 1000:")
    for name, col in COMPARATORS.items():
        if col in merged:
            print(f"  {name:24} {int(merged[col].notna().sum()):4d}")

    # Two model families on the same rows: does the transformer actually beat
    # the trees, or is the difference inside the noise?
    if "savant2_pk" in common and "attn_pk" in common and len(common) > 20:
        rng = np.random.default_rng(0)
        t = common[LABEL].to_numpy()
        a = common["attn_pk"].to_numpy()
        g = common["savant2_pk"].to_numpy()
        deltas = [np.corrcoef(t[i], a[i])[0, 1] - np.corrcoef(t[i], g[i])[0, 1]
                  for i in rng.integers(0, len(t), (2000, len(t)))]
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        print(f"\nattention minus GBDT: delta r = {np.mean(deltas):+.3f} "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"P(attention better) = {np.mean(np.array(deltas) > 0):.2f}")

    if "novelty_tier" in common.columns:
        print("\nBy novelty tier (Pearson r):")
        rows = []
        for tier, grp in common.groupby("novelty_tier"):
            row = {"tier": tier, "n": len(grp)}
            for name, col in COMPARATORS.items():
                if col in grp:
                    row[name] = regression_metrics(grp[LABEL], grp[col])["pearson_r"]
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()

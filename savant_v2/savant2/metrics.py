"""The evaluation metrics every v2 result is reported with.

RMSE, MAE, MAPE, Pearson r and Spearman rho, computed the same way everywhere so
numbers from training, the benchmark and the CLI are directly comparable.

A note on MAPE, because it is the one that misleads here. It is computed on pKd,
not on Kd. On Kd it would be meaningless: predictions span twelve orders of
magnitude, so a 1-log-unit miss on a picomolar binder and on a millimolar one
produce percentage errors differing by a factor of a billion. On pKd the values
sit in a bounded 0.4-15 range and MAPE behaves. It still inflates for the
weakest binders - a pKd of 0.4 makes any absolute error look enormous in
percentage terms - so it is reported alongside RMSE and MAE rather than instead
of them.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

METRIC_ORDER = ["n", "rmse", "mae", "mape", "pearson_r", "spearman_rho"]


def regression_metrics(
    y_true, y_pred, sample_weight: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """RMSE, MAE, MAPE, Pearson r, Spearman rho on paired non-null values."""
    y_true = pd.Series(np.asarray(y_true, dtype=float))
    y_pred = pd.Series(np.asarray(y_pred, dtype=float))
    mask = y_true.notna() & y_pred.notna()
    n = int(mask.sum())
    if n < 3:
        return {"n": n, **{k: float("nan") for k in METRIC_ORDER if k != "n"}}

    t, p = y_true[mask].to_numpy(), y_pred[mask].to_numpy()
    err = t - p
    # Guard the division: a true pKd of exactly 0 would make MAPE infinite.
    denom = np.where(np.abs(t) < 1e-9, np.nan, np.abs(t))

    # Correlation is undefined when either side has no variance, which happens
    # for real during early training (the model outputs a constant) and on tiny
    # slices. numpy would emit a divide warning and return nan; return the nan
    # quietly instead, so a training log stays readable.
    constant = np.std(t) < 1e-12 or np.std(p) < 1e-12
    pearson = float("nan") if constant else float(np.corrcoef(t, p)[0, 1])
    spearman = float("nan") if constant else float(
        pd.Series(t).corr(pd.Series(p), method="spearman")
    )

    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.nanmean(np.abs(err) / denom) * 100.0),
        "pearson_r": pearson,
        "spearman_rho": spearman,
    }


def classification_metrics(y_true_binder, y_pred_binder, p_binder=None) -> Dict[str, float]:
    """Gate performance. Recall on true binders is the number that matters:
    every binder the gate rejects is a compound the regressor never scores."""
    t = np.asarray(y_true_binder, dtype=bool)
    p = np.asarray(y_pred_binder, dtype=bool)
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    tn = int((~t & ~p).sum())
    out = {
        "n": int(len(t)),
        "accuracy": float((tp + tn) / max(len(t), 1)),
        "binder_recall": float(tp / max(tp + fn, 1)),
        "binder_precision": float(tp / max(tp + fp, 1)),
        "nonbinder_recall": float(tn / max(tn + fp, 1)),
        "true_pos": tp, "false_pos": fp, "false_neg": fn, "true_neg": tn,
    }
    if p_binder is not None and 0 < t.sum() < len(t):
        try:
            from sklearn.metrics import roc_auc_score
            out["roc_auc"] = float(roc_auc_score(t, np.asarray(p_binder, dtype=float)))
        except Exception:
            pass
    return out


def format_table(rows, columns=None) -> str:
    """Markdown table from a list of metric dicts."""
    frame = pd.DataFrame(rows)
    if columns:
        frame = frame[[c for c in columns if c in frame.columns]]
    return frame.to_string(index=False, float_format=lambda v: f"{v:.3f}")

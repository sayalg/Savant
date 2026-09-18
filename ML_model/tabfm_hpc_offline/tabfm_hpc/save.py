"""Persist a fitted TabFM run so the results are usable from other frameworks.

We deliberately write plain, framework-neutral formats:

  logits.npy / logits.csv     raw pre-softmax scores  (classification)
  proba.npy  / proba.csv      class probabilities     (classification)
  predictions.csv             final predictions (class label or regressed value)
  metadata.json               shapes, class order, task, dtype, versions
  model.joblib                the fitted sklearn-style estimator (reload in py)
  backbone_state_dict.pt      raw torch weights of the backbone (for other code)

Logits are what most downstream frameworks (XGBoost stacking, calibration,
ensembling, a JAX/other-lib head) actually want, so getting them out cleanly is
the point of this module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def extract_logits(estimator: Any, X: Any, proba: np.ndarray | None) -> np.ndarray | None:
    """Best-effort recovery of raw class logits from a classifier.

    Tries, in order: a native logits/decision_function method, then a log of the
    probabilities as a valid (up-to-additive-constant) softmax pre-image.
    Returns None if nothing usable is available.
    """
    for attr in ("predict_logits", "decision_function"):
        fn = getattr(estimator, attr, None)
        if callable(fn):
            try:
                out = np.asarray(fn(X), dtype=np.float64)
                if out.ndim == 1:  # binary decision_function -> 2-column logits
                    out = np.column_stack([-out, out])
                return out
            except Exception:  # noqa: BLE001 - fall back to log-proba
                pass
    if proba is not None:
        # softmax(log(p)) == p, so log(p) is a faithful logit vector.
        return np.log(np.clip(proba.astype(np.float64), 1e-12, 1.0))
    return None


def _versions() -> dict[str, str]:
    v: dict[str, str] = {}
    for mod in ("torch", "tabfm", "numpy", "pandas", "sklearn", "huggingface_hub"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            v[mod] = "unavailable"
    return v


def save_run(out_dir: str, *, task: str, estimator: Any, X_test: Any,
             predictions: np.ndarray, proba: np.ndarray | None,
             logits: np.ndarray | None, classes: Any | None,
             index: Any | None = None, save_model: bool = True,
             extra_meta: dict | None = None) -> dict[str, str]:
    """Write all artifacts to out_dir. Returns a map of name -> path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def _record(name: str, p: Path) -> None:
        written[name] = str(p)

    # --- predictions ---------------------------------------------------------
    pred_df = pd.DataFrame({"prediction": np.asarray(predictions).ravel()})
    if index is not None:
        pred_df.insert(0, "index", np.asarray(index))
    pred_path = out / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    _record("predictions", pred_path)

    # --- probabilities + logits (classification) -----------------------------
    class_labels = [str(c) for c in classes] if classes is not None else None
    if proba is not None:
        np.save(out / "proba.npy", proba)
        cols = class_labels if class_labels and len(class_labels) == proba.shape[1] else \
            [f"class_{i}" for i in range(proba.shape[1])]
        pd.DataFrame(proba, columns=cols).to_csv(out / "proba.csv", index=False)
        _record("proba_npy", out / "proba.npy")
        _record("proba_csv", out / "proba.csv")

    if logits is not None:
        np.save(out / "logits.npy", logits)
        cols = class_labels if class_labels and len(class_labels) == logits.shape[1] else \
            [f"class_{i}" for i in range(logits.shape[1])]
        pd.DataFrame(logits, columns=cols).to_csv(out / "logits.csv", index=False)
        _record("logits_npy", out / "logits.npy")
        _record("logits_csv", out / "logits.csv")

    # --- fitted model + raw backbone weights ---------------------------------
    if save_model:
        try:
            import joblib
            mpath = out / "model.joblib"
            joblib.dump(estimator, mpath)
            _record("model_joblib", mpath)
        except Exception as e:  # noqa: BLE001
            written["model_joblib_error"] = repr(e)

        try:
            import torch
            backbone = getattr(estimator, "model", None) or getattr(estimator, "model_", None)
            if backbone is not None and hasattr(backbone, "state_dict"):
                spath = out / "backbone_state_dict.pt"
                torch.save(backbone.state_dict(), spath)
                _record("backbone_state_dict", spath)
        except Exception as e:  # noqa: BLE001
            written["backbone_state_dict_error"] = repr(e)

    # --- metadata ------------------------------------------------------------
    meta = {
        "task": task,
        "n_test_rows": int(np.asarray(predictions).shape[0]),
        "n_features": int(getattr(X_test, "shape", (0, 0))[1]),
        "classes": class_labels,
        "has_proba": proba is not None,
        "has_logits": logits is not None,
        "logits_note": ("raw logits if the model exposed them, else log(proba) "
                        "which is a valid softmax pre-image"),
        "versions": _versions(),
        "artifacts": {k: v for k, v in written.items() if not k.endswith("_error")},
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    _record("metadata", meta_path)

    return written

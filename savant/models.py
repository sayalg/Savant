"""Load and run the two trained Savant models on the in-memory feature row.

``tender_comb_*`` is a scikit-learn RandomForest pipeline that tags a complex
as ``binder`` / ``nonbinder``. ``mango_sponge_*`` is an XGBoost pipeline that
predicts pK (``neg_log10_affinity_M``). Only binders are sent to the regressor.

Both pipelines expect exactly the 18 features in ``FEATURE_COLUMNS`` order.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd

from savant import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

BINDER_LABEL = "binder"

# Where to look for the .pkl model artifacts when no explicit path is given.
DEFAULT_MODEL_DIR = os.environ.get("SAVANT_MODEL_DIR", "ML_model/train")

# Preferred glob patterns (the azureml-free "_local" builds come first).
_CLASSIFIER_PATTERNS = ["tender_comb_*_local.pkl", "tender_comb_*.pkl"]
_REGRESSOR_PATTERNS = ["mango_sponge_*_local.pkl", "mango_sponge_*.pkl"]


def _discover(model_dir: Path, patterns: list[str], kind: str) -> Path:
    """Return the first model file in ``model_dir`` matching ``patterns``."""
    for pattern in patterns:
        matches = sorted(model_dir.glob(pattern))
        if matches:
            if len(matches) > 1:
                logger.warning(
                    "Multiple %s models match %s in %s; using %s",
                    kind, pattern, model_dir, matches[0].name,
                )
            return matches[0]
    raise FileNotFoundError(
        f"No {kind} model found in '{model_dir}' (looked for {patterns}). "
        f"Pass an explicit path or set SAVANT_MODEL_DIR."
    )


def resolve_model_paths(
    model_dir: str | Path,
    classifier_model: Optional[str | Path],
    regressor_model: Optional[str | Path],
) -> Tuple[Path, Path]:
    """Resolve classifier and regressor .pkl paths, discovering by pattern if
    an explicit path is not supplied."""
    mdir = Path(model_dir)
    clf_path = Path(classifier_model) if classifier_model else _discover(
        mdir, _CLASSIFIER_PATTERNS, "classifier"
    )
    reg_path = Path(regressor_model) if regressor_model else _discover(
        mdir, _REGRESSOR_PATTERNS, "regressor"
    )
    for path, kind in ((clf_path, "classifier"), (reg_path, "regressor")):
        if not path.is_file():
            raise FileNotFoundError(f"{kind} model not found: {path}")
    return clf_path, reg_path


def _features_frame(features: Dict[str, float]) -> pd.DataFrame:
    """One-row DataFrame with the exact column order the models were trained on."""
    return pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def classify(features: Dict[str, float], classifier_path: Path) -> Tuple[str, Dict[str, float]]:
    """Return (label, {class: probability}) for the feature row."""
    model = joblib.load(classifier_path)
    X = _features_frame(features)
    label = str(model.predict(X)[0])
    probabilities: Dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        classes = [str(c) for c in model.classes_]
        probabilities = {
            cls: float(p) for cls, p in zip(classes, model.predict_proba(X)[0])
        }
    logger.info("Classifier: %s (%s)", label, probabilities or "no probabilities")
    return label, probabilities


def predict_pk(features: Dict[str, float], regressor_path: Path) -> float:
    """Return the predicted pK (neg_log10_affinity_M) for the feature row."""
    model = joblib.load(regressor_path)
    # Older-sklearn scalers unpickled here may lack the newer `clip` attribute.
    scaler = model.named_steps.get("scaler")
    if scaler is not None and not hasattr(scaler, "clip"):
        scaler.clip = False
    pk = float(model.predict(_features_frame(features))[0])
    logger.info("Regressor pK: %.4f", pk)
    return pk

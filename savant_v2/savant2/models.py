"""Two models, two files, same flow as v1.

    classifier.pkl   binder vs nonbinder
    regressor.pkl    pKd, applied only to what the classifier passes

The staging is v1's and is kept deliberately. What changes is the gate's
behaviour. v1 took the classifier's argmax and rejected 288 of 935 benchmark
pairs (31%) that are all measured binders - it trained on 94% binders and then
used a 0.5 threshold. v2 tunes the threshold on a held-out protein-grouped split
to a target recall, and reports what it screens out instead of dropping it.

Both models are gradient-boosted trees over the full feature vector (geometry +
ligand + ECFP + pooled ESM). The cross-attention alternative lives in
attention.py and plugs into the same two-stage flow with the same file format.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from savant2.conformal import ConformalRegressor
from savant2.metrics import classification_metrics, regression_metrics

BINDER_LABEL = "binder"
NONBINDER_LABEL = "nonbinder"
# Recovered from v1's combined_50k_filtered_prepped_binary_classification.csv:
# nonbinder is pKd <= 4.0, i.e. Kd at or above 100 uM.
BINDER_PKD_THRESHOLD = 4.0


def _classifier(**kw):
    from sklearn.ensemble import HistGradientBoostingClassifier
    params = dict(max_iter=400, learning_rate=0.06, l2_regularization=1.0,
                  random_state=0, class_weight="balanced")
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


def _regressor(**kw):
    from sklearn.ensemble import HistGradientBoostingRegressor
    params = dict(max_iter=600, learning_rate=0.05, l2_regularization=1.0, random_state=0)
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


def _grouped_split(X, y, groups, test_size=0.25, seed=0):
    """Protein-grouped split: the same target never lands on both sides."""
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(X, y, groups))


@dataclass
class BinderClassifier:
    """Stage 1: does this complex bind at all (pKd > 4, Kd < 100 uM)?"""

    feature_names: List[str]
    model: object = None
    threshold: float = 0.5
    metadata: Dict = field(default_factory=dict)

    def fit(self, X, is_binder, groups, recall_target: float = 0.95, seed: int = 0,
            split=None, refit_on_all: bool = True):
        """``split`` overrides the internal protein-grouped split with explicit
        ``(train_idx, test_idx)``; ``refit_on_all=False`` keeps the final model
        trained on the training fold only, which a split experiment needs so the
        saved model has never seen its own test rows."""
        fit_idx, cal_idx = split if split is not None else _grouped_split(
            X, is_binder, groups, seed=seed)
        self.model = _classifier().fit(X[fit_idx], is_binder[fit_idx])

        proba = self.model.predict_proba(X[cal_idx])[:, 1]
        held_binder = is_binder[cal_idx].astype(bool)
        if held_binder.any():
            # The threshold that retains `recall_target` of genuine binders on
            # data the classifier has not seen.
            self.threshold = float(np.quantile(proba[held_binder], 1 - recall_target))

        held = classification_metrics(held_binder, proba >= self.threshold, proba)
        if refit_on_all:
            self.model = _classifier().fit(X, is_binder)   # all data for shipping
        self.metadata = {
            "n_train": int(len(X)), "n_binders": int(is_binder.sum()),
            "threshold": self.threshold, "recall_target": recall_target,
            "binder_pkd_threshold": BINDER_PKD_THRESHOLD,
            "held_out": held,
        }
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X) >= self.threshold

    def save(self, path): _save(self, path)

    @staticmethod
    def load(path) -> "BinderClassifier": return _load(path)


@dataclass
class PkRegressor:
    """Stage 2: pKd for complexes the classifier passed.

    Carries a sigma head and a conformal calibration so every prediction comes
    with an interval rather than a bare number.
    """

    feature_names: List[str]
    model: object = None
    sigma: object = None
    conformal: Optional[ConformalRegressor] = None
    metadata: Dict = field(default_factory=dict)

    def fit(self, X, y, groups, alpha: float = 0.1, seed: int = 0, split=None):
        """``split`` overrides the internal protein-grouped split with explicit
        ``(train_idx, test_idx)``. The regressor never refits on everything, so
        its reported metrics are always on rows it did not train on."""
        fit_idx, cal_idx = split if split is not None else _grouped_split(
            X, y, groups, seed=seed)
        self.model = _regressor().fit(X[fit_idx], y[fit_idx])

        cal_pred = self.model.predict(X[cal_idx])
        held = regression_metrics(y[cal_idx], cal_pred)

        residuals = np.abs(y[cal_idx] - cal_pred)
        self.sigma = _regressor(max_iter=300).fit(X[cal_idx], residuals)
        self.conformal = ConformalRegressor(alpha=alpha, normalized=True).calibrate(
            y[cal_idx], cal_pred, self.sigma.predict(X[cal_idx])
        )
        coverage = self.conformal.coverage(
            y[cal_idx], cal_pred, self.sigma.predict(X[cal_idx])
        )
        self.metadata = {
            "n_train": int(len(X)), "alpha": alpha,
            "conformal_quantile": self.conformal.quantile_,
            "held_out": held, "held_out_coverage": coverage,
        }
        return self

    def predict(self, X) -> Dict[str, np.ndarray]:
        pk = self.model.predict(X)
        sigma = np.maximum(self.sigma.predict(X), 1e-3)
        lo, hi = self.conformal.interval(pk, sigma)
        return {"pk": pk, "sigma": sigma, "pk_lower": lo, "pk_upper": hi}

    def save(self, path): _save(self, path)

    @staticmethod
    def load(path) -> "PkRegressor": return _load(path)


@dataclass
class SavantBundle:
    """Both stages, for scoring. Loads two .pkl files written independently."""

    classifier: BinderClassifier
    regressor: PkRegressor

    @property
    def feature_names(self) -> List[str]:
        return self.regressor.feature_names

    @property
    def gate_threshold(self) -> float:
        return self.classifier.threshold

    @property
    def conformal(self):
        return self.regressor.conformal

    def predict(self, X) -> Dict[str, np.ndarray]:
        p_binder = self.classifier.predict_proba(X)
        out = self.regressor.predict(X)
        out["p_binder"] = p_binder
        out["is_binder"] = p_binder >= self.classifier.threshold
        return out

    @staticmethod
    def load(directory) -> "SavantBundle":
        directory = Path(directory)
        return SavantBundle(
            classifier=BinderClassifier.load(directory / "classifier.pkl"),
            regressor=PkRegressor.load(directory / "regressor.pkl"),
        )


def _save(obj, path) -> None:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    path.with_suffix(".json").write_text(
        json.dumps(obj.metadata, indent=2, default=float) + "\n"
    )


def _load(path):
    import joblib
    return joblib.load(path)

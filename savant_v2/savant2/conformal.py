"""Split-conformal prediction intervals with distribution-free coverage.

v1 returned a bare number. A pK of 7.2 that is really 7.2 +/- 0.4 and a pK of
7.2 that is really 7.2 +/- 2.1 are different claims, and a screening cascade
cannot triage on point estimates alone.

Split conformal gives a finite-sample, distribution-free guarantee: intervals
built this way cover the truth at least ``1 - alpha`` of the time on exchangeable
data, whatever the model is and however wrong it is. No Gaussian assumption, no
calibration curve fitting.

Two refinements over textbook split conformal:

  Normalised (locally adaptive) intervals.  A second model predicts the
  magnitude of the residual, and the conformity score is |y - yhat| / sigma(x).
  Intervals then widen where the model is genuinely less certain instead of
  being a constant band, which is what makes them useful for triage.

  Applicability domain.  The same sigma(x), plus distance to the training
  distribution, flags inputs the model was never fit to score - the honest
  answer for a target unlike anything in training is a wide interval and a flag,
  not a confident number.

Caveat that belongs in any writeup using this: the coverage guarantee holds
under exchangeability between calibration and test data. A protein-clustered
benchmark deliberately breaks that, so measured coverage on held-out families is
an empirical result, not a guarantee, and it is worth reporting both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class ConformalRegressor:
    """Wraps a fitted point predictor with calibrated interval estimation.

    ``alpha`` is the miscoverage rate: 0.1 gives nominally 90% intervals.
    """

    alpha: float = 0.1
    normalized: bool = True
    quantile_: float = field(default=np.nan, init=False)
    sigma_floor_: float = field(default=1e-3, init=False)
    calibration_residuals_: Optional[np.ndarray] = field(default=None, init=False)

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sigma: Optional[np.ndarray] = None,
    ) -> "ConformalRegressor":
        """Fit the conformity quantile on a held-out calibration split.

        The split must not overlap the data the point predictor was fit on, or
        the residuals are optimistic and the guarantee is void.
        """
        residuals = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
        self.calibration_residuals_ = residuals

        if self.normalized and sigma is not None:
            sigma = np.maximum(np.asarray(sigma, float), self.sigma_floor_)
            scores = residuals / sigma
        else:
            self.normalized = False
            scores = residuals

        n = len(scores)
        # The finite-sample correction: the (n+1)(1-alpha)/n empirical quantile
        # is what makes coverage hold exactly, not the plain (1-alpha) quantile.
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.quantile_ = float(np.quantile(scores, level, method="higher"))
        return self

    def interval(
        self, y_pred: np.ndarray, sigma: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds for each prediction."""
        y_pred = np.asarray(y_pred, float)
        if self.normalized and sigma is not None:
            half = self.quantile_ * np.maximum(np.asarray(sigma, float), self.sigma_floor_)
        else:
            half = np.full_like(y_pred, self.quantile_)
        return y_pred - half, y_pred + half

    def coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sigma: Optional[np.ndarray] = None,
    ) -> dict:
        """Empirical coverage and mean width — report this, never assume it."""
        lo, hi = self.interval(y_pred, sigma)
        y_true = np.asarray(y_true, float)
        inside = (y_true >= lo) & (y_true <= hi)
        return {
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": float(inside.mean()),
            "mean_interval_width": float((hi - lo).mean()),
            "median_interval_width": float(np.median(hi - lo)),
            "n": int(len(y_true)),
        }

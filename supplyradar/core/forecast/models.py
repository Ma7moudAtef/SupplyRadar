"""Step 5: the forecast model library.

Every model implements fit_predict(train, horizon) -> np.ndarray and carries a
complexity rank for the Occam tiebreak. statsmodels-backed models degrade
gracefully: a fit failure returns None and the pipeline is skipped, never
crashed. Prophet / XGBoost / LightGBM are optional in the spec and are not
shipped in v1.

Weighted-average models (uses_weights=True) receive the monthly production
quantity of each training month and weight the consumption rate by it — a
month that produced more tonnage/heats counts more toward the average rate,
which is the correct way to aggregate a ratio. Without weights they fall back
to recency weighting.

Rates cannot be negative: all forecasts are clipped at zero.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

SEASON = 12  # monthly data

@dataclass(frozen=True)
class ModelSpec:
    name: str
    complexity: int              # 1 = simplest; Occam tiebreak prefers low
    min_points: int
    fn: Callable[..., np.ndarray | None]
    intermittent_only: bool = False
    heavy: bool = False          # MLE-fitted: competed only on long lookbacks
    uses_weights: bool = False   # production quantities weight the average

    def fit_predict(self, train: np.ndarray, horizon: int,
                    weights: np.ndarray | None = None) -> np.ndarray | None:
        if len(train) < self.min_points:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_a = np.asarray(train, dtype=float)
                if self.uses_weights:
                    w = None if weights is None else np.asarray(weights, dtype=float)
                    out = self.fn(train_a, horizon, weights=w)
                else:
                    out = self.fn(train_a, horizon)
        except Exception:
            return None
        if out is None:
            return None
        out = np.asarray(out, dtype=float)
        if len(out) != horizon or not np.all(np.isfinite(out)):
            return None
        return np.maximum(out, 0.0)

def _naive(train, h):
    return np.repeat(train[-1], h)

def _seasonal_naive(train, h):
    if len(train) < SEASON + 1:
        return None
    last_season = train[-SEASON:]
    return np.array([last_season[i % SEASON] for i in range(h)])

def _moving_average(train, h, k=3):
    return np.repeat(train[-min(k, len(train)):].mean(), h)

def _weighted_moving_average(train, h, k=4, weights=None):
    """Average of the last k rates. Weight each month by its production
    quantity when available (a high-output month drives the rate more);
    otherwise fall back to recency weights (older months count less)."""
    tail = train[-min(k, len(train)):]
    w = None
    if weights is not None and len(weights) >= len(tail):
        cand = np.asarray(weights[-len(tail):], dtype=float)
        if np.all(np.isfinite(cand)) and cand.sum() > 0:
            w = cand
    if w is None:
        w = np.arange(1, len(tail) + 1, dtype=float)
    return np.repeat(float(np.dot(tail, w) / w.sum()), h)

def _ses(train, h):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    fit = SimpleExpSmoothing(train, initialization_method="estimated").fit()
    return fit.forecast(h)

def _holt(train, h):
    from statsmodels.tsa.holtwinters import Holt
    fit = Holt(train, initialization_method="estimated").fit()
    return fit.forecast(h)

def _ets_damped(train, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = ExponentialSmoothing(train, trend="add", damped_trend=True,
                               initialization_method="estimated").fit()
    return fit.forecast(h)

def _holt_winters(train, h):
    if len(train) < 2 * SEASON:
        return None
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = ExponentialSmoothing(train, trend="add", seasonal="add",
                               seasonal_periods=SEASON,
                               initialization_method="estimated").fit()
    return fit.forecast(h)

def _arima(train, h):
    from statsmodels.tsa.arima.model import ARIMA
    fit = ARIMA(train, order=(1, 1, 1)).fit()
    return fit.forecast(h)

def _sarima(train, h):
    if len(train) < 2 * SEASON + 4:
        return None
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    fit = SARIMAX(train, order=(1, 0, 1),
                  seasonal_order=(1, 1, 0, SEASON)).fit(disp=False)
    return fit.forecast(h)

def _croston(train, h, alpha=0.2):
    """Croston's method for intermittent demand: smoothed size / interval."""
    nz = np.flatnonzero(train > 0)
    if len(nz) == 0:
        return np.zeros(h)
    size = train[nz[0]]
    interval = nz[0] + 1.0
    last = nz[0]
    for i in nz[1:]:
        size += alpha * (train[i] - size)
        interval += alpha * ((i - last) - interval)
        last = i
    return np.repeat(size / max(interval, 1.0), h)

MODEL_LIBRARY: list[ModelSpec] = [
    ModelSpec("Naive", 1, 1, _naive),
    ModelSpec("Moving Average", 1, 2, _moving_average),
    ModelSpec("Weighted Moving Average", 2, 2, _weighted_moving_average,
              uses_weights=True),
    ModelSpec("Seasonal Naive", 2, SEASON + 1, _seasonal_naive),
    ModelSpec("Simple Exponential Smoothing", 2, 4, _ses),
    ModelSpec("Holt Linear Trend", 3, 5, _holt),
    ModelSpec("ETS (damped trend)", 3, 8, _ets_damped, heavy=True),
    ModelSpec("Holt-Winters", 4, 2 * SEASON, _holt_winters, heavy=True),
    ModelSpec("ARIMA(1,1,1)", 4, 10, _arima, heavy=True),
    ModelSpec("SARIMA(1,0,1)(1,1,0)12", 5, 2 * SEASON + 4, _sarima, heavy=True),
    ModelSpec("Croston (Intermittent)", 2, 4, _croston, intermittent_only=True),
]

def applicable_models(n_points: int, *, intermittent: bool) -> list[ModelSpec]:
    """Models worth competing for a series of n_points months."""
    out = []
    for spec in MODEL_LIBRARY:
        if spec.min_points > n_points:
            continue
        if spec.intermittent_only and not intermittent:
            continue
        out.append(spec)
    return out

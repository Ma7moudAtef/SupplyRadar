"""Step 6: rolling-origin cross validation.

Chronology is sacred: train always ends strictly before the predicted period,
the origin rolls forward one month at a time, and historical data is NEVER
randomly split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import ModelSpec


@dataclass
class CVResult:
    model: str
    window: int                 # lookback in months (n = full history)
    n_folds: int
    mae: float
    rmse: float
    mape: float                 # % (NaN if every actual is zero)
    smape: float                # %
    mase: float                 # vs in-sample naive scale (NaN if scale 0)
    stability: float            # std of absolute fold errors
    abs_errors: np.ndarray = field(repr=False, default=None)
    residuals: np.ndarray = field(repr=False, default=None)

def rolling_validate(values: np.ndarray, spec: ModelSpec, window: int, *,
                     min_train: int = 2, max_folds: int = 8
                     ) -> CVResult | None:
    """One-step-ahead rolling-origin validation of (model, lookback window).

    For each origin t: train on values[t-window : t] (clipped to history
    start), predict period t, compare to the actual. Returns None when the
    pipeline never produced a usable forecast.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    origins = [t for t in range(max(min_train, spec.min_points), n)]
    origins = origins[-max_folds:]
    if not origins:
        return None

    preds, actuals = [], []
    for t in origins:
        train = v[max(0, t - window): t]
        pred = spec.fit_predict(train, 1)
        if pred is None:
            continue
        preds.append(float(pred[0]))
        actuals.append(v[t])
    if len(preds) < 2:
        return None

    preds_a = np.array(preds)
    actuals_a = np.array(actuals)
    resid = actuals_a - preds_a
    abs_err = np.abs(resid)

    nz = actuals_a != 0
    mape = float(np.mean(abs_err[nz] / np.abs(actuals_a[nz])) * 100) if nz.any() \
        else float("nan")
    denom = (np.abs(actuals_a) + np.abs(preds_a))
    dz = denom != 0
    smape = float(np.mean(2 * abs_err[dz] / denom[dz]) * 100) if dz.any() else 0.0
    naive_scale = float(np.mean(np.abs(np.diff(v)))) if n > 1 else 0.0
    mase = float(np.mean(abs_err) / naive_scale) if naive_scale > 0 else float("nan")

    return CVResult(
        model=spec.name, window=window, n_folds=len(preds),
        mae=float(np.mean(abs_err)),
        rmse=float(np.sqrt(np.mean(resid ** 2))),
        mape=mape, smape=smape, mase=mase,
        stability=float(np.std(abs_err)),
        abs_errors=abs_err, residuals=resid)

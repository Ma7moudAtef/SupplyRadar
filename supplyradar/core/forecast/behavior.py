"""Step 3: material behaviour analysis.

Statistics + a behavior classification per series. The classification is
INSIGHT ONLY — it never selects the forecasting model (the competition does).
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

CLASSIFICATIONS = ["Stable", "Trending", "Seasonal", "Highly Variable",
                   "Random", "Intermittent", "Structural Change Detected"]

@dataclass
class BehaviorProfile:
    n_points: int
    mean: float
    median: float
    minimum: float
    maximum: float
    std: float
    cv: float                      # coefficient of variation
    trend_strength: float          # |pearson r| of rate vs time
    trend_slope: float             # rate units per month
    trend_pvalue: float
    seasonality_strength: float    # autocorrelation at lag 12 (0 if unknowable)
    lag1_autocorr: float
    stationary: bool | None        # ADF, None if series too short
    outlier_pct: float
    missing_pct: float
    zero_share: float
    structural_break: bool
    break_month_index: int | None
    classification: str

    def to_dict(self) -> dict:
        return asdict(self)

def analyze_behavior(values: np.ndarray, *, missing_pct: float = 0.0
                     ) -> BehaviorProfile:
    """Profile a monthly rate series (NaN-free; preprocessing fills gaps)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(np.mean(v)) if n else 0.0
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    cv = std / abs(mean) if mean else float("inf") if std else 0.0

    slope, r, p = 0.0, 0.0, 1.0
    detrended = v
    if n >= 5 and std > 0:
        res = stats.linregress(np.arange(n), v)
        slope, r, p = float(res.slope), float(res.rvalue), float(res.pvalue)
        detrended = v - (res.intercept + res.slope * np.arange(n))
        # a pure trend detrends to numerical noise — no cyclic signal in that
        if np.std(detrended) <= 1e-9 * max(1.0, abs(mean)):
            detrended = np.zeros(n)

    # seasonality/autocorrelation on the DETRENDED series, otherwise a trend
    # masquerades as a 12-month cycle
    seasonality = _autocorr(detrended, 12) if n >= 24 else 0.0
    lag1 = _autocorr(detrended, 1) if n >= 4 else 0.0
    stationary = _adf_stationary(v) if n >= 12 and std > 0 else None

    med = float(np.median(v)) if n else 0.0
    mad = float(np.median(np.abs(v - med))) if n else 0.0
    outlier_pct = (float(np.mean(np.abs(v - med) > 3 * 1.4826 * mad))
                   if mad > 0 else 0.0)
    zero_share = float(np.mean(v == 0)) if n else 0.0
    brk, brk_idx = _structural_break(v)

    classification = _classify(cv=cv, trend_p=p, trend_r=r,
                               seasonality=seasonality, zero_share=zero_share,
                               structural_break=brk, n=n)
    return BehaviorProfile(
        n_points=n, mean=mean, median=med, minimum=float(np.min(v)) if n else 0.0,
        maximum=float(np.max(v)) if n else 0.0, std=std, cv=cv,
        trend_strength=abs(r), trend_slope=slope, trend_pvalue=p,
        seasonality_strength=seasonality, lag1_autocorr=lag1,
        stationary=stationary, outlier_pct=outlier_pct,
        missing_pct=missing_pct, zero_share=zero_share,
        structural_break=brk, break_month_index=brk_idx,
        classification=classification)

def _autocorr(v: np.ndarray, lag: int) -> float:
    if len(v) <= lag + 2 or np.std(v) == 0:
        return 0.0
    a, b = v[:-lag], v[lag:]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def _adf_stationary(v: np.ndarray) -> bool | None:
    try:
        from statsmodels.tsa.stattools import adfuller
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bool(adfuller(v, autolag="AIC")[1] < 0.05)
    except Exception:
        return None

def _structural_break(v: np.ndarray, *, min_segment: int = 4,
                      shift_sigmas: float = 2.0) -> tuple[bool, int | None]:
    """Largest mean shift between two segments of the DETRENDED series,
    in pooled-std units — a smooth trend is not a break; a level jump is."""
    n = len(v)
    if n < 2 * min_segment:
        return False, None
    t = np.arange(n)
    resid = v - np.polyval(np.polyfit(t, v, 1), t)
    # residuals of numerical-noise size cannot carry a level shift
    if np.std(resid) <= 1e-9 * max(1.0, float(np.abs(v).mean())):
        return False, None
    best, best_idx = 0.0, None
    for i in range(min_segment, n - min_segment + 1):
        a, b = resid[:i], resid[i:]
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        if pooled == 0:
            continue
        shift = abs(np.mean(a) - np.mean(b)) / pooled
        if shift > best:
            best, best_idx = shift, i
    return (best >= shift_sigmas, best_idx if best >= shift_sigmas else None)

def _classify(*, cv, trend_p, trend_r, seasonality, zero_share,
              structural_break, n) -> str:
    if zero_share >= 0.4:
        return "Intermittent"
    if structural_break:
        return "Structural Change Detected"
    if seasonality >= 0.5:
        return "Seasonal"
    if n >= 6 and trend_p < 0.05 and abs(trend_r) >= 0.5:
        return "Trending"
    if cv <= 0.25:
        return "Stable"
    if cv >= 0.75:
        return "Highly Variable"
    return "Random"

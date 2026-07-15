"""Step 7: automatic model selection.

Ranking considers forecast error (sMAPE/MAE/RMSE mean rank), stability of the
fold errors, and generalization (MASE); when several pipelines land within a
whisker of the best score the SIMPLEST model wins (Occam's razor).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cross_validation import CVResult
from .models import MODEL_LIBRARY

_COMPLEXITY = {m.name: m.complexity for m in MODEL_LIBRARY}
_TIE_TOLERANCE = 0.05  # within 5% of the best composite = a tie

def rank_pipelines(results: list[CVResult]) -> pd.DataFrame:
    """Competition table, best first, with `rank` and `selected` columns."""
    if not results:
        return pd.DataFrame(columns=["model", "window", "n_folds", "mae",
                                     "rmse", "mape", "smape", "mase",
                                     "stability", "complexity", "score",
                                     "rank", "selected"])
    df = pd.DataFrame([{
        "model": r.model, "window": r.window, "n_folds": r.n_folds,
        "mae": r.mae, "rmse": r.rmse, "mape": r.mape, "smape": r.smape,
        "mase": r.mase, "stability": r.stability,
        "complexity": _COMPLEXITY.get(r.model, 9),
    } for r in results])

    # mean of metric ranks; stability half-weighted (error first, then calm)
    ranks = pd.DataFrame({
        "mae": df["mae"].rank(),
        "rmse": df["rmse"].rank(),
        "smape": df["smape"].rank(),
        "stability": df["stability"].rank() * 0.5,
    })
    df["score"] = ranks.sum(axis=1) / 3.5

    best = df["score"].min()
    tied = df.loc[df["score"] <= best * (1 + _TIE_TOLERANCE)]
    winner_idx = tied.sort_values(
        ["complexity", "score", "window"]).index[0]

    df["rank"] = df["score"].rank(method="first").astype(int)
    df["selected"] = False
    df.loc[winner_idx, "selected"] = True
    return df.sort_values("rank").reset_index(drop=True)

def confidence_score(smape: float) -> float:
    """0-100; 100 = perfect rolling validation."""
    if not np.isfinite(smape):
        return 0.0
    return float(max(0.0, 100.0 - smape))

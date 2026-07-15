"""Step 4: adaptive memory detection.

No fixed historical window is assumed: every candidate lookback becomes a
pipeline parameter, and the EFFECTIVE memory of a material is the lookback of
whichever pipeline wins the rolling-validation competition.
"""

from __future__ import annotations

CANDIDATE_WINDOWS_MONTHS = [2, 3, 6, 12, 24]

def feasible_windows(n_points: int, *, min_train: int = 2,
                     heavy: bool = False) -> list[int]:
    """Candidate lookbacks usable for a series of n_points months.

    A window is feasible when at least `min_train` training points fit inside
    it and at least one validation origin remains beyond it. The full history
    (window >= n) is always on the table as an explicit candidate.

    heavy=True (MLE-fitted models: ETS/Holt-Winters/ARIMA/SARIMA) keeps only
    the two longest candidates — short lookbacks starve those models below
    statistical sense while multiplying fit cost in the rolling validation.
    """
    if n_points < min_train + 1:
        return []
    usable = [w for w in CANDIDATE_WINDOWS_MONTHS
              if min_train <= w < n_points]
    full = n_points  # "all history" pseudo-window
    if full not in usable:
        usable.append(full)
    return usable[-2:] if heavy else usable

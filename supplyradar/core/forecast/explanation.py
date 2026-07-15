"""Step 10: the explanation engine.

Every automatic decision must be explainable: why the winner won, why the
closest competitors lost, and what the recommendation is.
"""

from __future__ import annotations

import pandas as pd

from .behavior import BehaviorProfile


def explain_selection(competition: pd.DataFrame, behavior: BehaviorProfile,
                      n_points: int) -> str:
    if competition.empty:
        return "No pipeline could be validated on this history."
    win = competition.loc[competition["selected"]].iloc[0]
    memory = ("all history" if win["window"] >= n_points
              else f"{int(win['window'])} months")

    reasons = [f"effective memory: {memory}",
               "lowest rolling validation error" if win["rank"] == 1
               else "best error/simplicity balance (Occam's razor tie-break)"]
    if behavior.classification == "Trending":
        direction = "upward" if behavior.trend_slope > 0 else "downward"
        reasons.insert(0, f"{direction} trend detected "
                          f"(strength {behavior.trend_strength:.2f})")
    elif behavior.classification == "Seasonal":
        reasons.insert(0, f"seasonal pattern detected "
                          f"(strength {behavior.seasonality_strength:.2f})")
    elif behavior.classification == "Intermittent":
        reasons.insert(0, f"intermittent demand "
                          f"({behavior.zero_share:.0%} zero months)")
    elif behavior.classification == "Stable":
        reasons.insert(0, f"stable behavior (CV {behavior.cv:.2f}), "
                          "no significant trend or seasonality")
    if win["stability"] <= competition["stability"].median():
        reasons.append("stable residual distribution across folds")

    lines = [f"Selected model: {win['model']} "
             f"(lookback {memory}, sMAPE {win['smape']:.1f}%).",
             "Reasons:"]
    lines += [f"  • {r}" for r in reasons]

    rejected = competition.loc[~competition["selected"]].head(2)
    for _, row in rejected.iterrows():
        why = []
        if row["smape"] > win["smape"]:
            why.append(f"higher validation error "
                       f"(sMAPE {row['smape']:.1f}% vs {win['smape']:.1f}%)")
        if row["stability"] > win["stability"]:
            why.append("less stable fold errors")
        if row["complexity"] > win["complexity"] and not why:
            why.append("equal accuracy at higher complexity")
        if not why:
            why.append("ranked below on the error/stability composite")
        lines.append(f"{row['model']} rejected: " + "; ".join(why) + ".")
    return "\n".join(lines)

def recommend(behavior: BehaviorProfile, smape: float,
              missing_pct: float) -> str:
    if behavior.structural_break:
        return ("Manual review recommended — structural behavior change "
                "detected; recent history may not represent the future.")
    if missing_pct > 0.3:
        return "Data quality issue detected — many missing periods."
    if not (smape == smape) or smape > 45:  # NaN or high error
        return ("Manual review recommended — forecast confidence is low "
                "for this material.")
    return "Automatic forecast recommended."

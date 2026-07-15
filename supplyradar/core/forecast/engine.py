"""Steps 8-9-11 + orchestration: forecast generation, confidence intervals,
and reconstitution of material demand from forecast rates.

run_item_forecast() is the per-material entry point (Step 12's dashboard is a
thin view over its output). forecast_projected_consumption() turns forecast
rates into daily plan-consumption rows so the stock projection can run on the
forecast instead of the standard rates — the planner's "switch".

Continuous learning, v1 scope: every new workbook upload re-runs the
competition on the extended history; the latest validation folds double as the
forecast-vs-actual backtest readout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..loader import WorkbookData
from ..prep_consumption import CONSUMPTION_COLUMNS, check_base_date, horizon_end_for
from ..uom import convert, normalize_uom
from .behavior import BehaviorProfile, analyze_behavior
from .cross_validation import CVResult, rolling_validate
from .explanation import explain_selection, recommend
from .memory import feasible_windows
from .models import MODEL_LIBRARY, applicable_models
from .preprocessing import RateSeries, build_rate_series
from .selector import confidence_score, rank_pipelines

_SPEC_BY_NAME = {m.name: m for m in MODEL_LIBRARY}

@dataclass
class ComboForecast:
    """Forecast result for one (item, output_type, production_line) series."""

    item_code: str
    output_type: str
    production_line: str
    rate_uom: str
    history: pd.Series                    # PeriodIndex -> actual monthly rate
    behavior: BehaviorProfile
    competition: pd.DataFrame             # ranked pipelines
    selected_model: str
    selected_window: int
    effective_memory: str
    forecast: pd.DataFrame                # month, rate, lo80, hi80, lo95, hi95
    explanation: str
    recommendation: str
    confidence: float                     # 0-100
    data_quality: float                   # 0-100
    forecastability: float                # 0-100
    flags: list[str] = field(default_factory=list)
    is_override: bool = False

@dataclass
class ItemForecast:
    item_code: str
    horizon_months: int
    combos: list[ComboForecast]
    generated_at: str

    def combined_smape(self) -> float:
        vals = [c.competition.loc[c.competition["selected"], "smape"].iloc[0]
                for c in self.combos if not c.competition.empty]
        return float(np.mean(vals)) if vals else float("nan")

def run_item_forecast(
    data: WorkbookData,
    item_code: str,
    horizon_months: int,
    *,
    min_history: int = 3,
    override_model: str | None = None,
    override_window: int | None = None,
) -> ItemForecast:
    """Compete pipelines and forecast every combo of one item.

    Overrides (Step 12 manual control) pin the model and/or lookback window;
    the competition still runs so the planner sees what automation would do.
    """
    all_series = build_rate_series(data, items=[item_code])
    combos: list[ComboForecast] = []
    for key, rs in sorted(all_series.items()):
        if rs.n_valid < min_history or float(np.nansum(np.abs(rs.values))) == 0:
            continue
        combos.append(_forecast_combo(rs, horizon_months,
                                      override_model=override_model,
                                      override_window=override_window))
    return ItemForecast(item_code=item_code, horizon_months=horizon_months,
                        combos=combos,
                        generated_at=datetime.now().isoformat(timespec="seconds"))

def run_all_items(data: WorkbookData, horizon_months: int,
                  *, progress=None) -> pd.DataFrame:
    """Fleet overview: one row per item-combo with the winning pipeline."""
    items = data.stock["item_code"].dropna().unique().tolist()
    rows = []
    for i, item in enumerate(items):
        fc = run_item_forecast(data, item, horizon_months)
        for c in fc.combos:
            rows.append({
                "item_code": c.item_code, "output_type": c.output_type,
                "production_line": c.production_line,
                "classification": c.behavior.classification,
                "model": c.selected_model, "memory": c.effective_memory,
                "smape": round(_selected_metric(c, "smape"), 1),
                "mae": _selected_metric(c, "mae"),
                "confidence": round(c.confidence),
                "recommendation": c.recommendation,
            })
        if progress is not None:
            progress((i + 1) / len(items))
    return pd.DataFrame(rows)

def forecast_projected_consumption(
    data: WorkbookData,
    forecasts: dict[str, ItemForecast],
    base_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Daily consumption rows driven by FORECAST rates (consumption_type=
    'forecast'), for the items in `forecasts`. Combos an item forecast does
    not cover keep contributing via the standard plan projection — the caller
    stitches the two.

    Same three rate branches as the standard projection:
      kg/ton  -> rate x planned qty1 / 1000
      pc/heat -> rate x planned qty2
      ton/day -> rate flat per calendar day
    Months beyond the forecast horizon hold the last forecast rate.
    """
    base_date = pd.Timestamp(base_date)
    check_base_date(data, base_date)
    horizon_end = horizon_end_for(data)
    dates = pd.date_range(base_date, horizon_end, freq="D")
    all_months = pd.period_range(dates[0].to_period("M"),
                                 dates[-1].to_period("M"), freq="M")

    plan = data.prod.loc[
        (data.prod["production_type"] == "plan")
        & data.prod["date"].between(base_date, horizon_end)]
    plan_grid = (plan.groupby(["date", "output_type", "production_line"],
                              as_index=False)
                 [["production_qty1", "production_qty2"]].sum())
    plan_grid["month"] = plan_grid["date"].dt.to_period("M")
    bom_cols = data.bom.set_index("item_code")[["uom", "unit_wt_kg",
                                                "unit_price_$"]]

    frames = []
    for item, fc in forecasts.items():
        for c in fc.combos:
            rates = c.forecast.set_index("month")["rate"].reindex(
                all_months).ffill().bfill()
            if c.rate_uom == "ton/day":
                df = pd.DataFrame({"date": dates})
                df["month"] = df["date"].dt.to_period("M")
                df["natural_qty"] = df["month"].map(rates).astype(float)
                df["natural_uom"] = "ton"
            else:
                df = plan_grid.loc[
                    (plan_grid["output_type"] == c.output_type)
                    & (plan_grid["production_line"] == c.production_line)
                ].copy()
                if df.empty:
                    continue
                rate = df["month"].map(rates).astype(float)
                if c.rate_uom == "kg/ton":
                    df["natural_qty"] = rate * df["production_qty1"] / 1000.0
                    df["natural_uom"] = "ton"
                else:  # pc/heat
                    df["natural_qty"] = rate * df["production_qty2"]
                    df["natural_uom"] = "pc"
            df["item_code"] = item
            df["output_type"] = c.output_type
            df["production_line"] = c.production_line
            frames.append(df[["date", "item_code", "output_type",
                              "production_line", "natural_qty", "natural_uom"]])

    if not frames:
        return pd.DataFrame(columns=CONSUMPTION_COLUMNS)
    out = pd.concat(frames, ignore_index=True).join(bom_cols, on="item_code")
    out["cons_qty_ton"] = _convert(out, "ton")
    out["cons_qty_base_uom"] = _convert(out, None)
    out["cons_$"] = out["cons_qty_base_uom"] * out["unit_price_$"].fillna(0.0)
    out["consumption_type"] = "forecast"
    return out[CONSUMPTION_COLUMNS].sort_values(
        ["item_code", "date"]).reset_index(drop=True)

def expected_monthly_consumption(
    data: WorkbookData, combo: ComboForecast, base_date: str | pd.Timestamp
) -> pd.DataFrame:
    """Forecast Consumption = forecast rate x planned production, monthly,
    for one combo. Columns: month, qty (item's bom uom), uom."""
    base_date = pd.Timestamp(base_date)
    plan = data.prod.loc[
        (data.prod["production_type"] == "plan")
        & (data.prod["output_type"] == combo.output_type)
        & (data.prod["production_line"] == combo.production_line)
        & (data.prod["date"] >= base_date)].copy()
    bom = data.bom.set_index("item_code")
    uom = bom.loc[combo.item_code, "uom"]
    wt = bom.loc[combo.item_code, "unit_wt_kg"]

    plan["month"] = plan["date"].dt.to_period("M")
    driver = plan.groupby("month")[["production_qty1", "production_qty2"]].sum()
    driver["days"] = driver.index.days_in_month.astype(float)
    months = combo.forecast.set_index("month")
    joined = driver.join(months["rate"], how="inner")
    if joined.empty:
        return pd.DataFrame(columns=["month", "qty", "uom"])

    if combo.rate_uom == "kg/ton":
        natural = joined["rate"] * joined["production_qty1"] / 1000.0
        natural_uom = "ton"
    elif combo.rate_uom == "pc/heat":
        natural = joined["rate"] * joined["production_qty2"]
        natural_uom = "pc"
    else:  # ton/day
        natural = joined["rate"] * joined["days"]
        natural_uom = "ton"
    qty = convert(natural.to_numpy(dtype=float), natural_uom, uom,
                  unit_wt_kg=wt if pd.notna(wt) else None) \
        if normalize_uom(natural_uom) != normalize_uom(uom) else natural.to_numpy()
    return pd.DataFrame({"month": joined.index, "qty": qty, "uom": uom})

# ------------------------------------------------------------------ internals

def _forecast_combo(rs: RateSeries, horizon: int, *,
                    override_model: str | None,
                    override_window: int | None) -> ComboForecast:
    values = rs.values
    n = len(values)
    behavior = analyze_behavior(values, missing_pct=_missing_pct(rs))
    intermittent = behavior.classification == "Intermittent"

    results: list[CVResult] = []
    for spec in applicable_models(n, intermittent=intermittent):
        for window in feasible_windows(n, heavy=spec.heavy):
            res = rolling_validate(values, spec, window)
            if res is not None:
                results.append(res)
    competition = rank_pipelines(results)

    if override_model or override_window:
        competition = _apply_override(competition, override_model,
                                      override_window)
    if competition.empty:
        # not enough usable folds anywhere: fall back to naive over everything
        sel_model, sel_window = "Naive", n
        residuals = np.array([np.std(values)]) if n > 1 else np.array([0.0])
        smape = float("nan")
    else:
        win = competition.loc[competition["selected"]].iloc[0]
        sel_model, sel_window = str(win["model"]), int(win["window"])
        winner = next(r for r in results
                      if r.model == sel_model and r.window == sel_window)
        residuals = winner.abs_errors
        smape = float(win["smape"])

    spec = _SPEC_BY_NAME[sel_model]
    train = values[-sel_window:]
    point = spec.fit_predict(train, horizon)
    if point is None:
        point = np.repeat(train[-1] if len(train) else 0.0, horizon)

    q80 = float(np.quantile(residuals, 0.80)) if len(residuals) else 0.0
    q95 = float(np.quantile(residuals, 0.95)) if len(residuals) else 0.0
    months = pd.period_range(rs.series.index[-1] + 1, periods=horizon, freq="M")
    forecast = pd.DataFrame({
        "month": months, "rate": point,
        "lo80": np.maximum(point - q80, 0.0), "hi80": point + q80,
        "lo95": np.maximum(point - q95, 0.0), "hi95": point + q95,
    })

    memory = "all history" if sel_window >= n else f"{sel_window} months"
    quality = float(max(0.0, 100.0 * (1 - behavior.missing_pct
                                      - behavior.outlier_pct)))
    cv_pen = min(behavior.cv, 1.0) if np.isfinite(behavior.cv) else 1.0
    conf = confidence_score(smape)
    return ComboForecast(
        item_code=rs.item_code, output_type=rs.output_type,
        production_line=rs.production_line, rate_uom=rs.rate_uom,
        history=rs.series, behavior=behavior, competition=competition,
        selected_model=sel_model, selected_window=sel_window,
        effective_memory=memory, forecast=forecast,
        explanation=explain_selection(competition, behavior, n),
        recommendation=recommend(behavior, smape, behavior.missing_pct),
        confidence=conf,
        data_quality=quality,
        forecastability=float(max(0.0, 0.5 * conf + 50.0 * (1 - cv_pen))),
        flags=list(rs.flags),
        is_override=bool(override_model or override_window))

def _apply_override(competition: pd.DataFrame, model: str | None,
                    window: int | None) -> pd.DataFrame:
    df = competition.copy()
    mask = pd.Series(True, index=df.index)
    if model:
        mask &= df["model"] == model
    if window:
        mask &= df["window"] == window
    if mask.any():
        df["selected"] = False
        df.loc[df.loc[mask, "score"].idxmin(), "selected"] = True
    return df

def _selected_metric(c: ComboForecast, metric: str) -> float:
    if c.competition.empty:
        return float("nan")
    return float(c.competition.loc[c.competition["selected"], metric].iloc[0])

def _missing_pct(rs: RateSeries) -> float:
    for f in rs.flags:
        if f.startswith("missing_periods"):
            return int(f.split(":")[1]) / max(len(rs.series), 1)
    return 0.0

def _convert(df: pd.DataFrame, to: str | None) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    to_series = df["uom"] if to is None else pd.Series(to, index=df.index)
    pairs = pd.DataFrame({"f": df["natural_uom"].map(normalize_uom),
                          "t": to_series.map(normalize_uom)})
    for (f, t), grp in pairs.groupby(["f", "t"]):
        idx = grp.index
        wt = (df.loc[idx, "unit_wt_kg"].to_numpy(dtype=float)
              if ("pc" in (f, t)) and f != t else None)
        out.loc[idx] = convert(df.loc[idx, "natural_qty"].to_numpy(dtype=float),
                               f, t, unit_wt_kg=wt)
    return out

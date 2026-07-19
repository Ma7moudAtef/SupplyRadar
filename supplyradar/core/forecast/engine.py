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
from .preprocessing import RateSeries, build_rate_series, flag_rate_series
from .selector import confidence_score, rank_pipelines

_SPEC_BY_NAME = {m.name: m for m in MODEL_LIBRARY}

@dataclass
class TopForecast:
    """One of the top-ranked competing pipelines and its point forecast."""

    rank: int
    model: str
    window: int
    memory: str
    smape: float
    forecast: pd.DataFrame                # month, rate

ALL = "all"  # collapsed-axis label: this group aggregates every value of the axis

@dataclass
class ComboForecast:
    """Forecast result for one production-stream GROUP of an item.

    A group is one or more (output_type, production_line) sub-combos aggregated
    into a single rate series (total consumption / total production). When the
    planner breaks an axis out fully, each group is a single sub-combo; when an
    axis is left combined, `output_type`/`production_line` read 'all' and
    `covers` lists every sub-combo the group aggregates.
    """

    item_code: str
    output_type: str                      # a value, or 'all' when combined
    production_line: str                   # a value, or 'all' when combined
    covers: list[tuple[str, str]]          # the (output_type, line) sub-combos
    label: str                             # human label for the stream group
    rate_uom: str
    history: pd.Series                    # PeriodIndex -> actual monthly rate
    consumption: pd.Series                # PeriodIndex -> actual monthly consumption
    behavior: BehaviorProfile
    competition: pd.DataFrame             # ranked pipelines
    selected_model: str
    selected_window: int
    effective_memory: str
    forecast: pd.DataFrame                # month, rate, lo80, hi80, lo95, hi95
    top_forecasts: list[TopForecast]      # top 3 competitors, best first
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

    @property
    def covered_subcombos(self) -> set[tuple[str, str]]:
        """Every (output_type, line) sub-combo this forecast drives — used to
        decide which plan rows the projection switch replaces."""
        return {sub for c in self.combos for sub in c.covers}

def run_item_forecast(
    data: WorkbookData,
    item_code: str,
    horizon_months: int,
    *,
    by_output: tuple[str, ...] | None = None,
    by_line: tuple[str, ...] | None = None,
    min_history: int = 3,
    override_model: str | None = None,
    override_window: int | None = None,
) -> ItemForecast:
    """Compete pipelines and forecast each production-stream group of one item.

    Grouping (the planner's 'tree of choices'):
      None            -> break the axis out fully (one group per value present)
      () empty tuple  -> combine the axis (one group aggregating every value)
      (v1, v2, ...)   -> break out only these values
    So `by_output=('b',), by_line=()` yields ONE forecast for output type B with
    its lines aggregated; `by_output=('b',), by_line=('1','2')` yields two.

    Overrides pin the model and/or lookback window; the competition still runs
    so the planner sees what automation would do.
    """
    all_series = build_rate_series(data, items=[item_code])
    present = sorted(all_series.keys())  # (item, output_type, line) tuples
    out_branches = _branches(sorted({k[1] for k in present}), by_output)
    line_branches = _branches(sorted({k[2] for k in present}), by_line)

    combos: list[ComboForecast] = []
    for o_label, o_vals in out_branches:
        for l_label, l_vals in line_branches:
            members = [all_series[k] for k in present
                       if k[1] in o_vals and k[2] in l_vals]
            if not members:
                continue
            covers = [(k[1], k[2]) for k in present
                      if k[1] in o_vals and k[2] in l_vals]
            rs = _aggregate_members(members, item_code, o_label, l_label)
            if rs.n_valid < min_history or float(np.nansum(np.abs(rs.values))) == 0:
                continue
            combos.append(_forecast_combo(
                rs, horizon_months, covers=covers,
                label=_group_label(o_label, l_label),
                override_model=override_model, override_window=override_window))
    return ItemForecast(item_code=item_code, horizon_months=horizon_months,
                        combos=combos,
                        generated_at=datetime.now().isoformat(timespec="seconds"))

def _branches(present_values: list[str], selection: tuple[str, ...] | None
              ) -> list[tuple[str, frozenset[str]]]:
    """Turn an axis selection into (label, covered-values) branches.

    None -> one branch per present value (full breakout); () -> one combined
    branch over all present values; a tuple -> one branch per chosen value that
    is actually present.
    """
    if selection is None:
        return [(v, frozenset({v})) for v in present_values]
    if len(selection) == 0:
        return [(ALL, frozenset(present_values))] if present_values else []
    chosen = [v for v in selection if v in present_values]
    return [(v, frozenset({v})) for v in chosen]

def _group_label(o_label: str, l_label: str) -> str:
    o = "all types" if o_label == ALL else o_label.upper()
    ln = "all lines" if l_label == ALL else f"line {l_label}"
    return f"{o} / {ln}"

def _aggregate_members(members: list[RateSeries], item_code: str,
                       o_label: str, l_label: str) -> RateSeries:
    """Aggregate sub-combo rate series into one: the rate is total consumption
    over total production, i.e. a production-weighted mean of the member rates
    (num = sum rate_i*prod_i, den = sum prod_i). All members share a rate uom
    (verified: each item has a single std_cons_rate_uom)."""
    if len(members) == 1:
        m = members[0]
        return RateSeries(item_code=item_code, output_type=o_label,
                          production_line=l_label, rate_uom=m.rate_uom,
                          series=m.series, consumption=m.consumption,
                          production=m.production, flags=list(m.flags))
    months = members[0].series.index
    for m in members[1:]:
        months = months.union(m.series.index)
    num = pd.Series(0.0, index=months)
    den = pd.Series(0.0, index=months)
    cons = pd.Series(0.0, index=months)
    for m in members:
        r = m.series.reindex(months).fillna(0.0)
        p = m.production.reindex(months).fillna(0.0)
        num = num + r * p
        den = den + p
        cons = cons + m.consumption.reindex(months).fillna(0.0)
    rate = pd.Series(np.where(den > 0, num / den.replace(0, np.nan), 0.0),
                     index=months).fillna(0.0)
    return RateSeries(item_code=item_code, output_type=o_label,
                      production_line=l_label, rate_uom=members[0].rate_uom,
                      series=rate, consumption=cons, production=den,
                      flags=flag_rate_series(rate))

def run_all_items(data: WorkbookData, horizon_months: int,
                  *, items: list[str] | None = None, progress=None) -> pd.DataFrame:
    """Fleet overview: one row per item-combo with the winning pipeline.

    `items` restricts the competition to a chosen subset of materials (a hand
    pick or a category group); None runs every stock item.
    """
    if items is None:
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
                # flat tons/day: the group rate is the total already, so it is
                # emitted once (not once per covered sub-combo)
                df = pd.DataFrame({"date": dates})
                df["month"] = df["date"].dt.to_period("M")
                df["natural_qty"] = df["month"].map(rates).astype(float)
                df["natural_uom"] = "ton"
                df["item_code"] = item
                df["output_type"] = c.output_type
                df["production_line"] = c.production_line
                frames.append(df[["date", "item_code", "output_type",
                                  "production_line", "natural_qty", "natural_uom"]])
                continue
            # kg/ton and pc/heat: apply the group's rate to EACH covered
            # sub-combo's planned production (so an aggregated B rate lands on
            # B line 1 and B line 2)
            for o_sub, l_sub in c.covers:
                df = plan_grid.loc[(plan_grid["output_type"] == o_sub)
                                   & (plan_grid["production_line"] == l_sub)].copy()
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
                df["output_type"] = o_sub
                df["production_line"] = l_sub
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

def build_comparison_table(combo: ComboForecast) -> tuple[pd.DataFrame, list[str]]:
    """History + the top-3 competing forecasts, stacked by month.

    Historical months carry the actual rate and consumption; future months
    carry each of the top-3 pipelines' forecast rate. Returns (frame,
    forecast_column_names) so the UI can add an editable 'Your projection'
    column and lock the rest. Rate is the forecast variable throughout.
    """
    hist = pd.DataFrame({
        "Month": combo.history.index.astype(str),
        "Actual rate": combo.history.to_numpy(dtype=float),
        "Actual consumption": combo.consumption.to_numpy(dtype=float),
    })
    fc_cols: list[str] = []
    fut = pd.DataFrame({"Month": combo.forecast["month"].astype(str)})
    for t in combo.top_forecasts:
        col = f"{t.rank}. {t.model} ({t.memory})"
        fc_cols.append(col)
        fut[col] = t.forecast["rate"].to_numpy(dtype=float)
    table = pd.concat([hist, fut], ignore_index=True)
    for col in ["Actual rate", "Actual consumption", *fc_cols]:
        if col not in table.columns:
            table[col] = np.nan
    ordered = ["Month", "Actual rate", "Actual consumption", *fc_cols]
    return table[ordered], fc_cols

def expected_monthly_consumption(
    data: WorkbookData, combo: ComboForecast, base_date: str | pd.Timestamp
) -> pd.DataFrame:
    """Forecast Consumption = forecast rate x planned production, monthly,
    summed across the group's covered sub-combos. Columns: month, qty (item's
    bom uom), uom."""
    base_date = pd.Timestamp(base_date)
    covers = set(combo.covers)
    combo_plan = data.prod.loc[
        (data.prod["production_type"] == "plan")
        & (data.prod["date"] >= base_date)].copy()
    combo_plan = combo_plan.loc[[
        (o, ln) in covers for o, ln in
        zip(combo_plan["output_type"], combo_plan["production_line"])]]
    plan = combo_plan
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
                    covers: list[tuple[str, str]], label: str,
                    override_model: str | None,
                    override_window: int | None) -> ComboForecast:
    values = rs.values
    weights = rs.weights
    n = len(values)
    behavior = analyze_behavior(values, missing_pct=_missing_pct(rs))
    intermittent = behavior.classification == "Intermittent"

    results: list[CVResult] = []
    for spec in applicable_models(n, intermittent=intermittent):
        for window in feasible_windows(n, heavy=spec.heavy):
            res = rolling_validate(values, spec, window, weights=weights)
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

    months = pd.period_range(rs.series.index[-1] + 1, periods=horizon, freq="M")
    point = _point_forecast(values, weights, sel_model, sel_window, horizon)

    q80 = float(np.quantile(residuals, 0.80)) if len(residuals) else 0.0
    q95 = float(np.quantile(residuals, 0.95)) if len(residuals) else 0.0
    forecast = pd.DataFrame({
        "month": months, "rate": point,
        "lo80": np.maximum(point - q80, 0.0), "hi80": point + q80,
        "lo95": np.maximum(point - q95, 0.0), "hi95": point + q95,
    })

    top_forecasts = _top_forecasts(competition, values, weights, n, horizon, months)

    memory = "all history" if sel_window >= n else f"{sel_window} months"
    quality = float(max(0.0, 100.0 * (1 - behavior.missing_pct
                                      - behavior.outlier_pct)))
    cv_pen = min(behavior.cv, 1.0) if np.isfinite(behavior.cv) else 1.0
    conf = confidence_score(smape)
    return ComboForecast(
        item_code=rs.item_code, output_type=rs.output_type,
        production_line=rs.production_line, covers=covers, label=label,
        rate_uom=rs.rate_uom,
        history=rs.series, consumption=rs.consumption, behavior=behavior,
        competition=competition,
        selected_model=sel_model, selected_window=sel_window,
        effective_memory=memory, forecast=forecast,
        top_forecasts=top_forecasts,
        explanation=explain_selection(competition, behavior, n),
        recommendation=recommend(behavior, smape, behavior.missing_pct),
        confidence=conf,
        data_quality=quality,
        forecastability=float(max(0.0, 0.5 * conf + 50.0 * (1 - cv_pen))),
        flags=list(rs.flags),
        is_override=bool(override_model or override_window))

def _point_forecast(values: np.ndarray, weights: np.ndarray | None,
                    model: str, window: int, horizon: int) -> np.ndarray:
    """Fit `model` on the last `window` months and forecast `horizon` ahead."""
    spec = _SPEC_BY_NAME[model]
    train = values[-window:]
    w = None if weights is None else weights[-window:]
    point = spec.fit_predict(train, horizon, weights=w)
    if point is None:
        point = np.repeat(train[-1] if len(train) else 0.0, horizon)
    return np.asarray(point, dtype=float)

def _top_forecasts(competition: pd.DataFrame, values: np.ndarray,
                   weights: np.ndarray | None, n: int, horizon: int,
                   months: pd.PeriodIndex) -> list[TopForecast]:
    """Point forecast for each of the top-3 ranked pipelines, best first."""
    if competition.empty:
        return []
    out: list[TopForecast] = []
    for _, row in competition.sort_values("rank").head(3).iterrows():
        window = int(row["window"])
        point = _point_forecast(values, weights, str(row["model"]), window, horizon)
        out.append(TopForecast(
            rank=int(row["rank"]), model=str(row["model"]), window=window,
            memory="all history" if window >= n else f"{window} months",
            smape=float(row["smape"]),
            forecast=pd.DataFrame({"month": months, "rate": point})))
    return out

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
        if f.startswith("missing_periods") and ":" in f:
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

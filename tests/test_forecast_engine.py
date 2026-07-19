"""AMCIP forecast engine tests — synthetic series only, never the real workbook."""

import numpy as np
import pandas as pd
import pytest

from supplyradar.core.forecast.cross_validation import rolling_validate
from supplyradar.core.forecast.engine import forecast_projected_consumption, run_item_forecast
from supplyradar.core.forecast.memory import feasible_windows
from supplyradar.core.forecast.models import MODEL_LIBRARY, ModelSpec
from supplyradar.core.forecast.preprocessing import build_rate_series
from supplyradar.core.forecast.selector import rank_pipelines
from supplyradar.core.prep_stock import build_stock_projection

_SPEC = {m.name: m for m in MODEL_LIBRARY}


# ------------------------------------------------------------- preprocessing

def test_rate_series_sums_netting_rows_and_backfills_uom(data):
    # two rows in the same month NET (5.0 - 1.0); cons_rate_uom missing on one
    extra = data.consumption.iloc[[0]].assign(
        date=pd.Timestamp("2025-05-01"), cons_rate=-1.0, cons_rate_uom=None)
    data.consumption = pd.concat([data.consumption, extra], ignore_index=True)
    series = build_rate_series(data, items=["it-1"])
    rs = series[("it-1", "f", "1")]
    assert rs.rate_uom == "kg/ton"
    assert rs.series[pd.Period("2025-05", "M")] == pytest.approx(2.0 - 1.0)
    assert rs.series[pd.Period("2025-06", "M")] == pytest.approx(2.2)


def test_rate_series_derives_when_columns_absent(data):
    # old workbooks: no cons_rate columns -> derive from monthly actual prod
    data.consumption = data.consumption.drop(
        columns=["cons_rate", "cons_rate_uom"])
    data.prod = pd.concat([data.prod, pd.DataFrame({
        "date": pd.date_range("2025-05-01", periods=8, freq="MS"),
        "production_uom1": ["ms"] * 8, "production_qty1": [30000.0] * 8,
        "production_uom2": ["heat"] * 8, "production_qty2": [150.0] * 8,
        "output_type": ["f"] * 8, "production_line": ["1"] * 8,
        "production_type": ["actual"] * 8,
    })], ignore_index=True)
    series = build_rate_series(data, items=["it-1"])
    rs = series[("it-1", "f", "1")]
    # cons_qty_ton = rate*30 tons; derived rate = ton*1000/qty1 = rate*30*1000/30000
    assert rs.series[pd.Period("2025-05", "M")] == pytest.approx(2.0)


# ------------------------------------------------------------- model library

def test_simple_models_hand_computed():
    train = np.array([1.0, 2.0, 3.0, 4.0])
    assert _SPEC["Naive"].fit_predict(train, 2).tolist() == [4.0, 4.0]
    assert _SPEC["Moving Average"].fit_predict(train, 1)[0] == pytest.approx(3.0)
    wma = _SPEC["Weighted Moving Average"].fit_predict(train, 1)[0]
    assert wma == pytest.approx((1 * 1 + 2 * 2 + 3 * 3 + 4 * 4) / 10)


def test_weighted_moving_average_uses_production_weights():
    train = np.array([1.0, 2.0, 3.0, 4.0])
    # production quantities weight the rate: the last month produced most
    weights = np.array([10.0, 10.0, 10.0, 100.0])
    out = _SPEC["Weighted Moving Average"].fit_predict(train, 1, weights=weights)[0]
    assert out == pytest.approx((1 * 10 + 2 * 10 + 3 * 10 + 4 * 100) / 130)
    # zero / missing production weights fall back to recency weighting
    zero = _SPEC["Weighted Moving Average"].fit_predict(
        train, 1, weights=np.zeros(4))[0]
    assert zero == pytest.approx((1 * 1 + 2 * 2 + 3 * 3 + 4 * 4) / 10)


def test_seasonal_naive_repeats_last_season():
    train = np.arange(1.0, 14.0)  # 13 points
    out = _SPEC["Seasonal Naive"].fit_predict(train, 3)
    assert out.tolist() == [2.0, 3.0, 4.0]


def test_croston_flat_positive_on_intermittent():
    train = np.array([0, 0, 6.0, 0, 0, 0, 6.0, 0, 0, 6.0])
    out = _SPEC["Croston (Intermittent)"].fit_predict(train, 4)
    assert len(set(np.round(out, 9))) == 1  # flat
    assert 0 < out[0] < 6.0                 # size spread over the interval


def test_forecasts_never_negative():
    train = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
    out = _SPEC["Holt Linear Trend"].fit_predict(train, 6)
    assert out is not None and (out >= 0).all()


# ------------------------------------------------------------- validation

def test_rolling_cv_is_chronological_and_respects_window():
    seen: list[tuple] = []

    def probe(train, h):
        seen.append(tuple(train))
        return np.repeat(train[-1], h)

    spec = ModelSpec("probe", 1, 1, probe)
    values = np.arange(10.0)
    res = rolling_validate(values, spec, window=3, min_train=2, max_folds=5)
    assert res is not None and res.n_folds == 5
    for train in seen:
        assert len(train) <= 3                       # window respected
    # each fold trains on the values immediately BEFORE the predicted one
    assert seen[-1] == (6.0, 7.0, 8.0)               # predicting index 9


def test_feasible_windows_heavy_keeps_long_lookbacks():
    assert feasible_windows(30) == [2, 3, 6, 12, 24, 30]
    assert feasible_windows(30, heavy=True) == [24, 30]
    assert feasible_windows(2) == []


# ------------------------------------------------------------- selection

def test_tie_selects_simplest_model():
    from supplyradar.core.forecast.cross_validation import CVResult

    def cv(model, smape):
        return CVResult(model=model, window=6, n_folds=5, mae=1.0, rmse=1.0,
                        mape=smape, smape=smape, mase=1.0, stability=0.1,
                        abs_errors=np.ones(5), residuals=np.ones(5))

    # identical scores: complexity must break the tie
    table = rank_pipelines([cv("ARIMA(1,1,1)", 10.0), cv("Naive", 10.0)])
    assert table.loc[table["selected"], "model"].iloc[0] == "Naive"


# ------------------------------------------------------------- engine e2e

def _trend_workbook(data):
    """24 months of cleanly rising kg/ton rate for it-1."""
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    rates = 2.0 + 0.05 * np.arange(24)
    data.consumption = pd.DataFrame({
        "date": months, "item_code": "it-1",
        "cons_qty_base_uom": rates * 30, "cons_qty_ton": rates * 30,
        "cons_$": 100.0, "output_type": "f", "production_line": "1",
        "consumption_type": "actual",
        "cons_rate": rates, "cons_rate_uom": "kg/ton",
    })
    return data


def test_engine_end_to_end_on_trending_series(data):
    data = _trend_workbook(data)
    fc = run_item_forecast(data, "it-1", 6)
    assert len(fc.combos) == 1
    c = fc.combos[0]
    assert c.behavior.classification == "Trending"
    assert not c.competition.empty
    assert c.competition["selected"].sum() == 1
    assert len(c.forecast) == 6
    # intervals are ordered and the point forecast sits inside them
    f = c.forecast
    assert (f["lo95"] <= f["lo80"]).all()
    assert (f["lo80"] <= f["rate"]).all()
    assert (f["rate"] <= f["hi80"]).all()
    assert (f["hi80"] <= f["hi95"]).all()
    # a clean upward trend must forecast above the historical mean
    assert f["rate"].iloc[0] > c.history.mean()
    assert "Selected model" in c.explanation
    assert "rejected" in c.explanation


def test_manual_override_pins_model(data):
    data = _trend_workbook(data)
    fc = run_item_forecast(data, "it-1", 3, override_model="Naive")
    c = fc.combos[0]
    assert c.selected_model == "Naive"
    assert c.is_override
    # naive continues the last value flat
    assert c.forecast["rate"].tolist() == pytest.approx(
        [c.history.iloc[-1]] * 3)


def _two_line_workbook(data):
    """it-1 consumed on (f,1) at rate 2.0 and (f,2) at rate 4.0 (kg/ton),
    with line 2 producing 3x the tonnage of line 1."""
    months = pd.date_range("2024-01-01", periods=14, freq="MS")
    cons = []
    for ln, rate in [("1", 2.0), ("2", 4.0)]:
        cons.append(pd.DataFrame({
            "date": months, "item_code": "it-1",
            "cons_qty_base_uom": 0.0, "cons_qty_ton": 0.0, "cons_$": 0.0,
            "output_type": "f", "production_line": ln,
            "consumption_type": "actual",
            "cons_rate": rate, "cons_rate_uom": "kg/ton"}))
    data.consumption = pd.concat(cons, ignore_index=True)
    prod = []
    for ln, q in [("1", 1000.0), ("2", 3000.0)]:
        prod.append(pd.DataFrame({
            "date": months, "production_uom1": "ms", "production_qty1": q,
            "production_uom2": "heat", "production_qty2": q / 100.0,
            "output_type": "f", "production_line": ln,
            "production_type": "actual"}))
    data.prod = pd.concat([data.prod] + prod, ignore_index=True)
    return data


def test_grouping_tree_of_choices(data):
    data = _two_line_workbook(data)
    # break out both lines -> 2 forecasts
    finest = run_item_forecast(data, "it-1", 3, by_output=("f",),
                               by_line=("1", "2"))
    assert len(finest.combos) == 2
    # combine lines (line axis empty) -> ONE forecast covering both lines
    combined = run_item_forecast(data, "it-1", 3, by_output=("f",), by_line=())
    assert len(combined.combos) == 1
    c = combined.combos[0]
    assert set(c.covers) == {("f", "1"), ("f", "2")}
    assert c.label == "F / all lines"
    # production-weighted aggregate rate: (2*1000 + 4*3000)/(1000+3000) = 3.5
    assert c.history.iloc[-1] == pytest.approx(3.5)
    assert combined.covered_subcombos == {("f", "1"), ("f", "2")}


def test_aggregated_switch_lands_on_every_covered_stream(data):
    data = _two_line_workbook(data)
    # give it-1 a plan so the projection has production to apply the rate to
    plan_months = pd.date_range("2026-01-01", periods=1, freq="MS")
    data.prod = pd.concat([data.prod, pd.DataFrame({
        "date": list(plan_months) * 2,
        "production_uom1": "ms", "production_qty1": [1000.0, 3000.0],
        "production_uom2": "heat", "production_qty2": [10.0, 30.0],
        "output_type": "f", "production_line": ["1", "2"],
        "production_type": "plan"})], ignore_index=True)
    fc = run_item_forecast(data, "it-1", 3, by_output=("f",), by_line=())
    rows = forecast_projected_consumption(data, {"it-1": fc}, "2026-01-01")
    covered = set(zip(rows["output_type"], rows["production_line"]))
    assert covered == {("f", "1"), ("f", "2")}  # aggregate applied to both lines


def test_top_forecasts_and_comparison_table(data):
    from supplyradar.core.forecast.engine import build_comparison_table

    data = _trend_workbook(data)
    fc = run_item_forecast(data, "it-1", 6)
    c = fc.combos[0]
    # top-3 competitors are exposed, best first, each with a full-horizon forecast
    assert 1 <= len(c.top_forecasts) <= 3
    assert c.top_forecasts[0].rank == 1
    assert all(len(t.forecast) == 6 for t in c.top_forecasts)

    table, fc_cols = build_comparison_table(c)
    # history rows carry actuals; forecast rows carry the top-3 columns
    assert {"Month", "Actual rate", "Actual consumption"} <= set(table.columns)
    assert len(fc_cols) == len(c.top_forecasts)
    hist_months = c.history.index.astype(str).tolist()
    fut_months = c.forecast["month"].astype(str).tolist()
    assert table["Month"].tolist() == hist_months + fut_months
    # a history row has an actual rate but no forecast; a future row is the reverse
    hist_row = table.iloc[0]
    fut_row = table.iloc[len(hist_months)]
    assert pd.notna(hist_row["Actual rate"]) and pd.isna(hist_row[fc_cols[0]])
    assert pd.isna(fut_row["Actual rate"]) and pd.notna(fut_row[fc_cols[0]])


def test_every_model_row_carries_its_prediction(data):
    """No model result without its actual forecast value: every ranked
    pipeline in the competition table carries next_rate and avg_rate, and the
    selected row's values match the published forecast."""
    data = _trend_workbook(data)
    fc = run_item_forecast(data, "it-1", 6)
    c = fc.combos[0]
    comp = c.competition
    assert {"next_rate", "avg_rate"} <= set(comp.columns)
    assert comp["next_rate"].notna().all()
    assert comp["avg_rate"].notna().all()
    sel = comp.loc[comp["selected"]].iloc[0]
    assert sel["next_rate"] == pytest.approx(c.forecast["rate"].iloc[0])
    assert sel["avg_rate"] == pytest.approx(c.forecast["rate"].mean())


def test_multi_item_overview_carries_predictions(data):
    """The fleet competition (multiple materials) also ships each winner's
    prediction and its rate unit, not just error scores."""
    from supplyradar.core.forecast.engine import run_all_items

    overview = run_all_items(data, 3, items=["it-1", "it-2"])
    assert not overview.empty
    assert {"next_rate", "avg_rate", "rate_uom"} <= set(overview.columns)
    assert overview["next_rate"].notna().all()
    assert overview["avg_rate"].notna().all()
    assert set(overview["rate_uom"]) <= {"kg/ton", "pc/heat", "ton/day"}


def test_forecast_rows_drive_stock_projection(data):
    data = _trend_workbook(data)
    fc = run_item_forecast(data, "it-1", 3)
    rows = forecast_projected_consumption(data, {"it-1": fc}, "2026-01-01")
    assert (rows["consumption_type"] == "forecast").all()
    assert set(rows["item_code"]) == {"it-1"}
    # kg/ton: daily tons = rate * qty1/1000 with qty1=1000 -> rate itself
    jan2 = rows.loc[rows["date"] == "2026-01-02"].iloc[0]
    expected_rate = fc.combos[0].forecast["rate"].iloc[0]
    assert jan2["cons_qty_ton"] == pytest.approx(expected_rate, rel=1e-6)
    # and the stock projection accepts consumption_type='forecast'
    proj = build_stock_projection(data, rows, "2026-01-01")
    it1 = proj.loc[proj["item_code"] == "it-1"].sort_values("date")
    assert it1["qty"].iloc[0] == pytest.approx(100.0 - expected_rate)

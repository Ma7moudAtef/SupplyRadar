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

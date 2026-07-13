import pandas as pd
import pytest

from supplyradar.core.prep_consumption import (
    CONSUMPTION_COLUMNS,
    ContractError,
    build_projected_consumption,
    horizon_end_for,
)

BASE = "2026-01-01"


def test_golden_two_item_three_day(data):
    """Hand-computed golden values for the 2-item, 3-plan-day workbook."""
    proj = build_projected_consumption(data, BASE)

    assert list(proj.columns) == CONSUMPTION_COLUMNS
    assert (proj["consumption_type"] == "plan").all()

    it1 = proj.loc[(proj["item_code"] == "it-1")
                   & (proj["date"] == "2026-01-02")].iloc[0]
    # kg/ton: 2 kg/ton * 1000 t / 1000 = 2 t; base uom ton; $ = 2 * 100
    assert it1["cons_qty_ton"] == pytest.approx(2.0)
    assert it1["cons_qty_base_uom"] == pytest.approx(2.0)
    assert it1["cons_$"] == pytest.approx(200.0)

    it2 = proj.loc[(proj["item_code"] == "it-2")
                   & (proj["date"] == "2026-01-03")].iloc[0]
    # pc/heat: 0.4 pc/heat * 5 heats = 2 pc; 2 pc * 20 kg / 1000 = 0.04 t
    assert it2["cons_qty_base_uom"] == pytest.approx(2.0)
    assert it2["cons_qty_ton"] == pytest.approx(0.04)
    assert it2["cons_$"] == pytest.approx(100.0)

    # no plan rows after Jan 3 -> no production-driven consumption after Jan 3
    later = proj.loc[proj["date"] > "2026-01-03"]
    assert later.empty


def test_ton_per_day_is_flat_per_calendar_day(data):
    data.consumption_figs.loc[0, "std_cons_rate_uom"] = "ton/day"
    data.consumption_figs.loc[0, "std_cons_rate"] = 0.5
    proj = build_projected_consumption(data, BASE)
    it1 = proj.loc[proj["item_code"] == "it-1"]
    # flat every calendar day of [base .. horizon_end], not just plan days
    horizon = horizon_end_for(data)
    assert len(it1) == (horizon - pd.Timestamp(BASE)).days + 1
    assert (it1["cons_qty_ton"] == 0.5).all()


def test_missing_combo_raises_not_silent_zeros(data):
    # the plan produces (f, 1) but it-2 loses its rate for that combo
    data.consumption_figs = data.consumption_figs.loc[
        data.consumption_figs["item_code"] != "it-2"]
    with pytest.raises(ContractError, match="it-2"):
        build_projected_consumption(data, BASE)


def test_base_date_outside_window_rejected(data):
    with pytest.raises(ValueError, match="allowed window"):
        build_projected_consumption(data, "2025-12-15")  # before snapshot
    with pytest.raises(ValueError, match="allowed window"):
        build_projected_consumption(data, "2026-02-10")  # after last plan date


def test_unknown_rate_uom_raises(data):
    data.consumption_figs.loc[0, "std_cons_rate_uom"] = "liters/day"
    with pytest.raises(Exception, match="liters/day"):
        build_projected_consumption(data, BASE)

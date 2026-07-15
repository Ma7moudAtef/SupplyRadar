import numpy as np
import pandas as pd
import pytest

from supplyradar.core.prep_consumption import build_projected_consumption
from supplyradar.core.prep_stock import build_stock_projection, summarize_risk

BASE = "2026-01-01"


@pytest.fixture
def projection(data):
    projected = build_projected_consumption(data, BASE)
    return build_stock_projection(data, projected, BASE)


def test_golden_quantities(projection):
    it1 = projection.loc[projection["item_code"] == "it-1"].set_index("date")
    # opening 100, 2 t/day consumed Jan 1-3, +20 t lot arrives Jan 2
    assert it1.loc["2026-01-01", "qty"] == pytest.approx(98.0)
    assert it1.loc["2026-01-02", "qty"] == pytest.approx(116.0)
    assert it1.loc["2026-01-03", "qty"] == pytest.approx(114.0)
    assert it1.loc["2026-01-31", "qty"] == pytest.approx(114.0)
    assert (it1["stock_stage"] == "warehouse").all()  # cum demand never eats lot


def test_property_qty_never_diverges(data):
    """qty(D) == opening + arrived(D) - cumulative consumption(D), always."""
    projected = build_projected_consumption(data, BASE)
    projection = build_stock_projection(data, projected, BASE)
    plan = projected.loc[projected["consumption_type"] == "plan"]
    for item, grp in projection.groupby("item_code"):
        grp = grp.sort_values("date").reset_index(drop=True)
        opening = data.stock.set_index("item_code").loc[item, "qty"]
        lots = data.delivery.loc[data.delivery["item_code"] == item]
        col = ("cons_qty_ton" if grp["uom"].iloc[0] == "ton"
               else "cons_qty_base_uom")
        demand = (plan.loc[plan["item_code"] == item]
                  .groupby("date")[col].sum()
                  .reindex(grp["date"], fill_value=0.0).to_numpy())
        for i, row in grp.iterrows():
            arrived = lots.loc[
                lots["arrival_date_in_plant"] <= row["date"], "qty_delivered"].sum()
            expected = opening + arrived - demand[: i + 1].sum()
            assert row["qty"] == pytest.approx(expected), (item, row["date"])


def test_stage_transitions_on_exact_exhaustion_dates(data):
    """Stage flips exactly on the date a bucket runs dry."""
    # it-1 consumes a flat 5 t/day; warehouse holds 10 t; a 12 t lot lands Jan 5
    data.stock.loc[data.stock["item_code"] == "it-1", "qty"] = 10.0
    cf = data.consumption_figs
    cf.loc[cf["item_code"] == "it-1", ["std_cons_rate", "std_cons_rate_uom"]] = \
        [5.0, "ton/day"]
    data.delivery = pd.DataFrame({
        "item_code": ["it-1"], "supplier_name": ["acme"],
        "shipment_lot_no.": ["lot-9"], "qty_delivered": [12.0],
        "arrival_date_in_plant": pd.to_datetime(["2026-01-05"]),
        "delivery_label": ["on ship"],
    })
    projected = build_projected_consumption(data, BASE)
    proj = build_stock_projection(data, projected, BASE)
    it1 = proj.loc[proj["item_code"] == "it-1"].set_index("date")

    # cum demand: 5, 10, 15, 20, 25 ... buckets: warehouse 10 | on ship 12 -> 22
    assert it1.loc["2026-01-01", "stock_stage"] == "warehouse"   # cum 5 < 10
    assert it1.loc["2026-01-02", "stock_stage"] == "warehouse"   # cum 10 == 10
    assert it1.loc["2026-01-03", "stock_stage"] == "on ship"     # cum 15 > 10
    assert it1.loc["2026-01-04", "stock_stage"] == "on ship"     # cum 20 < 22
    assert it1.loc["2026-01-05", "stock_stage"] == "gap"         # cum 25 > 22
    # shortage before arrival is visible as negative qty, not clipped
    assert it1.loc["2026-01-03", "qty"] == pytest.approx(-5.0)
    assert it1.loc["2026-01-05", "qty"] == pytest.approx(22.0 - 25.0)
    # the delivery bucket carries its lot number for the hover
    assert it1.loc["2026-01-04", "lot_no"] == "lot-9"


def test_stage_entry_dates_match_transitions(data):
    from supplyradar.core.prep_stock import stage_entry_dates
    data.stock.loc[data.stock["item_code"] == "it-1", "qty"] = 10.0
    cf = data.consumption_figs
    cf.loc[cf["item_code"] == "it-1", ["std_cons_rate", "std_cons_rate_uom"]] = \
        [5.0, "ton/day"]
    data.delivery = pd.DataFrame({
        "item_code": ["it-1"], "supplier_name": ["acme"],
        "shipment_lot_no.": ["lot-9"], "qty_delivered": [12.0],
        "arrival_date_in_plant": pd.to_datetime(["2026-01-05"]),
        "delivery_label": ["on ship"],
    })
    projected = build_projected_consumption(data, BASE)
    proj = build_stock_projection(data, projected, BASE)
    entries = stage_entry_dates(proj).set_index(["item_code", "stock_stage"])
    assert entries.loc[("it-1", "warehouse"), "days_from_start"] == 0
    assert entries.loc[("it-1", "on ship"), "days_from_start"] == 2   # Jan 3
    assert entries.loc[("it-1", "gap"), "days_from_start"] == 4       # Jan 5


def test_daily_need_window_shrinks_never_pads_with_zeros(data):
    cf = data.consumption_figs
    cf.loc[cf["item_code"] == "it-1", ["std_cons_rate", "std_cons_rate_uom"]] = \
        [3.0, "ton/day"]
    projected = build_projected_consumption(data, BASE)
    proj = build_stock_projection(data, projected, BASE)
    it1 = proj.loc[proj["item_code"] == "it-1"].set_index("date")
    # flat 3 t/day to the horizon: a shrinking window must still average 3,
    # a zero-padded one would fake a smaller need near the end
    assert it1["daily_need"].iloc[-1] == pytest.approx(3.0)
    assert it1["daily_need"].iloc[0] == pytest.approx(3.0)
    # thresholds are days * daily_need
    assert it1["safety_qty"].iloc[0] == pytest.approx(2.0 * 3.0)
    assert it1["overstock_qty"].iloc[0] == pytest.approx((2.0 + 3.0) * 3.0)


def test_summarize_risk_orders_by_time_to_gap(data):
    cf = data.consumption_figs
    cf.loc[cf["item_code"] == "it-1", ["std_cons_rate", "std_cons_rate_uom"]] = \
        [60.0, "ton/day"]  # 100 t warehouse + 20 t lot vs 60 t/day
    projected = build_projected_consumption(data, BASE)
    proj = build_stock_projection(data, projected, BASE)
    risk = summarize_risk(proj, within_days=30)
    assert risk["item_code"].iloc[0] == "it-1"
    # Jan 2: 120 available - 120 consumed = 0 -> stocked out 1 day after base
    assert risk["time_to_gap_days"].iloc[0] == 1
    assert bool(risk["at_risk"].iloc[0]) is True
    it2 = risk.loc[risk["item_code"] == "it-2"].iloc[0]
    assert it2["time_to_gap_days"] == np.inf

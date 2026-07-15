"""Pipeline Status — the landing page.

build_pipeline_chart(stock_projection, bom, thresholds) + an at-risk table.
Planners see item DESCRIPTIONS (item_code stays the internal key), hover shows
only the lane under the cursor, risk windows are fully adjustable (global gap /
below-safety days plus a separate window per supply stage), and items can be
hidden from the lanes. Items switched to the AMCIP forecast drive their lanes
from the forecast.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import streamlit as st

from supplyradar.app import state
from supplyradar.app.components.validation_panel import render_validation_panel
from supplyradar.viz.pipeline_chart import build_pipeline_chart

st.title("Pipeline Status")

bundle = state.require_data()
if bundle is None:
    st.stop()

defaults = state.load_defaults()
report = st.session_state.get("sr_report")
if report is not None:
    render_validation_panel(report)

labels = state.item_labels(bundle)
display = bundle["display_names"]
clean = bundle["clean"]
bom = clean.bom
projected, projection = state.effective_tables(bundle)
risk, stage_entries = state.risk_tables(bundle)
stages_present = [s for s in dict.fromkeys(projection["stock_stage"].dropna())]

# ---------------------------------------------------------------- risk settings
with st.expander("Risk settings", expanded=False):
    c1, c2 = st.columns(2)
    gap_days = int(c1.number_input(
        "Gap risk window (days)", min_value=1, max_value=548,
        value=int(defaults.get("at_risk_window_days", 30)),
        help="Flag an item if its stock runs out within this many days."))
    safety_days = int(c2.number_input(
        "Below-safety risk window (days)", min_value=1, max_value=548,
        value=int(defaults.get("at_risk_window_days", 30)),
        help="Flag an item if it drops below safety stock within this window."))
    st.caption("Per-stage windows — flag an item if it starts consuming from "
               "that stage within N days (0 = off).")
    stage_days: dict[str, int] = {}
    stage_cols = st.columns(max(len(stages_present) - 1, 1))
    i = 0
    for stage in stages_present:
        if stage == "warehouse":
            continue  # consuming from the warehouse is the healthy state
        stage_days[stage] = int(stage_cols[i % len(stage_cols)].number_input(
            display.get(stage, stage).title() if stage == "gap"
            else display.get(stage, stage),
            min_value=0, max_value=548, value=0, key=f"risk_{stage}"))
        i += 1

# at-risk flags from the planner's windows
entry_days = stage_entries.set_index(["item_code", "stock_stage"])[
    "days_from_start"]
triggers: dict[str, list[str]] = {}
for row in risk.itertuples(index=False):
    fired = []
    if row.time_to_gap_days <= gap_days:
        fired.append(f"gap in {int(row.time_to_gap_days)}d")
    if row.time_to_below_safety_days <= safety_days:
        fired.append(f"below safety in {int(row.time_to_below_safety_days)}d")
    for stage, days in stage_days.items():
        if days > 0:
            d = entry_days.get((row.item_code, stage), np.inf)
            if d <= days:
                fired.append(f"{display.get(stage, stage)} in {int(d)}d")
    if fired:
        triggers[row.item_code] = fired

n_gap = int((risk["time_to_gap_days"] <= gap_days).sum())
n_below = int((risk["time_to_below_safety_days"] <= safety_days).sum())
headline = (f"### {n_gap} item(s) go to gap within {gap_days} days. "
            f"{n_below} drop below safety stock within {safety_days} days.")
n_stage_only = len([i for i, f in triggers.items()
                    if not any(x.startswith(("gap", "below")) for x in f)])
if n_stage_only:
    headline += f" {n_stage_only} more flagged by stage windows."
st.markdown(headline)

switched = state.switched_forecasts()
if switched:
    st.caption("Forecast-driven lanes (AMCIP switch): "
               + ", ".join(sorted(labels.get(i, i) for i in switched)))

# ---------------------------------------------------------------- filters
def _label(code: str) -> str:
    return labels.get(code, display.get(code, code))

f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 3, 3])
cats = sorted(bom["category_level1"].dropna().unique())
sel_cat = f1.multiselect("Category", cats)
sub = bom if not sel_cat else bom.loc[bom["category_level1"].isin(sel_cat)]
cats2 = sorted(sub["category_level2"].dropna().unique())
sel_cat2 = f2.multiselect("Sub-category", cats2)
if sel_cat2:
    sub = sub.loc[sub["category_level2"].isin(sel_cat2)]
sel_stage = f3.multiselect("Reaches stage", stages_present,
                           format_func=lambda s: display.get(s, s))
item_options = sorted(projection["item_code"].unique(), key=_label)
sel_items = f4.multiselect("Items (only show)", item_options,
                           format_func=_label)
hide_items = f5.multiselect("Hide items", item_options, format_func=_label)

c1, c2 = st.columns([1, 2])
at_risk_only = c1.toggle("Show items at risk only", value=False)
granularity = c2.radio("Granularity", ["day", "week", "month"],
                       horizontal=True, label_visibility="collapsed")

items = set(sub["item_code"]) & set(projection["item_code"])
if sel_items:
    items &= set(sel_items)
if hide_items:
    items -= set(hide_items)
if sel_stage:
    reaches = projection.loc[
        projection["stock_stage"].isin(sel_stage), "item_code"]
    items &= set(reaches)
if at_risk_only:
    items &= set(triggers)

ordered = [i for i in risk["item_code"] if i in items]  # time-to-gap ascending
filtered = projection.loc[projection["item_code"].isin(items)]

if not ordered:
    st.info("No items match the current filters — clear a filter to see lanes.")
else:
    fig = build_pipeline_chart(
        filtered, bom, stage_palette=bundle["palette"], items=ordered,
        granularity=granularity,
        display_names={**display, **labels},  # lanes/hover show descriptions
        lane_px=int(defaults.get("lane_px", 110)),
        max_tick_lanes=int(defaults.get("max_tick_lanes", 20)))
    st.plotly_chart(fig, width="stretch",
                    config={"scrollZoom": True, "displaylogo": False})

with st.expander("How to read this chart"):
    st.markdown("""
**How to read**
- The **height** of the area is the projected inventory level (each lane is
  scaled to its own maximum; tick labels show real quantities).
- The area rests on the lane's **zero-stock line** (solid black).
- The **color inside the area** is the supply bucket the plant is consuming
  from at that date: warehouse first, then each incoming delivery lot in
  arrival order.
- Compare the top of the area with the dashed reference lines to judge
  inventory status. Both lines move with the production plan.

**Inventory status guide**
- Above the red dashed line — **overstock**: more than safety cover plus a
  full replenishment cycle; capital is sitting idle.
- Between the dashed lines — healthy.
- Below the green dashed line — **below safety stock**.
- **Red fill** — the pipeline has run dry (`gap`): no bucket left to consume.
  A red area below the zero line is a projected shortage.
""")

st.subheader("At risk under the current windows")
at_risk_tbl = risk.loc[risk["item_code"].isin(triggers)].copy()
if at_risk_tbl.empty:
    st.write("No items at risk under the current windows.")
else:
    at_risk_tbl["item"] = at_risk_tbl["item_code"].map(_label)
    at_risk_tbl["uom"] = at_risk_tbl["uom"].map(lambda v: display.get(v, v))
    at_risk_tbl["triggered_by"] = at_risk_tbl["item_code"].map(
        lambda i: "; ".join(triggers[i]))
    show = at_risk_tbl[["item", "uom", "opening_qty", "first_gap_date",
                        "time_to_gap_days", "first_below_safety_date",
                        "triggered_by"]]
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config={
                     "opening_qty": st.column_config.NumberColumn(format="%.1f"),
                     "first_gap_date": st.column_config.DateColumn(
                         format="YYYY-MM-DD"),
                     "first_below_safety_date": st.column_config.DateColumn(
                         format="YYYY-MM-DD"),
                 })

with st.expander("Raw projection data"):
    st.dataframe(filtered, width="stretch", hide_index=True)

"""Pipeline Status — the landing page.

Exactly: build_pipeline_chart(stock_projection, bom, thresholds) + an at-risk
table. Widgets: item/category filter, granularity toggle, at-risk-only toggle.
Nothing else — no parameter tuning, no model settings. File + date in,
chart out.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from supplyradar.app import state
from supplyradar.app.components.validation_panel import render_validation_panel
from supplyradar.viz.pipeline_chart import build_pipeline_chart

st.title("Pipeline Status")

bundle = state.require_data()
if bundle is None:
    st.stop()

defaults = state.load_defaults()
within = int(defaults.get("at_risk_window_days", 30))
risk = state.get_risk(within)
report = st.session_state.get("sr_report")
if report is not None:
    render_validation_panel(report)

# the one-line answer, before any chart
n_gap = int((risk["time_to_gap_days"] <= within).sum())
n_below = int((risk["time_to_below_safety_days"] <= within).sum())
st.markdown(f"### {n_gap} item(s) go to gap within {within} days. "
            f"{n_below} drop below safety stock.")

clean = bundle["clean"]
projection = bundle["projection"]
display = bundle["display_names"]
bom = clean.bom

f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
cats = sorted(bom["category_level1"].dropna().unique())
sel_cat = f1.multiselect("Category", cats)
sub = bom if not sel_cat else bom.loc[bom["category_level1"].isin(sel_cat)]
cats2 = sorted(sub["category_level2"].dropna().unique())
sel_cat2 = f2.multiselect("Sub-category", cats2)
if sel_cat2:
    sub = sub.loc[sub["category_level2"].isin(sel_cat2)]
stages = list(dict.fromkeys(projection["stock_stage"].dropna()))
sel_stage = f3.multiselect("Reaches stage", stages,
                           format_func=lambda s: display.get(s, s))
sel_items = f4.multiselect("Item", sorted(projection["item_code"].unique()),
                           format_func=lambda s: display.get(s, s))

c1, c2 = st.columns([1, 2])
at_risk_only = c1.toggle(f"At risk within {within} days only", value=False)
granularity = c2.radio("Granularity", ["day", "week", "month"],
                       horizontal=True, label_visibility="collapsed")

items = set(sub["item_code"]) & set(projection["item_code"])
if sel_items:
    items &= set(sel_items)
if sel_stage:
    reaches = projection.loc[projection["stock_stage"].isin(sel_stage), "item_code"]
    items &= set(reaches)
if at_risk_only:
    items &= set(risk.loc[risk["at_risk"], "item_code"])

ordered = [i for i in risk["item_code"] if i in items]  # time-to-gap ascending
filtered = projection.loc[projection["item_code"].isin(items)]

if not ordered:
    st.info("No items match the current filters — clear a filter to see lanes.")
else:
    fig = build_pipeline_chart(
        filtered, bom, stage_palette=bundle["palette"], items=ordered,
        granularity=granularity, display_names=display,
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

st.subheader(f"At risk within {within} days")
at_risk_tbl = risk.loc[risk["at_risk"]].merge(
    bom[["item_code", "item_description"]], on="item_code", how="left")
if at_risk_tbl.empty:
    st.write("No items at risk in this window.")
else:
    show = at_risk_tbl.assign(
        item=at_risk_tbl["item_code"].map(lambda v: display.get(v, v)),
        uom=at_risk_tbl["uom"].map(lambda v: display.get(v, v)),
    )[["item", "item_description", "uom", "opening_qty", "first_gap_date",
       "time_to_gap_days", "first_below_safety_date"]]
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config={
                     "opening_qty": st.column_config.NumberColumn(format="%.1f"),
                     "first_gap_date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                     "first_below_safety_date": st.column_config.DateColumn(
                         format="YYYY-MM-DD"),
                 })

with st.expander("Raw projection data"):
    st.dataframe(filtered, width="stretch", hide_index=True)

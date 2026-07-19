"""Pipeline Status — the landing page.

build_pipeline_chart(stock_projection, bom, thresholds) + an at-risk table.
Planners see item DESCRIPTIONS (item_code stays the internal key), hover shows
only the lane under the cursor, risk windows are fully adjustable (global gap /
below-safety days plus a separate window per supply stage), and items can be
hidden from the lanes. Items switched to the AMCIP forecast drive their lanes
from the forecast.

PERFORMANCE:
- The risk panel and the filter panel are Apply forms (components/apply.py):
  composing choices causes NO rerun until the user clicks Apply (unless the
  sidebar auto-apply toggle is on).
- The Plotly figure comes from state.cached_pipeline_figure — a zero-copy
  st.cache_resource keyed on the applied choices — so a rerun with unchanged
  choices skips the ~3 s rebuild entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import streamlit as st

from supplyradar.app import state
from supplyradar.app.components import apply, tutorial
from supplyradar.app.components.validation_panel import render_validation_panel

# page title + the Tutorial toggle (tour mode highlights every section below)
tour = tutorial.begin("pipeline", "Pipeline Status")

# ---- data guard: everything below needs a computed bundle -------------------
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

def _label(code: str) -> str:
    """Display label (BOM description) for an item code."""
    return labels.get(code, display.get(code, code))

# ---- risk settings panel (Apply form) ---------------------------------------
# Values live in session_state under their widget keys; with the form they only
# change when 'Apply risk settings' is clicked.
tutorial.tip("pipeline", "Risk settings",
             "Set how early a material counts as at risk: the gap window "
             "(days until stock runs out), the below-safety window, and one "
             "window per supply stage (e.g. flag anything consuming from a "
             "'not paid' lot within N days). Click Apply to recompute.")
with st.expander("Risk settings", expanded=tour):
    with apply.panel("risk_settings"):
        c1, c2 = st.columns(2)
        c1.number_input(
            "Gap risk window (days)", min_value=1, max_value=548,
            value=int(defaults.get("at_risk_window_days", 30)), key="risk_gap_days",
            help="Flag an item if its stock runs out within this many days.")
        c2.number_input(
            "Below-safety risk window (days)", min_value=1, max_value=548,
            value=int(defaults.get("at_risk_window_days", 30)), key="risk_safety_days",
            help="Flag an item if it drops below safety stock within this "
                 "window.")
        st.caption("Per-stage windows — flag an item if it starts consuming "
                   "from that stage within N days (0 = off).")
        stage_cols = st.columns(max(len(stages_present) - 1, 1))
        i = 0
        for stg in stages_present:
            if stg == "warehouse":
                continue  # consuming from the warehouse is the healthy state
            stage_cols[i % len(stage_cols)].number_input(
                display.get(stg, stg).title() if stg == "gap"
                else display.get(stg, stg),
                min_value=0, max_value=548, value=0, key=f"risk_stage_{stg}",
                help=f"Flag an item if it starts consuming from the "
                     f"'{display.get(stg, stg)}' supply stage within this many "
                     "days (0 = don't flag on this stage).")
            i += 1
        apply.button("Apply risk settings",
                     help="Recompute the headline and at-risk flags with "
                          "these windows.")

# read the APPLIED values (form semantics: unchanged until Apply)
gap_days = int(st.session_state.get("risk_gap_days",
                                    defaults.get("at_risk_window_days", 30)))
safety_days = int(st.session_state.get("risk_safety_days",
                                       defaults.get("at_risk_window_days", 30)))
stage_days = {stg: int(st.session_state.get(f"risk_stage_{stg}", 0))
              for stg in stages_present if stg != "warehouse"}

# ---- at-risk flags from the applied windows ---------------------------------
# cheap python loop over <=138 items; recomputed per rerun by design
entry_days = stage_entries.set_index(["item_code", "stock_stage"])[
    "days_from_start"]
triggers: dict[str, list[str]] = {}
for row in risk.itertuples(index=False):
    fired = []
    if row.time_to_gap_days <= gap_days:
        fired.append(f"gap in {int(row.time_to_gap_days)}d")
    if row.time_to_below_safety_days <= safety_days:
        fired.append(f"below safety in {int(row.time_to_below_safety_days)}d")
    for stg, days in stage_days.items():
        if days > 0:
            d = entry_days.get((row.item_code, stg), np.inf)
            if d <= days:
                fired.append(f"{display.get(stg, stg)} in {int(d)}d")
    if fired:
        triggers[row.item_code] = fired

# ---- the one-line answer, before any chart ----------------------------------
tutorial.tip("pipeline", "Headline",
             "The one-line answer for planners: how many materials hit a "
             "stock gap or drop below safety within the applied windows.")
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

# ---- filter panel (Apply form) ----------------------------------------------
tutorial.tip("pipeline", "Filters",
             "Narrow the chart: by category, by supply stage reached, by "
             "picking or hiding specific materials, at-risk only, and the "
             "time granularity (day/week/month). Nothing changes until you "
             "click Apply filters.")
item_options = sorted(projection["item_code"].unique(), key=_label)
with apply.panel("pipeline_filters"):
    f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 3, 3])
    f1.multiselect(
        "Category", sorted(bom["category_level1"].dropna().unique()),
        key="flt_cat",
        help="Show only materials in the chosen top-level categories.")
    # sub-category options derive from the APPLIED categories (form values)
    applied_cats = st.session_state.get("flt_cat", [])
    sub_src = bom if not applied_cats else \
        bom.loc[bom["category_level1"].isin(applied_cats)]
    f2.multiselect(
        "Sub-category", sorted(sub_src["category_level2"].dropna().unique()),
        key="flt_cat2",
        help="Narrow further to sub-categories within the chosen categories.")
    f3.multiselect(
        "Reaches stage", stages_present, key="flt_stage",
        format_func=lambda s: display.get(s, s),
        help="Show only materials that at some point consume from the chosen "
             "supply stage(s).")
    f4.multiselect(
        "Items (only show)", item_options, key="flt_items", format_func=_label,
        help="Restrict the chart to just these materials.")
    f5.multiselect(
        "Hide items", item_options, key="flt_hide", format_func=_label,
        help="Remove these materials from the chart (applied after the "
             "filters above).")

    c1, c2 = st.columns([1, 2])
    c1.toggle(
        "Show items at risk only", value=False, key="flt_at_risk",
        help="Show only materials flagged by the risk windows set above.")
    c2.radio(
        "Time granularity", ["day", "week", "month"], horizontal=True,
        key="flt_gran",
        help="Aggregate the timeline to daily, weekly or monthly points.")
    apply.button("Apply filters",
                 help="Redraw the chart and tables with these filters.")

# ---- resolve the applied filters into the item set --------------------------
sel_cat = st.session_state.get("flt_cat", [])
sel_cat2 = st.session_state.get("flt_cat2", [])
sel_stage = st.session_state.get("flt_stage", [])
sel_items = st.session_state.get("flt_items", [])
hide_items = st.session_state.get("flt_hide", [])
at_risk_only = bool(st.session_state.get("flt_at_risk", False))
granularity = st.session_state.get("flt_gran", "day")

sub = bom if not sel_cat else bom.loc[bom["category_level1"].isin(sel_cat)]
if sel_cat2:
    sub = sub.loc[sub["category_level2"].isin(sel_cat2)]
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

# ---- the chart, from the zero-copy figure cache -----------------------------
tutorial.tip("pipeline", "Pipeline chart",
             "One lane per material, most urgent on top. The area height is "
             "the projected stock; the fill color is the supply bucket "
             "feeding the plant (green warehouse, then each incoming lot, "
             "red gap = nothing left). Dashed lines are the safety (green) "
             "and overstock (red) levels; hover any lane for the details.")
if not ordered:
    st.info("No items match the current filters — clear a filter to see lanes.")
else:
    fig = state.cached_pipeline_figure(
        (bundle["fingerprint"], str(bundle["base_date"].date()),
         state.switch_state_key()),
        projection, bom, dict(bundle["palette"]), {**display, **labels},
        tuple(ordered), granularity,
        int(defaults.get("lane_px", 110)),
        int(defaults.get("max_tick_lanes", 20)))
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

# ---- at-risk table under the applied windows --------------------------------
tutorial.tip("pipeline", "At-risk table",
             "Every material flagged by your windows, with its opening "
             "stock, first gap date and exactly which rule fired "
             "('triggered_by').")
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

# raw numbers for anyone who wants to audit the lanes
tutorial.tip("pipeline", "Raw data",
             "The daily projection table behind the chart, for auditing or "
             "export.")
with st.expander("Raw projection data"):
    st.dataframe(filtered, width="stretch", hide_index=True)

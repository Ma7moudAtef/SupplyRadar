"""Consumption History — actuals vs plan, per item / category / line."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from supplyradar.app import state
from supplyradar.viz.consumption_chart import build_consumption_chart

st.title("Consumption History")

bundle = state.require_data()
if bundle is None:
    st.stop()

clean = bundle["clean"]
display = bundle["display_names"]
cons = bundle["full_consumption"].merge(
    clean.bom[["item_code", "category_level1", "category_level2",
               "item_description"]], on="item_code", how="left")

actual = cons.loc[cons["consumption_type"] == "actual"]
months = actual["date"].dt.to_period("M").nunique()
st.markdown(
    f"### {months} months of actuals across "
    f"{actual['item_code'].nunique()} items; the plan runs to "
    f"{pd.Timestamp(cons['date'].max()).date()}.")

f1, f2, f3, f4 = st.columns(4)
sel_cat = f1.multiselect("Category",
                         sorted(cons["category_level1"].dropna().unique()))
sel_line = f2.multiselect("Production line",
                          sorted(cons["production_line"].dropna().unique()),
                          format_func=lambda s: display.get(s, s))
sel_type = f3.multiselect("Output type",
                          sorted(cons["output_type"].dropna().unique()),
                          format_func=lambda s: display.get(s, s))
sel_items = f4.multiselect("Item", sorted(cons["item_code"].dropna().unique()),
                           format_func=lambda s: display.get(s, s))

filtered = cons
if sel_cat:
    filtered = filtered.loc[filtered["category_level1"].isin(sel_cat)]
if sel_line:
    filtered = filtered.loc[filtered["production_line"].isin(sel_line)]
if sel_type:
    filtered = filtered.loc[filtered["output_type"].isin(sel_type)]
if sel_items:
    filtered = filtered.loc[filtered["item_code"].isin(sel_items)]

value = st.radio("Measure", ["Tons", "Dollars"], horizontal=True,
                 label_visibility="collapsed")
value_col, value_label = (("cons_qty_ton", "Consumption (ton)")
                          if value == "Tons" else ("cons_$", "Consumption ($)"))

if filtered.empty:
    st.info("No consumption rows match the current filters.")
else:
    st.plotly_chart(build_consumption_chart(
        filtered, value_col=value_col, value_label=value_label),
        width="stretch", config={"displaylogo": False})

    monthly = (filtered.assign(month=filtered["date"].dt.to_period("M").astype(str))
               .groupby(["month", "consumption_type"], as_index=False)
               [["cons_qty_ton", "cons_$"]].sum())
    st.dataframe(monthly, width="stretch", hide_index=True)

with st.expander("Raw consumption rows"):
    st.dataframe(filtered, width="stretch", hide_index=True)

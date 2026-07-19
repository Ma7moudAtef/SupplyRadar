"""Cost — value at risk and the cost of the projected consumption.

Uses the EFFECTIVE projected consumption: items the planner switched to the
AMCIP forecast are costed on forecast-driven rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from supplyradar.app import state
from supplyradar.app.components import tutorial
from supplyradar.viz.cost_chart import build_cost_by_category_chart, build_cost_over_time_chart

# page title + the Tutorial toggle (tour mode highlights every section below)
tour = tutorial.begin("cost", "Cost")

bundle = state.require_data()
if bundle is None:
    st.stop()

defaults = state.load_defaults()
within = int(defaults.get("at_risk_window_days", 30))
labels = state.item_labels(bundle)
clean = bundle["clean"]

# ---- effective (switch-aware) consumption, joined to BOM categories ---------
# effective_tables/risk_tables are memoized in session_state, so this page adds
# no recomputation on reruns beyond the cheap sums below.
projected, _ = state.effective_tables(bundle)
risk, _entries = state.risk_tables(bundle)
plan = projected.merge(
    clean.bom[["item_code", "category_level1", "category_level2"]],
    on="item_code", how="left")

# ---- the one-line answer: total projected spend + the share at risk ---------
tutorial.tip("cost", "Headline",
             "Total dollar value of the projected consumption over the whole "
             "horizon, and how much of that spend sits on materials "
             "currently flagged at risk.")
total_cost = plan["cons_$"].sum()
at_risk_items = set(risk.loc[risk["at_risk"], "item_code"])
risk_cost = plan.loc[plan["item_code"].isin(at_risk_items), "cons_$"].sum()

st.markdown(
    f"### Projected consumption costs ${total_cost:,.0f} over the horizon; "
    f"${risk_cost:,.0f} of it sits on items at risk within {within} days.")
if state.switched_forecasts():
    st.caption("Includes forecast-driven consumption for: " + ", ".join(
        sorted(labels.get(i, i) for i in state.switched_forecasts())))

tutorial.tip("cost", "Spend over time",
             "Projected consumption cost per month, stacked by material "
             "category — where the money goes and when.")
st.plotly_chart(build_cost_over_time_chart(plan), width="stretch",
                config={"displaylogo": False})

tutorial.tip("cost", "Category totals & at-risk spend",
             "Left: total projected spend per category. Right: the spend "
             "concentrated on at-risk materials — the money most exposed to "
             "supply gaps.")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_cost_by_category_chart(plan), width="stretch",
                    config={"displaylogo": False})
with c2:
    st.subheader(f"Spend on at-risk items (within {within} days)")
    spend = (plan.loc[plan["item_code"].isin(at_risk_items)]
             .groupby("item_code", as_index=False)["cons_$"].sum()
             .sort_values("cons_$", ascending=False))
    if spend.empty:
        st.write("No spend at risk in this window.")
    else:
        spend.insert(0, "material", spend["item_code"].map(
            lambda c: labels.get(c, c)))
        st.dataframe(spend[["material", "cons_$"]], width="stretch",
                     hide_index=True,
                     column_config={"cons_$": st.column_config.NumberColumn(
                         format="$%.0f")})

tutorial.tip("cost", "Raw data",
             "The projected consumption rows behind these figures, for "
             "auditing or export.")
with st.expander("Raw projected consumption"):
    st.dataframe(plan, width="stretch", hide_index=True)

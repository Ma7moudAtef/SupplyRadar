"""Forecast — stub page. The forecasting engine ships in a later release."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from supplyradar.app import state

st.title("Forecast")

bundle = state.require_data()
if bundle is None:
    st.stop()

st.markdown("### Forecasting is not part of this release.")
st.markdown("""
Today, projections use the **production plan** and standard consumption rates —
deterministic and auditable.

The forecast module will add, in a later release:
- statistical demand forecasts per item (actuals-driven, seasonality-aware),
- confidence bands on the stock projection,
- plan-vs-forecast divergence alerts.

The current projection already answers *"what happens if the plan holds"* —
see **Pipeline Status**.
""")

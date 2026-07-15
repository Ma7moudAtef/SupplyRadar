"""Forecast — the AMCIP planner dashboard.

Per material: intelligence card, history + forecast + confidence band,
model competition with the winner highlighted, explanation, recommendation,
manual override (model / lookback / horizon), and the SWITCH that makes the
forecast drive the stock projection instead of the standard plan rates.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from supplyradar.app import state
from supplyradar.core.forecast.engine import expected_monthly_consumption
from supplyradar.viz.forecast_chart import build_forecast_chart

st.title("Forecast")

bundle = state.require_data()
if bundle is None:
    st.stop()

labels = state.item_labels(bundle)
display = bundle["display_names"]
clean = bundle["clean"]
fingerprint = bundle["fingerprint"]
base_date = bundle["base_date"]

def _label(code: str) -> str:
    return labels.get(code, display.get(code, code))

plan_months = (bundle["projection"]["date"].max().to_period("M")
               - base_date.to_period("M")).n
horizons = [h for h in (3, 6, 12, 18) if h <= max(plan_months, 3)]

top1, top2 = st.columns([3, 1])
item_options = sorted(clean.stock["item_code"].dropna().unique(), key=_label)
item = top1.selectbox("Material", item_options, format_func=_label)
horizon = int(top2.selectbox("Horizon (months)", horizons,
                             index=min(1, len(horizons) - 1)))

fc = state.cached_item_forecast(fingerprint, item, horizon, None, None,
                                clean)
switched = state.switched_forecasts()

if not fc.combos:
    st.warning("This material has too little consumption history to compete "
               "forecast pipelines (fewer than 3 usable months). The standard "
               "plan-rate projection remains in charge.")
    st.stop()

# ------------------------------------------------------------- switch control
sw1, sw2 = st.columns([3, 1])
n_avg = fc.combined_smape()
sw1.markdown(
    f"### {_label(item)} — validation sMAPE {n_avg:.1f}% across "
    f"{len(fc.combos)} production stream(s)")
use_forecast = sw2.toggle(
    "Use forecast in projection", value=item in switched,
    help="Switch this material's stock projection from the standard plan "
         "rates to this forecast. The Pipeline page updates immediately.")

combo_tabs = st.tabs([
    f"{display.get(c.output_type, c.output_type).upper()} / line "
    f"{display.get(c.production_line, c.production_line)} ({c.rate_uom})"
    for c in fc.combos])

manual_active: dict[str, tuple[str | None, int | None]] = {}
for tab, combo in zip(combo_tabs, fc.combos):
    with tab:
        b = combo.behavior
        m = st.columns(6)
        m[0].metric("Behavior", b.classification)
        m[1].metric("Volatility (CV)", f"{b.cv:.2f}" if b.cv == b.cv else "—")
        m[2].metric("Effective memory", combo.effective_memory)
        m[3].metric("Confidence", f"{combo.confidence:.0f}/100")
        m[4].metric("Data quality", f"{combo.data_quality:.0f}/100")
        m[5].metric("Forecastability", f"{combo.forecastability:.0f}/100")
        m2 = st.columns(6)
        m2[0].metric("Model", combo.selected_model)
        trend = ("↑" if b.trend_slope > 0 else "↓") if b.trend_pvalue < 0.05 \
            else "flat"
        m2[1].metric("Trend", f"{trend} ({b.trend_strength:.2f})")
        m2[2].metric("Seasonality", f"{b.seasonality_strength:.2f}")
        m2[3].metric("History", f"{b.n_points} months")
        sel_smape = (combo.competition.loc[combo.competition['selected'],
                                           'smape'].iloc[0]
                     if not combo.competition.empty else float("nan"))
        m2[4].metric("Validation sMAPE", f"{sel_smape:.1f}%")
        m2[5].metric("Last retraining", fc.generated_at.split("T")[0])

        if combo.recommendation.startswith("Automatic"):
            st.success(combo.recommendation)
        else:
            st.warning(combo.recommendation)
        if combo.flags:
            st.caption("Data flags: " + "; ".join(combo.flags))

        # manual override
        with st.expander("Manual override (model / lookback)"):
            models = ["(automatic)"] + sorted(
                combo.competition["model"].unique().tolist())
            windows = ["(automatic)"] + sorted(
                combo.competition["window"].unique().tolist())
            key = f"{combo.output_type}-{combo.production_line}"
            ov_model = st.selectbox("Model", models, key=f"ovm_{key}")
            ov_window = st.selectbox("Lookback window (months)", windows,
                                     key=f"ovw_{key}")
            manual_active[key] = (
                None if ov_model == "(automatic)" else ov_model,
                None if ov_window == "(automatic)" else int(ov_window))

        manual_fc = None
        ov = manual_active.get(f"{combo.output_type}-{combo.production_line}",
                               (None, None))
        if ov != (None, None):
            manual_item = state.cached_item_forecast(
                fingerprint, item, horizon, ov[0], ov[1], clean)
            manual_combo = next(
                (c for c in manual_item.combos
                 if (c.output_type, c.production_line)
                 == (combo.output_type, combo.production_line)), None)
            if manual_combo is not None:
                manual_fc = manual_combo.forecast
                man_smape = (manual_combo.competition.loc[
                    manual_combo.competition['selected'], 'smape'].iloc[0]
                    if not manual_combo.competition.empty else float('nan'))
                st.caption(
                    f"Manual override: {manual_combo.selected_model}, "
                    f"lookback {manual_combo.effective_memory} — sMAPE "
                    f"{man_smape:.1f}% vs automatic {sel_smape:.1f}%.")

        expected = expected_monthly_consumption(clean, combo, base_date)
        uom_lbl = display.get(expected["uom"].iloc[0], expected["uom"].iloc[0]) \
            if not expected.empty else ""
        st.plotly_chart(build_forecast_chart(
            combo.history, combo.forecast,
            rate_label=f"Rate ({combo.rate_uom})",
            expected_consumption=expected,
            consumption_label=f"Expected consumption ({uom_lbl})",
            manual_forecast=manual_fc,
            title=f"{_label(item)} — consumption rate, history and forecast"),
            width="stretch", config={"displaylogo": False})

        st.markdown("**Why this model**")
        st.text(combo.explanation)

        st.markdown("**Model competition** (winner highlighted)")
        comp = combo.competition.copy()
        if not comp.empty:
            comp["window"] = comp["window"].map(
                lambda w: "all" if w >= b.n_points else f"{int(w)}m")
            styled = (comp[["model", "window", "n_folds", "mae", "rmse",
                            "mape", "smape", "mase", "rank", "selected"]]
                      .style.apply(
                          lambda r: ["background-color: #eaf3ea"] * len(r)
                          if r["selected"] else [""] * len(r), axis=1)
                      .format({"mae": "{:.4f}", "rmse": "{:.4f}",
                               "mape": "{:.1f}", "smape": "{:.1f}",
                               "mase": "{:.2f}"}))
            st.dataframe(styled, width="stretch", hide_index=True)

# apply the switch AFTER overrides are known: the displayed forecast is what
# gets applied (manual override included when one is active)
if use_forecast and item not in switched:
    ov_models = {k: v for k, v in manual_active.items() if v != (None, None)}
    if ov_models:
        any_model, any_window = next(iter(ov_models.values()))
        applied = state.cached_item_forecast(
            fingerprint, item, horizon, any_model, any_window, clean)
    else:
        applied = fc
    state.set_switch(item, applied)
    st.rerun()
elif not use_forecast and item in switched:
    state.set_switch(item, None)
    st.rerun()

if switched:
    st.divider()
    st.markdown("**Materials running on forecast in the stock projection:** "
                + ", ".join(sorted(_label(i) for i in switched)))
    if st.button("Reset all to the standard plan projection"):
        for code in list(switched):
            state.set_switch(code, None)
        st.rerun()

# ------------------------------------------------------------- fleet overview
st.divider()
st.subheader("All materials — competition overview")
if st.button("Run the competition for every material"):
    st.session_state["sr_run_all"] = True
if st.session_state.get("sr_run_all"):
    overview = state.cached_run_all(fingerprint, horizon, clean)
    if overview.empty:
        st.write("No material has enough history to forecast.")
    else:
        overview = overview.assign(
            material=overview["item_code"].map(_label),
            output_type=overview["output_type"].map(
                lambda v: display.get(v, v)),
        )[["material", "output_type", "production_line", "classification",
           "model", "memory", "smape", "confidence", "recommendation"]]
        st.dataframe(overview, width="stretch", hide_index=True)

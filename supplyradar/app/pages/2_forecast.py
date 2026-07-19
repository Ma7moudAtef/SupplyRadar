"""Forecast — the AMCIP planner dashboard.

Per material: a compact intelligence card (statistical features), the
history + forecast + confidence chart, a per-stream comparison table
(historical figures + top-3 competing forecasts + an editable "Your
projection" column), the model competition with the winner highlighted, the
explanation, the recommendation, a manual override (model / lookback /
horizon), and the SWITCH that makes the forecast drive the stock projection.

The planner chooses which production streams to forecast (output type and/or
line) and which materials to run the fleet competition for.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from supplyradar.app import state
from supplyradar.app.components.tables import highlight_rows
from supplyradar.core.forecast.engine import build_comparison_table, expected_monthly_consumption
from supplyradar.viz.forecast_chart import build_forecast_chart

st.title("Forecast")

bundle = state.require_data()
if bundle is None:
    st.stop()

labels = state.item_labels(bundle)
display = bundle["display_names"]
clean = bundle["clean"]
bom = clean.bom
fingerprint = bundle["fingerprint"]
base_date = bundle["base_date"]

def _label(code: str) -> str:
    return labels.get(code, display.get(code, code))

def _disp(v) -> str:
    return str(display.get(v, v))

plan_months = (bundle["projection"]["date"].max().to_period("M")
               - base_date.to_period("M")).n
horizons = [h for h in (3, 6, 12, 18) if h <= max(plan_months, 3)]

top1, top2 = st.columns([3, 1])
item_options = sorted(clean.stock["item_code"].dropna().unique(), key=_label)
item = top1.selectbox(
    "Material", item_options, format_func=_label,
    help="The material to forecast. Names come from the BOM description; the "
         "internal item code is used only for joins.")
horizon = int(top2.selectbox(
    "Horizon (months)", horizons, index=min(1, len(horizons) - 1),
    help="How many months ahead to forecast the consumption rate. Capped at "
         "the production plan's own horizon."))

# ------------------------------------------------------- stream grouping (5.a)
avail_types, avail_lines = state.available_streams(fingerprint, item, clean)
flt1, flt2 = st.columns(2)
sel_types = flt1.multiselect(
    "Break out output type(s)", avail_types, default=[], format_func=_disp,
    help="Leave empty to combine all output types into one forecast. Pick "
         "specific types (e.g. B) to forecast each one separately.")
sel_lines = flt2.multiselect(
    "Break out production line(s)", avail_lines, default=[], format_func=_disp,
    help="Leave empty to combine all lines into one forecast. Pick specific "
         "lines to forecast each separately. Example: output B + no line = 1 "
         "forecast for B; output B + lines 1 and 2 = 2 forecasts.")
by_output = tuple(sel_types)   # empty tuple => combine that axis into one group
by_line = tuple(sel_lines)

fc = state.cached_item_forecast(fingerprint, item, horizon, None, None,
                                by_output, by_line, clean)
switched = state.switched_forecasts()

if not fc.combos:
    st.warning("This material has too little consumption history to compete "
               "forecast pipelines (fewer than 3 usable months). The standard "
               "plan-rate projection remains in charge.")
    st.stop()

# ------------------------------------------------------------- switch control
sw1, sw2 = st.columns([3, 1])
shown_smape = [c.competition.loc[c.competition["selected"], "smape"].iloc[0]
               for c in fc.combos if not c.competition.empty]
avg_smape = (sum(shown_smape) / len(shown_smape)) if shown_smape else float("nan")
sw1.markdown(
    f"### {_label(item)} — validation sMAPE {avg_smape:.1f}% across "
    f"{len(fc.combos)} forecast(s)")
use_forecast = sw2.toggle(
    "Use forecast in projection", value=item in switched,
    help="Switch this material's stock projection to these forecasts. Each "
         "forecast is applied to every production stream it covers (an "
         "aggregated B forecast drives B line 1 and B line 2); streams you did "
         "not forecast keep their standard plan rate.")

combo_tabs = st.tabs([f"{c.label} ({c.rate_uom})" for c in fc.combos])

manual_active: dict[str, tuple[str | None, int | None]] = {}
for tab, combo in zip(combo_tabs, fc.combos):
    with tab:
        b = combo.behavior
        sel_smape = (combo.competition.loc[combo.competition["selected"],
                                           "smape"].iloc[0]
                     if not combo.competition.empty else float("nan"))
        trend = ("rising" if b.trend_slope > 0 else "falling") \
            if b.trend_pvalue < 0.05 else "flat"

        # ---- 5.b compact statistical-features card (fits on screen) ----------
        stats = pd.DataFrame({
            "Feature": ["Behaviour", "Volatility (CV)", "Trend", "Seasonality",
                        "Effective memory", "History",
                        "Selected model", "Validation sMAPE", "Confidence",
                        "Data quality", "Forecastability", "Last retraining"],
            "Value": [
                b.classification,
                f"{b.cv:.2f}" if b.cv == b.cv else "—",
                f"{trend} ({b.trend_strength:.2f})",
                f"{b.seasonality_strength:.2f}",
                combo.effective_memory,
                f"{b.n_points} months",
                combo.selected_model,
                f"{sel_smape:.1f}%",
                f"{combo.confidence:.0f}/100",
                f"{combo.data_quality:.0f}/100",
                f"{combo.forecastability:.0f}/100",
                fc.generated_at.split("T")[0],
            ],
        })
        st.markdown("**Material intelligence** — statistical features")
        # three side-by-side 4-row blocks keep every feature on one screen
        cols = st.columns(3)
        for i, col in enumerate(cols):
            block = stats.iloc[i * 4:(i + 1) * 4].reset_index(drop=True)
            col.dataframe(block.style.set_properties(
                **{"color": "#1a2b4c", "font-size": "12px"}),
                width="stretch", hide_index=True)

        if combo.recommendation.startswith("Automatic"):
            st.success(combo.recommendation)
        else:
            st.warning(combo.recommendation)
        if combo.flags:
            st.caption("Data flags: " + "; ".join(combo.flags))

        # ---- manual override (#3 help on every choice) -----------------------
        with st.expander("Manual override (model / lookback)"):
            models = ["(automatic)"] + sorted(
                combo.competition["model"].unique().tolist())
            windows = ["(automatic)"] + sorted(
                combo.competition["window"].unique().tolist())
            key = f"{combo.output_type}-{combo.production_line}"
            ov_model = st.selectbox(
                "Model", models, key=f"ovm_{key}",
                help="Force a specific forecasting model instead of the "
                     "automatically selected winner.")
            ov_window = st.selectbox(
                "Lookback window (months)", windows, key=f"ovw_{key}",
                help="Force how many recent months the model trains on. "
                     "'all' uses the full history.")
            manual_active[key] = (
                None if ov_model == "(automatic)" else ov_model,
                None if ov_window == "(automatic)" else int(ov_window))

        manual_fc = None
        ov = manual_active.get(f"{combo.output_type}-{combo.production_line}",
                               (None, None))
        if ov != (None, None):
            manual_item = state.cached_item_forecast(
                fingerprint, item, horizon, ov[0], ov[1], by_output, by_line,
                clean)
            manual_combo = next(
                (c for c in manual_item.combos
                 if (c.output_type, c.production_line)
                 == (combo.output_type, combo.production_line)), None)
            if manual_combo is not None:
                manual_fc = manual_combo.forecast
                man_smape = (manual_combo.competition.loc[
                    manual_combo.competition["selected"], "smape"].iloc[0]
                    if not manual_combo.competition.empty else float("nan"))
                st.caption(
                    f"Manual override: {manual_combo.selected_model}, "
                    f"lookback {manual_combo.effective_memory} — sMAPE "
                    f"{man_smape:.1f}% vs automatic {sel_smape:.1f}%.")

        expected = expected_monthly_consumption(clean, combo, base_date)
        uom_lbl = _disp(expected["uom"].iloc[0]) if not expected.empty else ""
        st.plotly_chart(build_forecast_chart(
            combo.history, combo.forecast,
            rate_label=f"Rate ({combo.rate_uom})",
            expected_consumption=expected,
            consumption_label=f"Expected consumption ({uom_lbl})",
            manual_forecast=manual_fc,
            title=f"{_label(item)} — consumption rate, history and forecast"),
            width="stretch", config={"displaylogo": False})

        # ---- 5.c comparison table: history + top-3 + your projection ---------
        st.markdown("**History, top-3 forecasts and your own projection** "
                    f"(rate, {combo.rate_uom})")
        table, fc_cols = build_comparison_table(combo)
        table["Your projection"] = float("nan")  # float so NumberColumn types cleanly
        editable_key = f"proj_{item}_{combo.output_type}_{combo.production_line}"
        edited = st.data_editor(
            table, width="stretch", hide_index=True, key=editable_key,
            disabled=[c for c in table.columns if c != "Your projection"],
            column_config={
                "Your projection": st.column_config.NumberColumn(
                    "Your projection",
                    help="Type your own rate projection for any month; blank "
                         "cells are ignored.", format="%.4f"),
                **{c: st.column_config.NumberColumn(c, format="%.4f")
                   for c in ["Actual rate", *fc_cols]},
                "Actual consumption": st.column_config.NumberColumn(
                    "Actual consumption", format="%.2f"),
            })
        user_vals = edited["Your projection"].dropna()
        if not user_vals.empty:
            st.caption(f"You entered {len(user_vals)} manual projection "
                       "value(s). These are yours to record; the automatic "
                       "forecast still drives the projection switch.")

        st.markdown("**Why this model**")
        st.text(combo.explanation)

        st.markdown("**Model competition** (winner highlighted)")
        comp = combo.competition.copy()
        if not comp.empty:
            comp["window"] = comp["window"].map(
                lambda w: "all" if w >= b.n_points else f"{int(w)}m")
            view = comp[["model", "window", "n_folds", "mae", "rmse",
                         "mape", "smape", "mase", "rank", "selected"]]
            styled = highlight_rows(view, comp["selected"]).format(
                {"mae": "{:.4f}", "rmse": "{:.4f}", "mape": "{:.1f}",
                 "smape": "{:.1f}", "mase": "{:.2f}"})
            st.dataframe(styled, width="stretch", hide_index=True)

# apply the switch AFTER overrides are known (at the current grouping; an
# active override on a shown stream is honoured)
if use_forecast and item not in switched:
    ov_models = {k: v for k, v in manual_active.items() if v != (None, None)}
    if ov_models:
        any_model, any_window = next(iter(ov_models.values()))
        applied = state.cached_item_forecast(
            fingerprint, item, horizon, any_model, any_window,
            by_output, by_line, clean)
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

# ------------------------------------------------------- fleet competition (5.c)
st.divider()
st.subheader("Run the competition for chosen materials")
scope = st.radio(
    "Scope", ["Selected materials", "By category"], horizontal=True,
    help="Pick the materials to compete individually, or run a whole "
         "category group at once.")
if scope == "Selected materials":
    chosen = st.multiselect(
        "Materials", item_options, format_func=_label,
        help="The materials to run the forecast competition for. Each takes a "
             "few seconds; results are cached.")
    run_items = list(chosen)
else:
    cats = sorted(bom["category_level1"].dropna().unique())
    chosen_cats = st.multiselect(
        "Categories", cats,
        help="Every material in the chosen category groups is competed.")
    run_items = bom.loc[bom["category_level1"].isin(chosen_cats),
                        "item_code"].tolist()
    run_items = [i for i in run_items if i in set(item_options)]

if st.button("Run competition", disabled=not run_items,
             help="Runs the rolling-origin competition for the chosen "
                  "materials and lists the winning model for each stream."):
    st.session_state["sr_run_items"] = tuple(sorted(set(run_items)))

run_set = st.session_state.get("sr_run_items")
if run_set:
    overview = state.cached_run_selected(fingerprint, horizon, run_set, clean)
    if overview.empty:
        st.write("No chosen material has enough history to forecast.")
    else:
        overview = overview.assign(
            material=overview["item_code"].map(_label),
            output_type=overview["output_type"].map(_disp),
        )[["material", "output_type", "production_line", "classification",
           "model", "memory", "smape", "confidence", "recommendation"]]
        st.dataframe(overview.style.set_properties(**{"color": "#1a2b4c"}),
                     width="stretch", hide_index=True)

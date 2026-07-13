"""Consumption history figures: actuals vs plan.

Pure Plotly figure factories. Stage colors are reserved for the pipeline chart;
this page uses a single neutral accent pair (actual = solid navy bars,
plan = dashed slate line) so color never competes with the semantic palette.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

ACTUAL_COLOR = "#1f4e9c"
PLAN_COLOR = "#8a94a6"
FONT_FAMILY = "Inter, Helvetica, system-ui, sans-serif"
GRID_COLOR = "#e8e8e8"

def build_consumption_chart(
    consumption: pd.DataFrame,
    *,
    value_col: str = "cons_qty_ton",
    value_label: str = "Consumption (ton)",
    title: str = "Consumption — actual vs plan",
) -> go.Figure:
    """Monthly actual vs plan consumption for a pre-filtered consumption slice.

    `consumption` must have: date, consumption_type{actual|plan} and value_col.
    Aggregates to month; the caller does the item/category/line filtering.
    """
    df = consumption.copy()
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    grouped = (df.groupby(["month", "consumption_type"], as_index=False)
               [value_col].sum())

    fig = go.Figure()
    actual = grouped.loc[grouped["consumption_type"] == "actual"]
    plan = grouped.loc[grouped["consumption_type"] == "plan"]
    if not actual.empty:
        fig.add_trace(go.Bar(
            x=actual["month"], y=actual[value_col], name="Actual",
            marker_color=ACTUAL_COLOR,
            hovertemplate="%{x|%b %Y}<br>Actual: %{y:,.1f}<extra></extra>"))
    if not plan.empty:
        fig.add_trace(go.Scatter(
            x=plan["month"], y=plan[value_col], name="Plan", mode="lines",
            line=dict(color=PLAN_COLOR, width=2, dash="dash"),
            hovertemplate="%{x|%b %Y}<br>Plan: %{y:,.1f}<extra></extra>"))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.01, font=dict(size=16, color="#1a2b4c")),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color="#333"),
        xaxis=dict(title="Month", showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(title=value_label, showgrid=True, gridcolor=GRID_COLOR,
                   zeroline=True, zerolinecolor="#333", zerolinewidth=1),
        hovermode="x unified",
        legend=dict(orientation="h", x=1.0, xanchor="right", y=1.08),
        margin=dict(l=60, r=30, t=70, b=50),
        height=420,
    )
    return fig

"""Forecast figures: historical consumption rate + forecast + confidence band,
with expected material consumption below. Pure Plotly; no Streamlit.

One x timeline, two stacked panels (never a dual axis): the top panel is the
forecast variable (consumption rate), the bottom panel the demand it implies.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HISTORY_COLOR = "#1f4e9c"
FORECAST_COLOR = "#b3541e"
OVERRIDE_COLOR = "#5b8a72"
BAND_FILL = "rgba(179, 84, 30, 0.15)"
CONSUMPTION_COLOR = "#7c94c4"
FONT_FAMILY = "Inter, Helvetica, system-ui, sans-serif"
GRID_COLOR = "#e8e8e8"

def build_forecast_chart(
    history: pd.Series,
    forecast: pd.DataFrame,
    *,
    rate_label: str,
    expected_consumption: pd.DataFrame | None = None,
    consumption_label: str = "Expected consumption",
    manual_forecast: pd.DataFrame | None = None,
    title: str = "Consumption rate — history and forecast",
) -> go.Figure:
    """history: PeriodIndex('M') -> rate. forecast: month, rate, lo80..hi95.

    manual_forecast (same schema) overlays the planner's override next to the
    automatic forecast. expected_consumption: month, qty (+ optional uom).
    """
    rows = 2 if expected_consumption is not None else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35] if rows == 2 else [1.0],
        vertical_spacing=0.08)

    hx = history.index.to_timestamp()
    fig.add_trace(go.Scatter(
        x=hx, y=history.values, mode="lines+markers", name="Historical rate",
        line=dict(color=HISTORY_COLOR, width=2), marker=dict(size=5),
        hovertemplate="%{x|%b %Y}<br>Rate: %{y:,.4f}<extra></extra>"),
        row=1, col=1)

    fx = forecast["month"].dt.to_timestamp()
    # bridge point so the forecast line starts where history ends
    bx = [hx[-1]] + list(fx)
    for lo, hi, name in (("lo95", "hi95", "95% interval"),
                         ("lo80", "hi80", "80% interval")):
        fig.add_trace(go.Scatter(
            x=list(fx) + list(fx[::-1]),
            y=list(forecast[hi]) + list(forecast[lo][::-1]),
            fill="toself", fillcolor=BAND_FILL, line=dict(width=0),
            name=name, hoverinfo="skip", showlegend=(lo == "lo80")),
            row=1, col=1)
    fig.add_trace(go.Scatter(
        x=bx, y=[history.values[-1]] + list(forecast["rate"]),
        mode="lines+markers", name="Forecast (automatic)",
        line=dict(color=FORECAST_COLOR, width=2, dash="dash"),
        marker=dict(size=5),
        hovertemplate="%{x|%b %Y}<br>Forecast: %{y:,.4f}<extra></extra>"),
        row=1, col=1)
    if manual_forecast is not None:
        mx = manual_forecast["month"].dt.to_timestamp()
        fig.add_trace(go.Scatter(
            x=[hx[-1]] + list(mx),
            y=[history.values[-1]] + list(manual_forecast["rate"]),
            mode="lines+markers", name="Forecast (manual override)",
            line=dict(color=OVERRIDE_COLOR, width=2, dash="dot"),
            marker=dict(size=5),
            hovertemplate="%{x|%b %Y}<br>Override: %{y:,.4f}<extra></extra>"),
            row=1, col=1)

    if expected_consumption is not None and not expected_consumption.empty:
        cx = expected_consumption["month"].dt.to_timestamp()
        fig.add_trace(go.Bar(
            x=cx, y=expected_consumption["qty"], name=consumption_label,
            marker_color=CONSUMPTION_COLOR,
            hovertemplate="%{x|%b %Y}<br>" + consumption_label +
                          ": %{y:,.2f}<extra></extra>"),
            row=2, col=1)
        fig.update_yaxes(title_text=consumption_label, row=2, col=1,
                         showgrid=True, gridcolor=GRID_COLOR)

    fig.update_yaxes(title_text=rate_label, row=1, col=1,
                     showgrid=True, gridcolor=GRID_COLOR, rangemode="tozero")
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.01,
                   font=dict(size=16, color="#1a2b4c")),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color="#333"),
        hovermode="x unified",
        legend=dict(orientation="h", x=1.0, xanchor="right", y=1.1),
        margin=dict(l=70, r=30, t=70, b=50),
        height=520 if rows == 2 else 400,
        barmode="group",
    )
    return fig

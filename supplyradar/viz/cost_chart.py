"""Cost figures: projected consumption spend over time and by category.

Pure Plotly figure factories; neutral categorical palette deliberately distinct
from the reserved stage colors.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

CATEGORY_CYCLE = ["#1f4e9c", "#7c94c4", "#b08b4f", "#5b8a72", "#8a94a6", "#4f6d7a"]
FONT_FAMILY = "Inter, Helvetica, system-ui, sans-serif"
GRID_COLOR = "#e8e8e8"

def build_cost_over_time_chart(
    consumption: pd.DataFrame,
    *,
    category_col: str = "category_level1",
    title: str = "Projected consumption cost by month",
) -> go.Figure:
    """Stacked monthly plan cost by category.

    `consumption` must carry: date, cons_$ and category_col (joined from bom by
    the caller); plan rows only is the usual input.
    """
    df = consumption.copy()
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    df[category_col] = df[category_col].fillna("uncategorized")
    grouped = (df.groupby(["month", category_col], as_index=False)["cons_$"].sum())
    categories = (grouped.groupby(category_col)["cons_$"].sum()
                  .sort_values(ascending=False).index.tolist())

    fig = go.Figure()
    for i, cat in enumerate(categories):
        part = grouped.loc[grouped[category_col] == cat]
        fig.add_trace(go.Bar(
            x=part["month"], y=part["cons_$"], name=str(cat),
            marker_color=CATEGORY_CYCLE[i % len(CATEGORY_CYCLE)],
            hovertemplate="%{x|%b %Y}<br>" + str(cat) +
                          ": $%{y:,.0f}<extra></extra>"))
    fig.update_layout(
        barmode="stack",
        title=dict(text=f"<b>{title}</b>", x=0.01, font=dict(size=16, color="#1a2b4c")),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color="#333"),
        xaxis=dict(title="Month", showgrid=False),
        yaxis=dict(title="Cost ($)", showgrid=True, gridcolor=GRID_COLOR),
        hovermode="x unified",
        legend=dict(orientation="h", x=1.0, xanchor="right", y=1.08),
        margin=dict(l=60, r=30, t=70, b=50),
        height=420,
    )
    return fig

def build_cost_by_category_chart(
    consumption: pd.DataFrame,
    *,
    category_col: str = "category_level1",
    title: str = "Projected cost by category",
) -> go.Figure:
    """Horizontal bar of total plan cost per category, largest first."""
    df = consumption.copy()
    df[category_col] = df[category_col].fillna("uncategorized")
    totals = (df.groupby(category_col)["cons_$"].sum()
              .sort_values(ascending=True))
    fig = go.Figure(go.Bar(
        x=totals.values, y=[str(v) for v in totals.index], orientation="h",
        marker_color=CATEGORY_CYCLE[0],
        hovertemplate="%{y}: $%{x:,.0f}<extra></extra>"))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.01, font=dict(size=16, color="#1a2b4c")),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color="#333"),
        xaxis=dict(title="Cost ($)", showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(title=""),
        margin=dict(l=140, r=30, t=70, b=50),
        height=max(300, 60 + 36 * len(totals)),
    )
    return fig

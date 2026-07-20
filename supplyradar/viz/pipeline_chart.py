"""The SupplyRadar signature visualization: a multi-lane, stage-colored
inventory area chart.

One horizontal lane per item. The AREA HEIGHT is projected stock quantity
(baseline always zero); the FILL COLOR is the stock_stage at that date — which
supply bucket the plant is currently consuming from. Geometry and color are
independent: the top curve stays continuous across a color change.

Design decisions required by the spec:
- The stage vocabulary is read from the data at runtime (warehouse, each
  delivery label, gap). 'gap' is RESERVED red; no palette can override it.
- Each lane is independently normalized to its own max
  (display_y = lane_offset + value / lane_max * LANE_HEIGHT) because real items
  span orders of magnitude; axis labels show real quantities.
- The top boundary is smoothed with PCHIP (no overshoot -> no phantom stock on
  delivery step-ups). Geometry only: stage labels are never interpolated, each
  polygon inherits the stage of its source interval and color changes snap to
  the exact date the new bucket starts feeding the plant.
- Consecutive same-stage intervals are merged; all polygons of one stage within
  a lane share ONE trace (None-separated), so the trace count stays < 10/item.
- safety_qty / overstock_qty are TIME-VARYING series drawn as Scatter lines
  (they scale with the production plan), not add_shape constants.
- Negative qty (shortage) is drawn below the zero line in the gap color. It is
  clamped for display at a fraction of the lane height so a deep shortage
  cannot invade the lane below — the hover always carries the true quantity.
  (Chosen over a hatched band: the below-zero red dip reads instantly and
  keeps the lane's zero baseline meaningful.)

Pure function; no Streamlit imports; no global state.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator

GAP_STAGE = "gap"
WAREHOUSE_STAGE = "warehouse"

GAP_COLOR = "#e74c3c"  # reserved; always red, always loud
DEFAULT_STAGE_COLORS = {WAREHOUSE_STAGE: "#27ae60", GAP_STAGE: GAP_COLOR}
# Delivery-stage ramp: shades of BLUE only, light -> dark. Supply stages are
# spread evenly across it in ETA order (nearest arrival = lightest, furthest =
# darkest), so no supply stage can ever resemble the warehouse green, the gap
# red, or the safety/overstock line colors.
DEFAULT_DELIVERY_CYCLE = [
    "#aed6f1", "#85c1e9", "#5dade2", "#3498db",
    "#2e86c1", "#2874a6", "#21618c", "#1b4f72",
]

TITLE_COLOR = "#1a2b4c"
SUBTITLE_COLOR = "#4b5563"       # darkened from #6b7280 for readability
AXIS_INK = "#1a2b4c"            # axis titles / tick labels / legend text — dark on white
ITEM_LABEL_COLOR = "#1f4e9c"
OUTLINE_COLOR = "#333333"
SAFETY_COLOR = "#1e8449"        # darkened green so the tick text reads on white
OVERSTOCK_COLOR = "#c0392b"     # darkened red so the tick text reads on white
LINE_SAFETY_COLOR = "#27ae60"  # the drawn dashed line stays the brighter hue
LINE_OVERSTOCK_COLOR = "#e74c3c"
GRID_COLOR = "#e8e8e8"
FONT_FAMILY = "Inter, Helvetica, system-ui, sans-serif"

LANE_H = 1.0          # normalized lane height
NEG_H = 0.22          # display clamp for shortage dips (fraction of lane height)
LANE_GAP = 0.42       # whitespace between lanes
LANE_PITCH = LANE_H + NEG_H + LANE_GAP

def resolve_stage_palette(
    stages_in_eta_order: list[str],
    stage_palette: Mapping[str, str] | None = None,
    delivery_cycle: list[str] | None = None,
) -> dict[str, str]:
    """Color per stage. Warehouse keeps its green anchor; delivery labels get
    shades from the blue ramp in ETA order — spread EVENLY across the ramp so
    the full light->dark range is used whether the file has 2 supply stages
    or 8 (nearest supply lightest, furthest darkest).

    'gap' is always GAP_COLOR — that is the one rule no configuration can
    override.
    """
    palette = dict(DEFAULT_STAGE_COLORS)
    if stage_palette:
        palette.update(stage_palette)
    ramp = list(delivery_cycle or DEFAULT_DELIVERY_CYCLE)
    unassigned = [s for s in stages_in_eta_order if s not in palette]
    for stage, color in zip(unassigned, _spread_ramp(ramp, len(unassigned))):
        palette[stage] = color
    palette[GAP_STAGE] = GAP_COLOR
    return palette

def _spread_ramp(ramp: list[str], n: int) -> list[str]:
    """n colors evenly spaced across the ramp, endpoints included, so adjacent
    stages stay clearly distinguishable. A single stage takes the ramp's
    middle shade (an endpoint would be too washed / too dark alone); more
    stages than ramp colors falls back to cycling."""
    m = len(ramp)
    if n <= 0 or m == 0:
        return []
    if n == 1:
        return [ramp[m // 2]]
    if n > m:
        return [ramp[i % m] for i in range(n)]
    return [ramp[round(i * (m - 1) / (n - 1))] for i in range(n)]

def palette_from_delivery(
    delivery: pd.DataFrame,
    stage_palette: Mapping[str, str] | None = None,
    delivery_cycle: list[str] | None = None,
) -> dict[str, str]:
    """Stable stage palette computed ONCE from the delivery sheet.

    Labels are ordered by supply certainty (earliest arrival first) so the
    nearest supply gets the first cycle color. Compute this on the full data
    and pass it to build_pipeline_chart so filtering items never repaints a
    stage — color follows the entity, not the current view.
    """
    eta = (delivery.dropna(subset=["delivery_label"])
           .groupby("delivery_label")["arrival_date_in_plant"].min()
           .sort_values())
    return resolve_stage_palette(list(eta.index), stage_palette, delivery_cycle)

def build_pipeline_chart(
    stock_projection: pd.DataFrame,
    bom: pd.DataFrame | None = None,
    thresholds: pd.DataFrame | None = None,
    *,
    stage_palette: Mapping[str, str] | None = None,
    items: list[str] | None = None,
    granularity: str = "day",
    display_names: Mapping[str, str] | None = None,
    lane_px: int = 110,
    max_tick_lanes: int = 20,
    title: str = "PIPELINE STATUS",
    subtitle: str = "Projected inventory over time by item, with supply stage and thresholds",
) -> go.Figure:
    """Build the pipeline status figure from a stock_projection table.

    stock_projection columns: item_code, date, qty, stock_stage, uom,
    daily_need, days_of_cover, safety_qty, overstock_qty [, lot_no, supplier].
    If `items` is None every item is shown, sorted by time-to-gap ascending
    (most urgent lane on top). `thresholds` may override the safety/overstock
    series (item_code, date, safety_qty, overstock_qty).
    """
    if granularity not in ("day", "week", "month"):
        raise ValueError(f"granularity must be day|week|month, got {granularity!r}")
    display_names = dict(display_names or {})

    df = stock_projection.copy()
    if thresholds is not None:
        df = df.drop(columns=["safety_qty", "overstock_qty"], errors="ignore").merge(
            thresholds[["item_code", "date", "safety_qty", "overstock_qty"]],
            on=["item_code", "date"], how="left")

    if items is None:
        order = _time_to_gap_order(df)
    else:
        order = [i for i in items if i in set(df["item_code"])]
    df = df.loc[df["item_code"].isin(order)]

    if df.empty or not order:
        fig = go.Figure()
        fig.add_annotation(text="No items match the current filters.",
                           showarrow=False, font=dict(size=14, color=SUBTITLE_COLOR))
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        return fig

    df = _aggregate(df, granularity)

    stages_eta = _stages_in_eta_order(df)
    palette = resolve_stage_palette(stages_eta, stage_palette)

    n = len(order)
    offsets = {item: (n - 1 - i) * LANE_PITCH for i, item in enumerate(order)}
    x_min = pd.Timestamp(df["date"].min()).to_pydatetime()
    x_max = pd.Timestamp(df["date"].max()).to_pydatetime()

    fig = go.Figure()
    zero_x: list = []
    zero_y: list = []
    safety_x: list = []
    safety_y: list = []
    over_x: list = []
    over_y: list = []
    annotations: list[dict] = []
    seen_stages: set[str] = set()

    refine = _refine_factor(n, df["date"].nunique())
    hover_step = 1 if n <= 40 else 7
    date_strings = {d: _fmt_date(d) for d in df["date"].unique()}
    lanes = {item: grp for item, grp in df.groupby("item_code", sort=False)}

    for item in order:
        lane = lanes[item].sort_values("date")
        off = offsets[item]
        qty = lane["qty"].to_numpy(dtype=float)
        safety = np.nan_to_num(lane["safety_qty"].to_numpy(dtype=float))
        over = np.nan_to_num(lane["overstock_qty"].to_numpy(dtype=float))
        lane_max = max(float(np.max(qty, initial=0.0)), float(safety.max()),
                       float(over.max()), 1e-9)
        scale = LANE_H / lane_max

        x = lane["date"].to_numpy()
        x_num = x.astype("datetime64[ns]").astype("int64") / 86_400_000_000_000.0
        y = qty * scale

        dense_x_num, dense_y = _pchip_dense(x_num, y, refine)
        dense_y = np.clip(dense_y, -NEG_H, LANE_H)
        dense_x = (dense_x_num * 86_400_000_000_000.0).astype("int64").astype(
            "datetime64[ns]")

        stages = lane["stock_stage"].to_numpy(dtype=object)
        runs = _stage_runs(stages)

        # one None-separated polygon trace per stage present in this lane
        for stage in dict.fromkeys(stages):  # preserves first-appearance order
            poly_x: list = []
            poly_y: list = []
            for start, end in runs[stage]:
                # the polygon extends to the first sample of the NEXT run so
                # the color changes exactly on the date the new bucket starts
                x_lo = x_num[start]
                x_hi = x_num[end + 1] if end + 1 < len(x_num) else x_num[end]
                m = (dense_x_num >= x_lo) & (dense_x_num <= x_hi)
                if not m.any():
                    continue
                seg_x = _to_datetimes(dense_x[m])
                seg_y = np.maximum(dense_y[m], 0.0)
                if poly_x:
                    poly_x.append(None)
                    poly_y.append(None)
                poly_x.extend(seg_x + [seg_x[-1], seg_x[0]])
                poly_y.extend((off + seg_y).tolist() + [off, off])
            if not poly_x:
                continue
            fig.add_trace(go.Scatter(
                x=_x_array(poly_x), y=_y_array(poly_y), mode="lines", fill="toself",
                fillcolor=palette[stage], line=dict(width=0),
                legendgroup=f"stage-{stage}", showlegend=False,
                hoverinfo="skip", name=_stage_label(stage, display_names)))
            seen_stages.add(stage)

        # shortage: below-zero dips, gap red, one trace per lane
        short_x, short_y = _shortage_polygons(dense_x, dense_x_num, dense_y, off)
        if short_x:
            fig.add_trace(go.Scatter(
                x=_x_array(short_x), y=_y_array(short_y), mode="lines", fill="toself",
                fillcolor=GAP_COLOR, line=dict(width=0),
                legendgroup=f"stage-{GAP_STAGE}", showlegend=False,
                hoverinfo="skip", name=_stage_label(GAP_STAGE, display_names)))
            seen_stages.add(GAP_STAGE)

        # single continuous top outline per item (spline-smoothed so few-lane
        # views read as gently curved, not polygonal)
        fig.add_trace(go.Scatter(
            x=dense_x, y=off + dense_y, mode="lines",
            line=dict(color=OUTLINE_COLOR, width=1.5,
                      shape="spline" if n <= 12 else "linear", smoothing=0.5),
            showlegend=False, hoverinfo="skip"))

        # invisible markers carrying the tooltip (decimated on very tall charts;
        # a filtered view gets full daily hover)
        hv = slice(None, None, hover_step)
        fig.add_trace(go.Scatter(
            x=x[hv], y=(off + np.clip(y, -NEG_H, LANE_H))[hv], mode="markers",
            marker=dict(size=9, color="rgba(0,0,0,0)",
                        line=dict(width=0)),
            text=_hover_texts(item, lane.iloc[hv], palette, display_names,
                              date_strings),
            hovertemplate="%{text}<extra></extra>", showlegend=False))

        # per-lane reference lines, merged across lanes below
        if zero_x:
            for acc in (zero_x, zero_y, safety_x, safety_y, over_x, over_y):
                acc.append(None)
        th = slice(None, None, hover_step)
        x_dt = _to_datetimes(x[th])
        zero_x.extend([x_min, x_max])
        zero_y.extend([off, off])
        safety_x.extend(x_dt)
        safety_y.extend((off + safety[th] * scale).tolist())
        over_x.extend(x_dt)
        over_y.extend((off + over[th] * scale).tolist())

        annotations.extend(_lane_annotations(
            item, off, safety * scale, over * scale, safety, over,
            display_names, show_ticks=n <= max_tick_lanes))

    # reference lines: one trace each for the whole figure. Curved (spline) so
    # the time-varying thresholds read as smooth guides, not sawtooth polylines.
    fig.add_trace(go.Scatter(x=_x_array(safety_x), y=_y_array(safety_y), mode="lines",
                             line=dict(color=LINE_SAFETY_COLOR, width=1.4,
                                       dash="dash", shape="spline", smoothing=0.6),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=_x_array(over_x), y=_y_array(over_y), mode="lines",
                             line=dict(color=LINE_OVERSTOCK_COLOR, width=1.4,
                                       dash="dash", shape="spline", smoothing=0.6),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=_x_array(zero_x), y=_y_array(zero_y), mode="lines",
                             line=dict(color="black", width=1),
                             showlegend=False, hoverinfo="skip"))

    _add_legend_entries(fig, stages_eta, seen_stages, palette, display_names)

    span_days = (x_max - x_min).days
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=AXIS_INK),
        title=dict(
            text=(f"<b>{title}</b><br>"
                  f"<span style='font-size:12px;color:{SUBTITLE_COLOR}'>"
                  f"{subtitle}</span>"),
            x=0.01, xanchor="left", y=1.0, yanchor="top", pad=dict(t=12),
            font=dict(size=20, color=TITLE_COLOR)),
        margin=dict(l=250, r=70, t=150, b=60),
        height=max(430, 170 + n * lane_px),
        # 'closest' shows ONLY the lane under the cursor (planner feedback:
        # a unified tooltip for 100+ lanes is unreadable)
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", bordercolor="#b8c0cc",
                        font=dict(family=FONT_FAMILY, size=12, color=AXIS_INK)),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.0, yanchor="bottom",
            title=dict(text="<b>Supply Stage (Fill Color)</b>",
                       font=dict(size=11, color=AXIS_INK), side="top"),
            font=dict(size=11, color=AXIS_INK), itemsizing="constant",
            bordercolor="rgba(0,0,0,0)"),
        xaxis=dict(
            title=dict(text="Date", font=dict(size=13, color=AXIS_INK)),
            tickfont=dict(size=11, color=AXIS_INK),
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
            zeroline=False, showline=False,
            dtick=7 * 86_400_000 if span_days <= 130 else "M1",
            tickformat="%b %-d" if span_days <= 130 else "%b %Y",
            range=[x_min, x_max]),
        yaxis=dict(visible=False, fixedrange=True,
                   range=[-NEG_H - 0.15, (n - 1) * LANE_PITCH + LANE_H + 0.25]),
        annotations=annotations,
        dragmode="pan",
    )
    return fig

# ------------------------------------------------------------------ helpers

def _to_datetimes(arr: np.ndarray) -> list:
    """datetime64 array -> list of datetime.datetime (ns .tolist() yields ints)."""
    return arr.astype("datetime64[ms]").tolist()

def _x_array(values: list) -> np.ndarray:
    """datetime list with None separators -> datetime64 array (None -> NaT).

    numpy arrays take plotly's fast validation path; elementwise list
    validation dominates build time at 138 lanes."""
    return np.array(values, dtype="datetime64[ms]")

def _y_array(values: list) -> np.ndarray:
    """float list with None separators -> float array (None -> NaN)."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)

def _time_to_gap_order(df: pd.DataFrame) -> list[str]:
    """Items sorted by time-to-gap ascending — the most urgent lane on top."""
    first_gap = {}
    far_future = pd.Timestamp.max
    for item, grp in df.groupby("item_code", sort=False):
        gaps = grp.loc[grp["qty"] <= 0, "date"]
        below = grp.loc[grp["qty"] < grp["safety_qty"], "date"]
        first_gap[item] = (
            gaps.min() if not gaps.empty else far_future,
            below.min() if not below.empty else far_future,
            str(item),
        )
    return sorted(first_gap, key=first_gap.get)

def _aggregate(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if granularity == "day":
        return df
    freq = {"week": "W-MON", "month": "MS"}[granularity]
    df = df.copy()
    df["date"] = df["date"].dt.to_period(freq[0] if granularity == "month" else "W")
    df["date"] = df["date"].dt.start_time
    # a period's row is the state at its END (last daily row) — a snapshot, not a mean
    return (df.sort_values("date")
              .groupby(["item_code", "date"], as_index=False).last())

def _refine_factor(n_items: int, n_days: int) -> int:
    """Interpolation density budget: smooth for a few lanes, lean for 138.

    At tens of lanes each lane is ~100px tall — sub-day smoothing is invisible
    there, so the budget degrades gracefully to the raw daily polyline. This is
    what keeps a 138 x 550 build under the 2s bar and the payload browsable.
    """
    budget = 400_000
    r = int(budget / max(n_items * max(n_days, 1) * 3, 1))
    return max(1, min(6, r))

def _pchip_dense(x_num: np.ndarray, y: np.ndarray, refine: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """PCHIP-smoothed curve sampled on a grid that contains every source point."""
    if len(x_num) < 3 or refine <= 1:
        return x_num, y
    frac = np.arange(refine) / refine
    grid = x_num[:-1, None] + np.diff(x_num)[:, None] * frac[None, :]
    dense = np.concatenate([grid.ravel(), x_num[-1:]])
    return dense, PchipInterpolator(x_num, y)(dense)

def _stage_runs(stages: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    """Contiguous same-stage runs as {stage: [(start_idx, end_idx)]}, merged."""
    runs: dict[str, list[tuple[int, int]]] = {}
    start = 0
    for i in range(1, len(stages) + 1):
        if i == len(stages) or stages[i] != stages[start]:
            runs.setdefault(stages[start], []).append((start, i - 1))
            start = i
    return runs

def _shortage_polygons(dense_x, dense_x_num, dense_y, off
                       ) -> tuple[list, list]:
    """None-separated closed polygons for every below-zero span of the curve."""
    neg = dense_y < 0
    if not neg.any():
        return [], []
    xs: list = []
    ys: list = []
    idx = np.flatnonzero(np.diff(np.concatenate([[False], neg, [False]])))
    for lo, hi in zip(idx[::2], idx[1::2]):
        lo = max(lo - 1, 0)                    # include the zero-crossing
        hi = min(hi + 1, len(dense_y))
        seg_x = _to_datetimes(dense_x[lo:hi])
        seg_y = np.minimum(dense_y[lo:hi], 0.0)
        if xs:
            xs.append(None)
            ys.append(None)
        xs.extend(seg_x + [seg_x[-1], seg_x[0]])
        ys.extend((off + seg_y).tolist() + [off, off])
    return xs, ys

def _stage_label(stage: str, display_names: Mapping[str, str]) -> str:
    if stage in display_names:
        label = display_names[stage]
    else:
        label = stage
    if stage in (WAREHOUSE_STAGE, GAP_STAGE):
        return label.title() if label == label.lower() else label
    return label

def _uom_label(uom: str, display_names: Mapping[str, str]) -> str:
    """Units display exactly as the data file spells them (display map covers
    both the raw value and its canonical form) — nothing is hardcoded here."""
    return display_names.get(uom, uom)

def _fmt_date(d) -> str:
    """'Feb 23, 2025' — built without strftime's no-padding directive.

    '%-d' is a glibc (Linux/mac) extension and Windows' C runtime rejects it
    with 'ValueError: Invalid format string' (Windows spells it '%#d'). Using
    the day as a plain int keeps the app runnable on any OS. Plotly
    tickformat/hovertemplate strings may keep '%-d': those run in the
    browser via d3-time-format, which supports it everywhere.
    """
    ts = pd.Timestamp(d)
    return f"{ts.strftime('%b')} {ts.day}, {ts.year}"

def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "∞"
    if abs(v) >= 10_000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:,.1f}".rstrip("0").rstrip(".")

def _hover_texts(item: str, lane: pd.DataFrame, palette: Mapping[str, str],
                 display_names: Mapping[str, str],
                 date_strings: Mapping | None = None) -> list[str]:
    """Mock-style tooltip: item, date, qty+uom, stage (in its color),
    safety (green), overstock (red), days of cover, lot/supplier if present."""
    item_disp = display_names.get(item, item)
    date_strings = date_strings or {}
    texts = []
    has_lot = "lot_no" in lane.columns
    for row in lane.itertuples(index=False):
        stage = row.stock_stage
        color = palette.get(stage, "#333")
        uom = _uom_label(row.uom, display_names)
        date_s = date_strings.get(row.date) or _fmt_date(row.date)
        lines = [
            f"<b>{item_disp}</b>",
            f"Date: {date_s}",
            f"Inventory: {_fmt(row.qty)} {uom}",
            f"Stage: <span style='color:{color}'><b>"
            f"{_stage_label(stage, display_names)}</b></span>",
            f"Safety Stock: <span style='color:{SAFETY_COLOR}'>"
            f"{_fmt(row.safety_qty)} {uom}</span>",
            f"Overstock: <span style='color:{OVERSTOCK_COLOR}'>"
            f"{_fmt(row.overstock_qty)} {uom}</span>",
            f"Days of Cover: {_fmt(row.days_of_cover)}",
        ]
        if has_lot and row.lot_no is not None and pd.notna(row.lot_no):
            lines.append(f"Shipment Lot: {row.lot_no}")
            if row.supplier is not None and pd.notna(row.supplier):
                lines.append(f"Supplier: {row.supplier}")
        texts.append("<br>".join(lines))
    return texts

def _lane_annotations(item, off, safety_norm, over_norm, safety, over,
                      display_names, *, show_ticks: bool) -> list[dict]:
    """Item label (left, blue, bold) + mirrored per-lane value ticks.

    Ticks label the lane's own scale: 0 (black), the safety level (green) and
    the overstock level (red) at their median normalized position. With many
    lanes the numeric ticks are dropped (hover carries the values) — the
    pixels belong to the lanes.
    """
    label = str(display_names.get(item, item))
    if len(label) > 24:  # descriptions can be long; hover carries the full name
        label = label[:23] + "…"
    anns = [dict(
        text=f"<b>{label}</b>", xref="paper", yref="y",
        x=-0.004, xanchor="right", y=off + LANE_H * 0.5, yanchor="middle",
        showarrow=False, font=dict(size=12, color=ITEM_LABEL_COLOR),
        xshift=-58)]
    if not show_ticks:
        return anns
    ticks = [(0.0, "0", "black")]
    s_pos, s_val = float(np.median(safety_norm)), float(np.median(safety))
    o_pos, o_val = float(np.median(over_norm)), float(np.median(over))
    if o_pos > 0.08:
        ticks.append((min(o_pos, LANE_H), _fmt(o_val), OVERSTOCK_COLOR))
    if s_pos > 0.08 and abs(s_pos - o_pos) > 0.14:
        ticks.append((min(s_pos, LANE_H), _fmt(s_val), SAFETY_COLOR))
    for pos, label, color in ticks:
        for xref, anchor, shift in ((-0.004, "right", 0), (1.004, "left", 0)):
            anns.append(dict(
                text=label, xref="paper", yref="y", x=max(xref, 0) if xref > 0 else 0,
                xanchor=anchor, y=off + pos, yanchor="middle", showarrow=False,
                font=dict(size=10, color=color),
                xshift=-6 if anchor == "right" else 6))
            anns[-1]["x"] = 0 if anchor == "right" else 1
    return anns

def _stages_in_eta_order(df: pd.DataFrame) -> list[str]:
    """Distinct stages ordered by supply certainty: warehouse first, then each
    delivery label by the first date it becomes the active bucket, gap last."""
    first_seen = (df.loc[df["stock_stage"].notna()]
                  .groupby("stock_stage")["date"].min().sort_values())
    ordered = [s for s in first_seen.index if s not in (WAREHOUSE_STAGE, GAP_STAGE)]
    out = []
    if (df["stock_stage"] == WAREHOUSE_STAGE).any():
        out.append(WAREHOUSE_STAGE)
    out.extend(ordered)
    if (df["stock_stage"] == GAP_STAGE).any() or (df["qty"] < 0).any():
        out.append(GAP_STAGE)
    return out

def _add_legend_entries(fig, stages_eta, seen_stages, palette, display_names):
    """ONE legend entry per stage + the line-style key (right block)."""
    for stage in stages_eta:
        if stage not in seen_stages:
            continue
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=12, color=palette[stage]),
            name=_stage_label(stage, display_names),
            legendgroup=f"stage-{stage}", showlegend=True, hoverinfo="skip"))
    for name, color, dash in (
            ("Overstock (max)", LINE_OVERSTOCK_COLOR, "dash"),
            ("Safety Stock (min)", LINE_SAFETY_COLOR, "dash"),
            ("Zero Stock", "black", "solid")):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=color, width=1.5, dash=dash),
            name=name, legendgroup="linekey", showlegend=True, hoverinfo="skip"))

import numpy as np
import pandas as pd
import pytest

from supplyradar.viz.pipeline_chart import GAP_COLOR, build_pipeline_chart, resolve_stage_palette


def make_projection(n_items: int = 3, n_days: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_days)
    frames = []
    for i in range(n_items):
        qty = np.linspace(100, -10, n_days)
        stage = np.where(np.arange(n_days) < 12, "warehouse",
                         np.where(np.arange(n_days) < 24, "on ship", "gap"))
        frames.append(pd.DataFrame({
            "item_code": f"item-{i}", "date": dates, "qty": qty,
            "stock_stage": stage, "uom": "ton",
            "daily_need": 3.5, "days_of_cover": np.maximum(qty, 0) / 3.5,
            "safety_qty": 20.0, "overstock_qty": 60.0,
            "lot_no": None, "supplier": None,
        }))
    return pd.concat(frames, ignore_index=True)


def test_trace_count_stays_bounded():
    n_items = 12
    fig = build_pipeline_chart(make_projection(n_items=n_items))
    # < 10 traces per item plus a constant overhead (legend + reference lines)
    assert len(fig.data) < 10 * n_items + 15


def test_stage_boundary_lands_on_exact_date():
    fig = build_pipeline_chart(make_projection(n_items=1))
    on_ship = [t for t in fig.data if t.name == "on ship" and t.fill == "toself"]
    assert len(on_ship) == 1  # merged: one trace per stage per item
    xs = np.array(on_ship[0].x, dtype="datetime64[ms]")
    # the 'on ship' run starts on day 12 — its polygon must start exactly there
    assert xs.min() == np.datetime64("2026-01-13")


def test_gap_is_always_red_even_when_palette_says_otherwise():
    palette = resolve_stage_palette(["warehouse", "gap"],
                                    stage_palette={"gap": "#000000"})
    assert palette["gap"] == GAP_COLOR
    fig = build_pipeline_chart(make_projection(n_items=1),
                               stage_palette={"gap": "#000000"})
    gap_fills = [t for t in fig.data if t.name and t.name.lower() == "gap"
                 and t.fill == "toself"]
    assert gap_fills, "gap polygons missing"
    assert all(t.fillcolor == GAP_COLOR for t in gap_fills)


def test_display_corners_are_rounded_without_overshoot():
    """The drawn curve must be visibly rounded at sharp corners (a V and a
    delivery-style step) while never overshooting the data range and leaving
    straight segments in place."""
    from supplyradar.viz.pipeline_chart import _corner_window, _round_corners

    y = np.concatenate([np.linspace(1.0, 0.0, 100),   # straight decline
                        np.zeros(50),                  # flat at zero
                        np.linspace(0.9, 0.3, 100)])   # step-up then decline
    win = _corner_window(len(y))
    assert win >= 3, "a long daily lane must get real corner rounding"
    s = _round_corners(y, win)

    def max_curvature(a):
        return float(np.max(np.abs(np.diff(a, 2))))

    assert max_curvature(s) < 0.6 * max_curvature(y)   # corners rounded
    assert s.min() >= y.min() - 1e-9                   # no overshoot below
    assert s.max() <= y.max() + 1e-9                   # no phantom stock above
    # straight interior of the decline is untouched (convex-combination kernel)
    assert np.allclose(s[20:70], y[20:70], atol=1e-9)
    # short lanes (weekly/monthly granularity) skip rounding entirely
    assert _corner_window(30) == 1
    short = np.linspace(1.0, 0.0, 30)
    assert np.array_equal(_round_corners(short, 1), short)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def _luminance(hex_color: str) -> float:
    r, g, b = _rgb(hex_color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_delivery_stages_are_blues_light_to_dark_never_greenish():
    """Warehouse is green; supply stages are DISTINCT shades of blue running
    light (nearest arrival) to dark (furthest) — spread across the full ramp
    regardless of how many stages the file has, so none can be confused with
    the warehouse green (or each other)."""
    for stages in (["s1", "s2"], ["s1", "s2", "s3"],
                   ["s1", "s2", "s3", "s4", "s5"]):
        palette = resolve_stage_palette(["warehouse", *stages, "gap"])
        assert palette["warehouse"] == "#27ae60"
        blues = [palette[s] for s in stages]
        assert len(set(blues)) == len(blues), "shades must be distinct"
        lums = [_luminance(c) for c in blues]
        assert lums == sorted(lums, reverse=True), "must run light -> dark"
        for c in blues:  # blue channel dominates -> never reads as green
            r, g, b = _rgb(c)
            assert b > g > r, f"{c} is not a blue shade"


def test_negative_qty_rendered_not_clipped():
    fig = build_pipeline_chart(make_projection(n_items=1))
    lane_floor = 0.0  # single lane: its zero line sits at offset 0
    min_y = min(np.nanmin(np.asarray(t.y, dtype=float))
                for t in fig.data
                if t.y is not None and len(t.y) and t.fill == "toself")
    assert min_y < lane_floor  # shortage visible below the zero line


def test_items_selection_and_order_respected():
    fig = build_pipeline_chart(make_projection(n_items=3),
                               items=["item-2", "item-0"])
    labels = [a["text"] for a in fig.layout.annotations
              if "item-" in str(a["text"])]
    assert labels == ["<b>item-2</b>", "<b>item-0</b>"]


def test_one_legend_entry_per_stage():
    fig = build_pipeline_chart(make_projection(n_items=5))
    legend_names = [t.name for t in fig.data if t.showlegend]
    # one per stage + the three line-style keys, no duplicates
    assert len(legend_names) == len(set(legend_names))
    assert {"Warehouse", "on ship", "Gap"} <= set(legend_names)


def test_bad_granularity_rejected():
    with pytest.raises(ValueError, match="granularity"):
        build_pipeline_chart(make_projection(), granularity="hour")

"""Daily stock projection: a FIFO waterfall of supply buckets vs cumulative demand.

For each item, supply buckets are:
    bucket 0  = opening warehouse qty (from `stock` at base_date), stage 'warehouse'
    bucket i  = each delivery lot, ordered by arrival_date_in_plant, stage =
                its delivery_label, available on/after its arrival date
                (past-due ETAs are clamped to base_date).

For each date D in [base_date .. horizon_end]:
    cum_demand(D) = projected consumption summed over [base_date .. D], in the
                    item's stock uom (ton if stock uom is ton, else base uom)
    available(D)  = warehouse qty + sum(qty_delivered for lots arrived <= D)
    qty(D)        = available(D) - cum_demand(D)      # negative = shortage
    stock_stage(D)= label of the bucket cum_demand is currently consuming from
                    (searchsorted over cumulative bucket capacity);
                    'gap' once cum_demand exceeds every bucket that will ever exist.

Thresholds are stored in DAYS and converted to QUANTITY per day:
    daily_need(D)    = forward-looking mean of projected consumption over
                       [D .. D+29]; the window shrinks near the horizon end and
                       is never padded with zeros.
    safety_qty(D)    = safety_level_days * daily_need(D)
    overstock_qty(D) = (safety_level_days + replenishment_level_days) * daily_need(D)

All series are time-varying because daily_need moves with the production plan.
Vectorized with cumulative sums; no per-day loops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import WorkbookData
from .prep_consumption import ContractError, check_base_date, horizon_end_for
from .uom import convert

PROJECTION_COLUMNS = [
    "item_code", "date", "qty", "stock_stage", "uom", "daily_need",
    "days_of_cover", "safety_qty", "overstock_qty", "lot_no", "supplier",
]

GAP_STAGE = "gap"
WAREHOUSE_STAGE = "warehouse"
_EPS = 1e-9  # float tolerance: a bucket exhausted to the last drop is still that bucket

def build_stock_projection(
    data: WorkbookData,
    projected_consumption: pd.DataFrame,
    base_date: str | pd.Timestamp,
    *,
    daily_need_window: int = 30,
) -> pd.DataFrame:
    """One row per (item_code, date). See module docstring for the model.

    data must have validation exclusions applied. projected_consumption is the
    output of build_projected_consumption (plan rows only are used).
    """
    base_date = pd.Timestamp(base_date)
    check_base_date(data, base_date)
    horizon_end = horizon_end_for(data)
    dates = pd.date_range(base_date, horizon_end, freq="D")
    n_days = len(dates)

    # 'plan' rows come from standard rates; 'forecast' rows from the AMCIP
    # engine when the planner switches an item — both drive the projection
    plan = projected_consumption.loc[
        projected_consumption["consumption_type"].isin(["plan", "forecast"])]

    bom = data.bom.set_index("item_code")[["uom", "unit_wt_kg"]]
    stock = data.stock.set_index("item_code")
    thresholds = _item_thresholds(data)

    # demand per (item, date) in ton and in base uom, dense over the date grid
    demand = (plan.groupby(["item_code", "date"])
              [["cons_qty_ton", "cons_qty_base_uom"]].sum())

    deliveries = data.delivery.sort_values(["arrival_date_in_plant"], kind="stable")

    frames: list[pd.DataFrame] = []
    for item in stock.index:
        srow = stock.loc[item]
        stock_uom = srow["uom"]
        if item not in bom.index:
            raise ContractError(f"stock item {item!r} missing from bom")
        bom_uom = bom.loc[item, "uom"]
        unit_wt = bom.loc[item, "unit_wt_kg"]

        daily = _demand_in_stock_uom(
            demand, item, dates, stock_uom, bom_uom, unit_wt)
        cum_demand = np.cumsum(daily)

        buckets = _buckets_for(item, float(srow["qty"]), deliveries, base_date)
        cum_cap = np.cumsum(buckets["qty"].to_numpy(dtype=float))

        # availability over time (buckets whose arrival <= D)
        arrivals = buckets["available_from"].to_numpy(dtype="datetime64[ns]")
        arrived_count = np.searchsorted(arrivals, dates.to_numpy(), side="right")
        available = np.where(arrived_count > 0,
                             cum_cap[np.maximum(arrived_count - 1, 0)], 0.0)
        qty = available - cum_demand

        # stage: which bucket is cumulative demand currently eating into
        bucket_idx = np.searchsorted(cum_cap, cum_demand - _EPS, side="left")
        in_gap = bucket_idx >= len(cum_cap)
        safe_idx = np.minimum(bucket_idx, len(cum_cap) - 1)
        stage = np.where(in_gap, GAP_STAGE,
                         buckets["stage"].to_numpy(dtype=object)[safe_idx])
        lot_no = np.where(in_gap, None,
                          buckets["lot_no"].to_numpy(dtype=object)[safe_idx])
        supplier = np.where(in_gap, None,
                            buckets["supplier"].to_numpy(dtype=object)[safe_idx])

        daily_need = _forward_mean(daily, daily_need_window)
        safety_days, repl_days = thresholds.get(item, (0.0, 0.0))
        safety_qty = safety_days * daily_need
        overstock_qty = (safety_days + repl_days) * daily_need
        days_of_cover = np.where(
            qty <= 0, 0.0,
            np.divide(qty, daily_need, out=np.full(n_days, np.inf),
                      where=daily_need > 0))

        frames.append(pd.DataFrame({
            "item_code": item, "date": dates, "qty": qty, "stock_stage": stage,
            "uom": stock_uom, "daily_need": daily_need,
            "days_of_cover": days_of_cover, "safety_qty": safety_qty,
            "overstock_qty": overstock_qty, "lot_no": lot_no, "supplier": supplier,
        }))

    if not frames:
        return pd.DataFrame(columns=PROJECTION_COLUMNS)
    return pd.concat(frames, ignore_index=True)[PROJECTION_COLUMNS]

def summarize_risk(
    projection: pd.DataFrame, *, within_days: int = 30
) -> pd.DataFrame:
    """Per-item risk summary, sorted by time-to-gap ascending (most urgent first).

    time_to_gap_days   = days from the first projected date until qty first
                         drops to <= 0 (inf if it never does).
    first_below_safety = first date qty < safety_qty (NaT if never).
    at_risk            = hits gap or drops below safety within `within_days`.
    """
    rows = []
    base = projection["date"].min()
    for item, grp in projection.groupby("item_code", sort=False):
        grp = grp.sort_values("date")
        gap_dates = grp.loc[grp["qty"] <= 0, "date"]
        below = grp.loc[grp["qty"] < grp["safety_qty"], "date"]
        gap_date = gap_dates.iloc[0] if not gap_dates.empty else pd.NaT
        below_date = below.iloc[0] if not below.empty else pd.NaT
        ttg = (gap_date - base).days if pd.notna(gap_date) else np.inf
        ttb = (below_date - base).days if pd.notna(below_date) else np.inf
        rows.append({
            "item_code": item,
            "uom": grp["uom"].iloc[0],
            "opening_qty": grp["qty"].iloc[0],
            "first_gap_date": gap_date,
            "time_to_gap_days": ttg,
            "first_below_safety_date": below_date,
            "time_to_below_safety_days": ttb,
            "at_risk": bool(min(ttg, ttb) <= within_days),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["time_to_gap_days", "time_to_below_safety_days", "item_code"]
    ).reset_index(drop=True)

def stage_entry_dates(projection: pd.DataFrame) -> pd.DataFrame:
    """First date each item starts consuming from each stage.

    One row per (item_code, stock_stage) with `first_date` and
    `days_from_start` (days after the projection base date). Feeds the
    per-stage risk windows: 'flag the item if it starts eating from stage X
    within N days'.
    """
    base = projection["date"].min()
    first = (projection.groupby(["item_code", "stock_stage"], sort=False)
             ["date"].min().reset_index(name="first_date"))
    first["days_from_start"] = (first["first_date"] - base).dt.days
    return first

# ------------------------------------------------------------------ internals

def _item_thresholds(data: WorkbookData) -> dict[str, tuple[float, float]]:
    """(safety_level_days, replenishment_level_days) per item.

    The days are defined per (item, output_type, line) in consumption_figs; when
    they differ across combos the max is used (conservative) — in practice they
    are constant per item.
    """
    grp = data.consumption_figs.groupby("item_code")[
        ["safety_level_days", "replenishment_level_days"]].max()
    return {item: (float(r["safety_level_days"]), float(r["replenishment_level_days"]))
            for item, r in grp.fillna(0.0).iterrows()}

def _demand_in_stock_uom(
    demand: pd.DataFrame,
    item: str,
    dates: pd.DatetimeIndex,
    stock_uom: str,
    bom_uom: str,
    unit_wt: float,
) -> np.ndarray:
    """Daily projected demand for `item` expressed in its stock uom, dense/zero-filled."""
    try:
        item_demand = demand.xs(item, level="item_code")
    except KeyError:
        return np.zeros(len(dates))
    item_demand = item_demand.reindex(dates, fill_value=0.0)
    if stock_uom == "ton":
        return item_demand["cons_qty_ton"].to_numpy(dtype=float)
    base = item_demand["cons_qty_base_uom"].to_numpy(dtype=float)
    if stock_uom == bom_uom:
        return base
    return np.asarray(convert(base, bom_uom, stock_uom, unit_wt_kg=unit_wt),
                      dtype=float)

def _buckets_for(
    item: str, opening_qty: float, deliveries: pd.DataFrame, base_date: pd.Timestamp
) -> pd.DataFrame:
    """Supply buckets in FIFO order: warehouse first, then lots by arrival date."""
    lots = deliveries.loc[deliveries["item_code"] == item]
    bucket_rows = [{
        "qty": max(opening_qty, 0.0), "stage": WAREHOUSE_STAGE,
        "available_from": base_date, "lot_no": None, "supplier": None,
    }]
    for _, lot in lots.iterrows():
        bucket_rows.append({
            "qty": float(lot["qty_delivered"]),
            "stage": lot["delivery_label"],
            # past-due ETAs count as on hand from the projection start
            "available_from": max(pd.Timestamp(lot["arrival_date_in_plant"]), base_date),
            "lot_no": lot["shipment_lot_no."],
            "supplier": lot["supplier_name"] if pd.notna(lot["supplier_name"]) else None,
        })
    return pd.DataFrame(bucket_rows)

def _forward_mean(daily: np.ndarray, window: int) -> np.ndarray:
    """Forward-looking `window`-day mean; the window shrinks near the series end
    (never padded with zeros, which would fake an infinite days-of-cover)."""
    n = len(daily)
    cum = np.concatenate([[0.0], np.cumsum(daily)])
    idx = np.arange(n)
    end = np.minimum(idx + window, n)
    span = (end - idx).astype(float)
    return (cum[end] - cum[idx]) / span

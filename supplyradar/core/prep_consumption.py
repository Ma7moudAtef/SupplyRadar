"""Projected consumption builder.

Turns the production PLAN into a daily consumption plan per
(item_code, output_type, production_line), using consumption_figs rates, and
appends it to the actual consumption rows. Output schema matches the
`consumption` sheet exactly.

Contract:
- base_date is supplied by the caller and must lie in
  [stock snapshot date .. max plan date] (raises ValueError outside it).
- horizon_end = last day of the month containing max(prod.date where type='plan').
- Exactly three std_cons_rate_uom branches exist (kg/ton, ton/day, pc/heat).
  Anything else raises UOMError — no conversion is guessed.
- A stock item lacking a consumption_figs row for a combo the plan produces
  raises ContractError (fail loud, never silently produce zeros).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import WorkbookData
from .uom import UOMError, convert, normalize_uom


class ContractError(Exception):
    """A data contract violation that would corrupt downstream numbers."""

CONSUMPTION_COLUMNS = [
    "date", "item_code", "cons_qty_base_uom", "cons_qty_ton", "cons_$",
    "output_type", "production_line", "consumption_type",
]

def horizon_end_for(data: WorkbookData) -> pd.Timestamp:
    """Last day of the month containing the last production-plan date."""
    return data.max_plan_date + pd.offsets.MonthEnd(0)

def check_base_date(data: WorkbookData, base_date: pd.Timestamp) -> None:
    """base_date must lie in [stock snapshot date .. max plan date].

    The opening warehouse quantity is only valid as of the snapshot date;
    projecting from any other date silently corrupts every downstream number.
    """
    lo, hi = data.stock_snapshot_date, data.max_plan_date
    if not (lo <= base_date <= hi):
        raise ValueError(
            f"base_date {base_date.date()} outside the allowed window "
            f"[{lo.date()} .. {hi.date()}] (stock snapshot .. last plan date)"
        )

def build_projected_consumption(
    data: WorkbookData,
    base_date: str | pd.Timestamp,
    *,
    items: list[str] | None = None,
) -> pd.DataFrame:
    """Build plan consumption rows for [base_date .. horizon_end].

    data must already have validation exclusions applied (see
    validation.apply_exclusions). If `items` is None, every item_code in `stock`
    is projected and full combo coverage is enforced.
    """
    base_date = pd.Timestamp(base_date)
    check_base_date(data, base_date)
    horizon_end = horizon_end_for(data)
    dates = pd.date_range(base_date, horizon_end, freq="D")

    stock_items = data.stock["item_code"].dropna().unique().tolist()
    items = stock_items if items is None else list(items)

    cf = data.consumption_figs.loc[data.consumption_figs["item_code"].isin(items)]
    _check_combo_coverage(data, cf, items)

    unknown = set(cf["std_cons_rate_uom"].dropna()) - {"kg/ton", "ton/day", "pc/heat"}
    if unknown:
        raise UOMError(f"unhandled std_cons_rate_uom value(s): {sorted(unknown)}")

    plan = data.prod.loc[
        (data.prod["production_type"] == "plan")
        & data.prod["date"].between(base_date, horizon_end)
    ]
    plan_grid = (
        plan.groupby(["date", "output_type", "production_line"], as_index=False)
        [["production_qty1", "production_qty2"]].sum()
    )

    bom_cols = data.bom.set_index("item_code")[["uom", "unit_wt_kg", "unit_price_$"]]

    parts: list[pd.DataFrame] = []

    # kg/ton: consumption follows the mass produced (production_qty1, tons).
    kg = cf.loc[cf["std_cons_rate_uom"] == "kg/ton"].merge(
        plan_grid, on=["output_type", "production_line"])
    if not kg.empty:
        kg["natural_qty"] = kg["std_cons_rate"] * kg["production_qty1"] / 1000.0
        kg["natural_uom"] = "ton"
        parts.append(kg)

    # pc/heat: consumption follows the number of heats (production_qty2).
    pc = cf.loc[cf["std_cons_rate_uom"] == "pc/heat"].merge(
        plan_grid, on=["output_type", "production_line"])
    if not pc.empty:
        pc["natural_qty"] = pc["std_cons_rate"] * pc["production_qty2"]
        pc["natural_uom"] = "pc"
        parts.append(pc)

    # ton/day: flat per calendar day, independent of the production plan.
    td = cf.loc[cf["std_cons_rate_uom"] == "ton/day"]
    if not td.empty:
        td = td.merge(pd.DataFrame({"date": dates}), how="cross")
        td["natural_qty"] = td["std_cons_rate"]
        td["natural_uom"] = "ton"
        parts.append(td)

    if not parts:
        return pd.DataFrame(columns=CONSUMPTION_COLUMNS)

    proj = pd.concat(parts, ignore_index=True)
    proj = proj.join(bom_cols, on="item_code")

    missing_bom = proj.loc[proj["uom"].isna(), "item_code"].unique()
    if len(missing_bom):
        raise ContractError(
            f"item(s) missing from bom, cannot convert units: {sorted(missing_bom)[:10]}")

    proj["cons_qty_ton"] = _convert_series(proj, proj["natural_uom"], "ton")
    proj["cons_qty_base_uom"] = _convert_series(proj, proj["natural_uom"], proj["uom"])
    proj["cons_$"] = proj["cons_qty_base_uom"] * proj["unit_price_$"].fillna(0.0)
    proj["consumption_type"] = "plan"

    return proj[CONSUMPTION_COLUMNS].sort_values(
        ["item_code", "date", "output_type", "production_line"]).reset_index(drop=True)

def append_to_actuals(data: WorkbookData, projected: pd.DataFrame) -> pd.DataFrame:
    """Actual consumption rows + projected plan rows, one schema."""
    actual = data.consumption[CONSUMPTION_COLUMNS]
    return pd.concat([actual, projected], ignore_index=True)

def _check_combo_coverage(
    data: WorkbookData, cf: pd.DataFrame, items: list[str]
) -> None:
    """Every projected item must have a rate for every combo the plan produces."""
    plan = data.prod.loc[data.prod["production_type"] == "plan"]
    plan_combos = set(map(tuple, plan[["output_type", "production_line"]]
                          .drop_duplicates().values))
    have: dict[str, set] = {
        item: set(map(tuple, grp[["output_type", "production_line"]].values))
        for item, grp in cf.groupby("item_code")
    }
    problems = {}
    for item in items:
        missing = plan_combos - have.get(item, set())
        if missing:
            problems[item] = sorted(missing)
    if problems:
        sample = dict(list(problems.items())[:5])
        raise ContractError(
            f"{len(problems)} item(s) have no consumption_figs rate for combo(s) "
            f"the plan produces — cannot project (refusing to emit silent zeros). "
            f"e.g. {sample}"
        )

def _convert_series(
    df: pd.DataFrame, from_uom: pd.Series, to_uom: pd.Series | str
) -> pd.Series:
    """Vectorized uom.convert over mixed unit groups (one call per unit pair)."""
    if isinstance(to_uom, str):
        to_uom = pd.Series(to_uom, index=df.index)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    pairs = pd.DataFrame({"f": from_uom.map(normalize_uom),
                          "t": to_uom.map(normalize_uom)})
    for (f, t), grp in pairs.groupby(["f", "t"]):
        idx = grp.index
        needs_wt = ("pc" in (f, t)) and f != t
        wt = df.loc[idx, "unit_wt_kg"].to_numpy(dtype=float) if needs_wt else None
        out.loc[idx] = convert(
            df.loc[idx, "natural_qty"].to_numpy(dtype=float), f, t, unit_wt_kg=wt)
    return out

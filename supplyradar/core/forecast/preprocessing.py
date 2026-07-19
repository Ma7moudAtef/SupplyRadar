"""Step 1-2: validation flags + normalized monthly consumption-rate series.

The forecasting variable is the consumption RATE per
(item_code, output_type, production_line), monthly:

    rate(month) = material consumption / production driver

The `consumption` sheet's cons_rate / cons_rate_uom columns are used directly
when present (rows within a month share the month's driver, so the month rate
is the SUM of row rates — netting rows included). For older workbooks without
those columns the rate is derived from monthly actual production in `prod`.
Nothing is deleted: every anomaly becomes a flag the planner can see.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..loader import WorkbookData

DRIVER_BY_UOM = {"kg/ton": "production_qty1", "pc/heat": "production_qty2",
                 "ton/day": "days"}

@dataclass
class RateSeries:
    """One forecastable series: monthly consumption rate for an item-combo."""

    item_code: str
    output_type: str
    production_line: str
    rate_uom: str
    series: pd.Series          # PeriodIndex('M') -> rate (float, NaN = unknown)
    consumption: pd.Series     # monthly material consumption (base uom)
    production: pd.Series       # monthly production driver, aligned to series
    flags: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.item_code, self.output_type, self.production_line)

    @property
    def values(self) -> np.ndarray:
        """Rate values with interior NaN filled by interpolation (flagged)."""
        return self.series.to_numpy(dtype=float)

    @property
    def weights(self) -> np.ndarray:
        """Production quantity per month, aligned to `values` — the weight a
        weighted-average model uses to aggregate the rate."""
        return self.production.to_numpy(dtype=float)

    @property
    def n_valid(self) -> int:
        return int(self.series.notna().sum())

def build_rate_series(data: WorkbookData, *, items: list[str] | None = None
                      ) -> dict[tuple[str, str, str], RateSeries]:
    """Monthly rate series for every (item, output_type, line) with history.

    Combos whose rate uom cannot be established (neither cons_rate_uom nor a
    consumption_figs row) are skipped with a flag — no conversion is guessed.
    """
    cons = data.consumption.loc[
        data.consumption["consumption_type"] == "actual"].copy()
    if items is not None:
        cons = cons.loc[cons["item_code"].isin(items)]
    if cons.empty:
        return {}
    cons["month"] = cons["date"].dt.to_period("M")

    cf_uom = (data.consumption_figs
              .set_index(["item_code", "output_type", "production_line"])
              ["std_cons_rate_uom"].to_dict())
    drivers = _monthly_production_drivers(data)

    has_rate = "cons_rate" in cons.columns
    if not has_rate:
        cons = _derive_rates(cons, data)

    keys = ["item_code", "output_type", "production_line"]
    grouped = (cons.groupby(keys + ["month"])
               .agg(rate=("cons_rate", "sum"),
                    rate_uom=("cons_rate_uom", "first"),
                    consumption=("cons_qty_base_uom", "sum"))
               .reset_index())

    out: dict[tuple[str, str, str], RateSeries] = {}
    for key, grp in grouped.groupby(keys):
        item, otype, line = key
        flags: list[str] = []
        uoms = grp["rate_uom"].dropna().unique()
        rate_uom = uoms[0] if len(uoms) else cf_uom.get(key)
        if len(uoms) > 1:
            flags.append(f"unit_inconsistency: {sorted(uoms)}")
        if rate_uom is None:
            continue  # no way to know what the rate means — skip, don't guess

        months = pd.period_range(grp["month"].min(), grp["month"].max(), freq="M")
        series = grp.set_index("month")["rate"].reindex(months)
        consumption = grp.set_index("month")["consumption"].reindex(months)
        production = _combo_production(drivers, otype, line, rate_uom, months)

        n_missing = int(series.isna().sum())
        if n_missing:
            flags.append(f"missing_periods: {n_missing}")
            # a missing month is most plausibly zero usage; interpolating would
            # invent consumption that never happened
            series = series.fillna(0.0)
            consumption = consumption.fillna(0.0)
        if (series < 0).any():
            flags.append(f"negative_values: {int((series < 0).sum())}")
        med = series.median()
        mad = float(np.median(np.abs(series - med)))
        if mad > 0:
            n_out = int((np.abs(series - med) > 3 * 1.4826 * mad).sum())
            if n_out:
                flags.append(f"extreme_outliers: {n_out}")
        zero_share = float((series == 0).mean())
        if zero_share >= 0.4:
            flags.append(f"intermittent: {zero_share:.0%} zero months")

        out[key] = RateSeries(item_code=item, output_type=otype,
                              production_line=line, rate_uom=rate_uom,
                              series=series, consumption=consumption,
                              production=production, flags=flags)
    return out

def _monthly_production_drivers(data: WorkbookData) -> pd.DataFrame:
    """Actual monthly production per (output_type, production_line): the mass
    (qty1), heat count (qty2) and calendar days behind each month's rate."""
    prod = data.prod.loc[data.prod["production_type"] == "actual"].copy()
    prod["month"] = prod["date"].dt.to_period("M")
    g = (prod.groupby(["output_type", "production_line", "month"])
         [["production_qty1", "production_qty2"]].sum())
    return g

def _combo_production(drivers: pd.DataFrame, otype: str, line: str,
                      rate_uom: str, months: pd.PeriodIndex) -> pd.Series:
    """Production driver for one combo, aligned to `months`. The driver matches
    the rate's denominator: mass for kg/ton, heats for pc/heat, days for
    ton/day (near-uniform, so ton/day stays effectively an unweighted mean)."""
    if rate_uom == "ton/day":
        return pd.Series(months.days_in_month.astype(float), index=months)
    col = "production_qty1" if rate_uom == "kg/ton" else "production_qty2"
    try:
        sub = drivers.loc[(otype, line), col]
    except KeyError:
        return pd.Series(0.0, index=months)
    return sub.reindex(months).fillna(0.0)

def _derive_rates(cons: pd.DataFrame, data: WorkbookData) -> pd.DataFrame:
    """Fallback for workbooks without cons_rate columns: rate from monthly
    actual production (kg/ton via qty1, pc/heat via qty2, ton/day via days)."""
    prod = data.prod.loc[data.prod["production_type"] == "actual"].copy()
    prod["month"] = prod["date"].dt.to_period("M")
    driver = (prod.groupby(["month", "output_type", "production_line"])
              [["production_qty1", "production_qty2"]].sum().reset_index())
    driver["days"] = driver["month"].dt.days_in_month.astype(float)

    cf_uom = (data.consumption_figs
              .set_index(["item_code", "output_type", "production_line"])
              ["std_cons_rate_uom"])
    cons = cons.join(cf_uom.rename("cons_rate_uom"),
                     on=["item_code", "output_type", "production_line"])
    cons = cons.merge(driver, on=["month", "output_type", "production_line"],
                      how="left")

    rate = pd.Series(np.nan, index=cons.index, dtype=float)
    uom = cons["cons_rate_uom"]
    q1, q2 = cons["production_qty1"], cons["production_qty2"]
    days = cons["days"]
    m = uom.eq("kg/ton") & (q1 > 0)
    rate[m] = cons.loc[m, "cons_qty_ton"] * 1000.0 / q1[m]
    m = uom.eq("pc/heat") & (q2 > 0)
    rate[m] = cons.loc[m, "cons_qty_base_uom"] / q2[m]
    m = uom.eq("ton/day") & (days > 0)
    rate[m] = cons.loc[m, "cons_qty_ton"] / days[m]
    cons["cons_rate"] = rate
    return cons.drop(columns=["production_qty1", "production_qty2", "days"])

"""Data validation. Runs BEFORE any calculation.

Produces a ValidationReport of {severity, sheet, rule, row_ids, message} issues.
Nothing is auto-deleted: rows an issue marks with action='drop' are excluded from
computation by apply_exclusions(), counted, and reported — the originals stay in
the loaded WorkbookData.

SEVERITY POLICY: only a broken SCHEMA blocks the app (missing sheet / column and
unparseable dates are raised by the loader; an unknown std_cons_rate_uom is the
one blocking rule here). Everything else is a warning or info: the row is
excluded where needed, counted, and reported. A single bad row must never prevent
a dashboard from rendering.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .loader import SHEET_SCHEMAS, WorkbookData
from .uom import UOMError, normalize_uom

KNOWN_RATE_UOMS = {"kg/ton", "ton/day", "pc/heat"}

Severity = Literal["error", "warning", "info"]

@dataclass
class ValidationIssue:
    severity: Severity
    sheet: str
    rule: str
    row_ids: list[int]
    message: str
    action: Literal["drop", "flag"] = "flag"

@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    @property
    def blocking(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def excluded_row_ids(self, sheet: str) -> set[int]:
        out: set[int] = set()
        for i in self.issues:
            if i.sheet == sheet and i.action == "drop":
                out.update(i.row_ids)
        return out

    def excluded_counts(self) -> dict[str, int]:
        return {s: len(self.excluded_row_ids(s)) for s in SHEET_SCHEMAS
                if self.excluded_row_ids(s)}

    def to_dict(self) -> dict:
        return {
            "blocking": self.blocking,
            "excluded_row_counts": self.excluded_counts(),
            "issues": [asdict(i) for i in self.issues],
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), default=str, **kwargs)

def validate(data: WorkbookData) -> ValidationReport:
    """Run every rule; never raises on data content. See module docstring."""
    report = ValidationReport()
    _rule_rate_uom_known(data, report)
    _rule_item_membership(data, report)
    _rule_unmatched_combos(data, report)
    _rule_duplicate_consumption(data, report)
    _rule_production_qty(data, report)
    _rule_negative_quantities(data, report)
    _rule_uom_conflicts(data, report)
    _rule_unit_weight_missing(data, report)
    _rule_monthly_gaps(data, report)
    _rule_consumption_outliers(data, report)
    _rule_delivery_sanity(data, report)
    return report

def apply_exclusions(data: WorkbookData, report: ValidationReport) -> WorkbookData:
    """Return a copy of the data with action='drop' rows removed (counted in report)."""
    clean = data.copy()
    for sheet in SHEET_SCHEMAS:
        dropped = report.excluded_row_ids(sheet)
        if dropped:
            df = clean.sheet(sheet)
            setattr(clean, sheet, df.loc[~df.index.isin(dropped)])
    return clean

# --------------------------------------------------------------------------- rules

def _rule_rate_uom_known(data: WorkbookData, report: ValidationReport) -> None:
    cf = data.consumption_figs
    bad = cf.loc[~cf["std_cons_rate_uom"].isin(KNOWN_RATE_UOMS)]
    if not bad.empty:
        report.add(ValidationIssue(
            "error", "consumption_figs", "unknown_std_cons_rate_uom",
            bad.index.tolist(),
            f"std_cons_rate_uom outside the known set {sorted(KNOWN_RATE_UOMS)}: "
            f"{sorted(bad['std_cons_rate_uom'].dropna().unique().tolist())}. "
            "No conversion will be guessed.",
        ))

def _rule_item_membership(data: WorkbookData, report: ValidationReport) -> None:
    sets = {s: set(data.sheet(s)["item_code"].dropna()) for s in
            ("stock", "bom", "consumption_figs")}
    pairs = [("stock", "bom"), ("stock", "consumption_figs"),
             ("bom", "stock"), ("consumption_figs", "stock")]
    for a, b in pairs:
        missing = sets[a] - sets[b]
        if missing:
            df = data.sheet(a)
            rows = df.loc[df["item_code"].isin(missing)]
            report.add(ValidationIssue(
                "warning", a, f"item_missing_from_{b}", rows.index.tolist(),
                f"{len(missing)} item_code(s) present in '{a}' but missing from "
                f"'{b}': {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}",
            ))

def _rule_unmatched_combos(data: WorkbookData, report: ValidationReport) -> None:
    known = set(map(tuple, pd.concat([
        data.prod[["output_type", "production_line"]],
        data.consumption_figs[["output_type", "production_line"]],
    ]).drop_duplicates().values))
    combos = data.consumption[["output_type", "production_line"]].apply(tuple, axis=1)
    bad = data.consumption.loc[~combos.isin(known)]
    if not bad.empty:
        unmatched = sorted(set(map(tuple, bad[["output_type", "production_line"]].values)))
        report.add(ValidationIssue(
            "warning", "consumption", "unmatched_output_type_line_combo",
            bad.index.tolist(),
            f"{len(bad)} consumption row(s) with (output_type, production_line) "
            f"combo(s) {unmatched} that exist in neither 'prod' nor "
            "'consumption_figs'. They cannot be normalized and are DROPPED "
            "(no mapping is guessed).",
            action="drop",
        ))

def _rule_duplicate_consumption(data: WorkbookData, report: ValidationReport) -> None:
    keys = ["date", "item_code", "output_type", "production_line"]
    dup = data.consumption.duplicated(subset=keys, keep="first")
    if dup.any():
        # In this data multiple rows per key are transactions (incl. negative
        # adjustments) that NET to the month's figure — deleting them would
        # corrupt history, so they are flagged and summed at the reporting grain.
        report.add(ValidationIssue(
            "warning", "consumption", "duplicate_consumption_key",
            data.consumption.index[dup].tolist(),
            f"{int(dup.sum())} extra row(s) share a (date, item_code, output_type, "
            "production_line) key; kept and summed at the reporting grain "
            "(they net together, e.g. adjustment entries).",
        ))

def _rule_production_qty(data: WorkbookData, report: ValidationReport) -> None:
    prod = data.prod
    bad = prod.loc[prod["production_qty1"].isna() | (prod["production_qty1"] < 0)]
    if not bad.empty:
        report.add(ValidationIssue(
            "warning", "prod", "missing_or_negative_production_qty",
            bad.index.tolist(),
            f"{len(bad)} prod row(s) with missing or negative production_qty1; excluded.",
            action="drop",
        ))
    zero = prod.loc[prod["production_qty1"] == 0]
    if not zero.empty:
        report.add(ValidationIssue(
            "info", "prod", "zero_production_qty", zero.index.tolist(),
            f"{len(zero)} prod row(s) with production_qty1 == 0 "
            "(idle days are normal; kept, they contribute zero consumption).",
        ))

def _rule_negative_quantities(data: WorkbookData, report: ValidationReport) -> None:
    cons = data.consumption
    neg = cons.loc[(cons["cons_qty_base_uom"] < 0) | (cons["cons_qty_ton"] < 0)]
    if not neg.empty:
        # Negative rows are return/adjustment entries that net against positive
        # rows in the same month; excluding them would inflate actuals.
        report.add(ValidationIssue(
            "warning", "consumption", "negative_consumption", neg.index.tolist(),
            f"{len(neg)} consumption row(s) with negative quantity; kept as "
            "adjustment entries (they net against same-month rows).",
        ))
    neg_stock = data.stock.loc[data.stock["qty"] < 0]
    if not neg_stock.empty:
        report.add(ValidationIssue(
            "warning", "stock", "negative_stock", neg_stock.index.tolist(),
            f"{len(neg_stock)} stock row(s) with negative qty; excluded.",
            action="drop",
        ))

def _rule_uom_conflicts(data: WorkbookData, report: ValidationReport) -> None:
    bom = data.bom.set_index("item_code")["uom"]
    merged = data.stock.join(bom.rename("bom_uom"), on="item_code")
    conflict = merged.loc[
        merged["bom_uom"].notna() & (merged["uom"] != merged["bom_uom"])]
    if not conflict.empty:
        items = sorted(conflict["item_code"].unique().tolist())
        report.add(ValidationIssue(
            "warning", "stock", "uom_conflict_stock_vs_bom", conflict.index.tolist(),
            f"{len(items)} item(s) where stock.uom differs from bom.uom "
            f"(e.g. {items[:5]}); quantities are converted via unit_wt_kg.",
        ))

    implied = {"kg/ton": "mass", "ton/day": "mass", "pc/heat": "pc"}
    cf = data.consumption_figs.join(bom.rename("bom_uom"), on="item_code")
    kinds = []
    for u in cf["bom_uom"]:
        try:
            kinds.append("pc" if normalize_uom(u) == "pc" else "mass")
        except UOMError:
            kinds.append(None)
    cf = cf.assign(bom_kind=kinds, implied_kind=cf["std_cons_rate_uom"].map(implied))
    bad = cf.loc[cf["bom_kind"].notna() & cf["implied_kind"].notna()
                 & (cf["bom_kind"] != cf["implied_kind"])]
    if not bad.empty:
        report.add(ValidationIssue(
            "warning", "consumption_figs", "uom_conflict_rate_vs_bom",
            bad.index.tolist(),
            f"{len(bad)} consumption_figs row(s) whose std_cons_rate_uom implies a "
            "different unit kind than bom.uom; converted via unit_wt_kg.",
        ))

def _rule_unit_weight_missing(data: WorkbookData, report: ValidationReport) -> None:
    bom = data.bom
    pc_items = bom.loc[bom["uom"] == "pc"]
    bad = pc_items.loc[pc_items["unit_wt_kg"].isna() | (pc_items["unit_wt_kg"] <= 0)]
    if not bad.empty:
        report.add(ValidationIssue(
            "warning", "bom", "unit_wt_missing_for_pc_item", bad.index.tolist(),
            f"{len(bad)} pc item(s) with missing/zero unit_wt_kg; their "
            "ton-denominated figures are zero (the source data's own convention).",
        ))

def _rule_monthly_gaps(data: WorkbookData, report: ValidationReport) -> None:
    actual = data.consumption.loc[data.consumption["consumption_type"] == "actual"]
    if actual.empty:
        return
    gap_items: list[str] = []
    for item, grp in actual.groupby("item_code"):
        months = pd.PeriodIndex(grp["date"], freq="M").unique()
        expected = pd.period_range(months.min(), months.max(), freq="M")
        if len(expected.difference(months)) > 0:
            gap_items.append(str(item))
    if gap_items:
        report.add(ValidationIssue(
            "info", "consumption", "monthly_series_gaps", [],
            f"{len(gap_items)} item(s) have gaps in their monthly actual series "
            f"(e.g. {gap_items[:5]}). Kept; gaps may be genuine zero-usage months.",
        ))

def _rule_consumption_outliers(data: WorkbookData, report: ValidationReport) -> None:
    actual = data.consumption.loc[data.consumption["consumption_type"] == "actual"]
    if actual.empty:
        return
    out_ids: list[int] = []
    for _, grp in actual.groupby("item_code"):
        vals = grp["cons_qty_ton"].astype(float)
        med = vals.median()
        mad = float(np.median(np.abs(vals - med)))
        if mad == 0:
            continue
        mask = np.abs(vals - med) > 3 * 1.4826 * mad
        out_ids.extend(grp.index[mask].tolist())
    if out_ids:
        report.add(ValidationIssue(
            "warning", "consumption", "consumption_rate_outlier", out_ids,
            f"{len(out_ids)} actual consumption row(s) outside median +/- 3*MAD "
            "for their item. Flagged for review; kept (history is not rewritten).",
        ))

def _rule_delivery_sanity(data: WorkbookData, report: ValidationReport) -> None:
    dlv = data.delivery
    known_items = set(data.stock["item_code"].dropna()) | set(data.bom["item_code"].dropna())
    orphan = dlv.loc[~dlv["item_code"].isin(known_items)]
    if not orphan.empty:
        report.add(ValidationIssue(
            "warning", "delivery", "delivery_item_unknown", orphan.index.tolist(),
            f"{len(orphan)} delivery lot(s) for item_code(s) not present in "
            "'stock'/'bom'; they cannot join a projection lane and are ignored.",
        ))
    bad_qty = dlv.loc[dlv["qty_delivered"].isna() | (dlv["qty_delivered"] <= 0)]
    if not bad_qty.empty:
        report.add(ValidationIssue(
            "warning", "delivery", "non_positive_delivery_qty", bad_qty.index.tolist(),
            f"{len(bad_qty)} delivery lot(s) with missing/zero/negative "
            "qty_delivered; excluded.",
            action="drop",
        ))
    snapshot = data.stock_snapshot_date
    past_due = dlv.loc[dlv["arrival_date_in_plant"] < snapshot]
    if not past_due.empty:
        report.add(ValidationIssue(
            "warning", "delivery", "arrival_before_snapshot", past_due.index.tolist(),
            f"{len(past_due)} delivery lot(s) whose arrival_date_in_plant is before "
            f"the stock snapshot ({snapshot.date()}). They are treated as available "
            "from the projection base date (past-due ETAs).",
        ))

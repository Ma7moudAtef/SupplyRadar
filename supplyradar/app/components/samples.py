"""Sample-data downloads: a top-10-rows CSV per table + a data dictionary.

Gives a new user two files that explain the input format of the working
dataset without shipping the whole workbook:
- sample_data.csv   — for EACH of the 7 tables: a '# table:' marker line, the
                      header row, and the first 10 data rows.
- data_dictionary.txt — what each table is for and what every column means.

Both are built from the currently loaded (normalized) workbook, so the sample
always matches whatever file the user is actually working with.
"""

from __future__ import annotations

import io

from supplyradar.core.loader import SHEET_SCHEMAS, WorkbookData

SAMPLE_ROWS = 10

# Column-by-column explanations of the universal data form. Static because the
# schema is fixed (SHEET_SCHEMAS); per-file row counts are appended at build
# time. Optional columns are marked as such.
_TABLE_DOCS: dict[str, tuple[str, dict[str, str]]] = {
    "prod": (
        "Production output per date, output type and production line. Actual "
        "rows are MONTHLY history; plan rows are DAILY and drive every "
        "projection.",
        {
            "date": "Production date (actuals: first day of the month; plan: "
                    "the exact day).",
            "production_uom1": "Unit of the first production quantity — a mass "
                               "unit that drives mass-based consumption rates.",
            "production_qty1": "Amount produced in uom1 — the driver for "
                               "mass-per-mass (kg/ton) consumption rates.",
            "production_uom2": "Unit of the second production quantity — a "
                               "count / batch unit that drives per-batch "
                               "consumption rates.",
            "production_qty2": "Count produced in uom2 — the driver for "
                               "per-batch (p/s) consumption rates.",
            "output_type": "The product / output family this row produced.",
            "production_line": "Which production line produced it.",
            "production_type": "'actual' (history) or 'plan' (the future "
                               "schedule).",
        }),
    "bom": (
        "The material master: one row per material with its unit, price, "
        "weight and category tree. item_code is the join key to every other "
        "table; item_description is what the app displays.",
        {
            "item_code": "Unique material code — the join key across all "
                         "tables.",
            "item_description": "Human-readable material name (shown in the "
                                "app instead of the code).",
            "uom": "The material's base unit of measure (ton / pc / kg).",
            "unit_price_$": "Price per base unit, in dollars — used for all "
                            "cost figures.",
            "unit_wt_kg": "Weight of one piece in kg — converts between "
                          "pieces and mass.",
            "category_level1": "Top-level material category.",
            "category_level2": "Second-level material category.",
            "category_level3": "Third-level material category.",
        }),
    "consumption_figs": (
        "Standard consumption rates and stock policies per material, output "
        "type and line: how much of the material one unit of production "
        "consumes, plus the safety / replenishment day-counts.",
        {
            "item_code": "Material code (joins bom).",
            "std_cons_rate": "Standard consumption rate value.",
            "std_cons_rate_uom": "Rate unit — one of three branches. Mass "
                                 "branch (k/t, a.k.a. kg/ton): consumption per "
                                 "unit mass produced, driven by qty1. "
                                 "Per-batch branch (p/s): consumption per unit "
                                 "of the second production quantity, driven by "
                                 "qty2. Daily branch (t/d, a.k.a. ton/day): a "
                                 "flat amount per calendar day.",
            "output_type": "Output family the rate applies to.",
            "production_line": "Production line the rate applies to.",
            "daily_need_uom": "Always empty — daily need is DERIVED from the "
                              "projected consumption, never read.",
            "safety_level_days": "Days of cover to hold as safety stock.",
            "replenishment_level_days": "Days of a full replenishment cycle; "
                                        "safety + replenishment days define "
                                        "the overstock level.",
        }),
    "consumption": (
        "Actual material consumption history, monthly, per material, output "
        "type and line. Multiple rows in one month are transactions that NET "
        "together (negative rows are adjustments).",
        {
            "date": "Consumption month (first day of the month).",
            "item_code": "Material code (joins bom).",
            "cons_qty_base_uom": "Consumed quantity in the material's own "
                                 "unit.",
            "cons_qty_ton": "Consumed quantity expressed in tons.",
            "cons_$": "Consumed value in dollars.",
            "output_type": "Output family that consumed it.",
            "production_line": "Line that consumed it.",
            "consumption_type": "'actual' here; the app adds 'plan'/"
                                "'forecast' rows internally.",
            "cons_rate": "OPTIONAL: this row's consumption normalized by "
                         "that month's production — the forecast variable.",
            "cons_rate_uom": "OPTIONAL: unit of cons_rate (same three "
                             "branches as std_cons_rate_uom).",
        }),
    "delivery": (
        "Incoming supply: every purchase lot not yet in the warehouse, with "
        "its pipeline stage and expected arrival. These are the colored "
        "supply buckets on the Pipeline chart.",
        {
            "item_code": "Material code (joins bom).",
            "supplier_name": "Supplier of the lot (may be empty).",
            "shipment_lot_no.": "Lot / shipment reference shown in the "
                                "chart tooltip.",
            "qty_delivered": "Lot quantity, in the material's unit.",
            "arrival_date_in_plant": "Expected arrival date; past-due dates "
                                     "are treated as available immediately.",
            "delivery_label": "The lot's pipeline stage — your own label for "
                              "where the lot sits in the supply pipeline; it "
                              "becomes the fill color of that supply bucket "
                              "on the chart.",
        }),
    "stock": (
        "The opening warehouse snapshot: one row per material with the "
        "quantity on hand at the snapshot date. The projection starts from "
        "these quantities.",
        {
            "item_code": "Material code (joins bom).",
            "uom": "Unit of the quantity (ton / pc / kg).",
            "qty": "Quantity on hand at the snapshot date.",
            "date": "Snapshot date — the earliest allowed projection base "
                    "date.",
            "stock_stage": "Always 'warehouse' in the snapshot.",
        }),
    "suppliers": (
        "The supplier directory per material. Informational in v1 "
        "(lead_time_days is unused).",
        {
            "item_code": "Material code (joins bom).",
            "description": "Material description as the supplier knows it.",
            "supplier_name": "Supplier company name.",
            "lead_time_days": "Reserved for a future release — currently "
                              "empty and ignored.",
        }),
}

def _display_spelling(df, display_names: dict[str, str]):
    """Render normalized label cells back in the file's own spelling.

    The loader lower-cases/aliases labels (e.g. 'k/t' -> 'kg/ton', 'Ton' ->
    'ton') for its internal joins; `display_names` remembers the original
    spelling for each. Mapping the sample's text cells back through it makes
    the downloaded CSV an exact echo of the user's workbook — same unit
    spellings (k/t, p/s, t/d, t/p/k) and casing — not the engine's canon.
    """
    if not display_names:
        return df
    out = df.copy()
    # The per-cell isinstance guard passes numeric/date cells through
    # untouched, so this is safe to run on every column regardless of dtype
    # (label columns land as 'str'/'object', not just 'object').
    for col in out.columns:
        out[col] = out[col].map(
            lambda v: display_names.get(v, v) if isinstance(v, str) else v)
    return out

def build_sample_csv(data: WorkbookData) -> str:
    """One CSV: per table a '# table:' marker, header, and first 10 rows.

    Label cells echo the workbook's own spelling (via the display-name map),
    so the sample matches the source file rather than the normalized form.
    """
    out = io.StringIO()
    out.write("# SupplyRadar sample data — first "
              f"{SAMPLE_ROWS} rows of each table\n")
    for sheet in SHEET_SCHEMAS:
        df = data.sheet(sheet)
        out.write(f"\n# ===== table: {sheet} ({len(df)} rows total) =====\n")
        _display_spelling(df.head(SAMPLE_ROWS), data.display_names).to_csv(
            out, index=False)
    return out.getvalue()

def build_data_dictionary(data: WorkbookData) -> str:
    """Plain-text dictionary: every table, every column, what it does."""
    lines = [
        "SUPPLYRADAR — DATA DICTIONARY",
        "=" * 60,
        "The input workbook has 7 sheets ('tables'). item_code is the key",
        "that connects them. All text matching is case- and whitespace-",
        "insensitive. Below: each table, its purpose, and every column.",
        "",
    ]
    for sheet in SHEET_SCHEMAS:
        purpose, columns = _TABLE_DOCS[sheet]
        df = data.sheet(sheet)
        lines.append(f"TABLE: {sheet}  ({len(df)} rows in the current file)")
        lines.append("-" * 60)
        lines.append(purpose)
        lines.append("")
        for col, doc in columns.items():
            marker = "" if col in df.columns else "  [not in this file]"
            lines.append(f"  {col}{marker}")
            lines.append(f"      {doc}")
        lines.append("")
    return "\n".join(lines)

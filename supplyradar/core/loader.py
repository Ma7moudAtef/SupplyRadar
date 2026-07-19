"""Workbook loading and normalization.

Reads the 7-sheet universal data form into typed DataFrames.

Contract:
- Every key/label column (item_code, uom, output_type, production_line,
  stock_stage, delivery_label, *_type, std_cons_rate_uom, supplier_name) is
  normalized with strip().lower() at load time. A display-name map preserves the
  original casing for the UI.
- Dates are parsed to datetime64. An unparseable date is a SCHEMA error (blocks).
- A missing sheet or missing required column is a SCHEMA error (blocks).
- No rows are dropped here; validation.py decides what to exclude.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import pandas as pd


class SchemaError(Exception):
    """A structural problem that makes the workbook unusable (blocks the app)."""

# Required columns per sheet, exactly as the universal data form defines them today.
SHEET_SCHEMAS: dict[str, list[str]] = {
    "prod": [
        "date", "production_uom1", "production_qty1", "production_uom2",
        "production_qty2", "output_type", "production_line", "production_type",
    ],
    "bom": [
        "item_code", "item_description", "uom", "unit_price_$", "unit_wt_kg",
        "category_level1", "category_level2", "category_level3",
    ],
    "consumption_figs": [
        "item_code", "std_cons_rate", "std_cons_rate_uom", "output_type",
        "production_line", "daily_need_uom", "safety_level_days",
        "replenishment_level_days",
    ],
    "consumption": [
        "date", "item_code", "cons_qty_base_uom", "cons_qty_ton", "cons_$",
        "output_type", "production_line", "consumption_type",
    ],
    "delivery": [
        "item_code", "supplier_name", "shipment_lot_no.", "qty_delivered",
        "arrival_date_in_plant", "delivery_label",
    ],
    "stock": ["item_code", "uom", "qty", "date", "stock_stage"],
    "suppliers": ["item_code", "description", "supplier_name", "lead_time_days"],
}

# Columns whose values are join keys / labels -> normalized strip().lower().
NORMALIZED_COLUMNS: dict[str, list[str]] = {
    "prod": ["production_uom1", "production_uom2", "output_type",
             "production_line", "production_type"],
    "bom": ["item_code", "uom"],
    "consumption_figs": ["item_code", "std_cons_rate_uom", "output_type",
                         "production_line"],
    "consumption": ["item_code", "output_type", "production_line", "consumption_type"],
    "delivery": ["item_code", "supplier_name", "delivery_label"],
    "stock": ["item_code", "uom", "stock_stage"],
    "suppliers": ["item_code", "supplier_name"],
}

# Columns that newer workbooks may carry; absent in older ones, never required.
OPTIONAL_COLUMNS: dict[str, dict[str, str]] = {
    # sheet -> {column: kind}, kind in {"numeric", "label"}
    "consumption": {"cons_rate": "numeric", "cons_rate_uom": "label"},
}

# Known spellings of the three consumption-rate units, mapped to canonical.
# The abbreviated forms were VERIFIED against the long-form workbook (identical
# rate values row-for-row: k/t<->kg/ton, p/s<->pc/heat, t/d<->ton/day) — this
# is a spelling table like 'Ton'/'ton', not an invented mapping. Anything not
# listed here still fails validation's unknown-rate-uom blocking rule.
RATE_UOM_ALIASES: dict[str, str] = {
    "kg/ton": "kg/ton", "k/t": "kg/ton", "kg/t": "kg/ton",
    "pc/heat": "pc/heat", "p/s": "pc/heat", "pc/h": "pc/heat",
    "ton/day": "ton/day", "t/d": "ton/day", "ton/d": "ton/day",
}

# Which (sheet, column) pairs hold a rate unit and get the alias mapping.
RATE_UOM_COLUMNS: dict[str, list[str]] = {
    "consumption_figs": ["std_cons_rate_uom"],
    "consumption": ["cons_rate_uom"],
}

DATE_COLUMNS: dict[str, list[str]] = {
    "prod": ["date"],
    "consumption": ["date"],
    "delivery": ["arrival_date_in_plant"],
    "stock": ["date"],
}

NUMERIC_COLUMNS: dict[str, list[str]] = {
    "prod": ["production_qty1", "production_qty2"],
    "bom": ["unit_price_$", "unit_wt_kg"],
    "consumption_figs": ["std_cons_rate", "safety_level_days", "replenishment_level_days"],
    "consumption": ["cons_qty_base_uom", "cons_qty_ton", "cons_$"],
    "delivery": ["qty_delivered"],
    "stock": ["qty"],
    "suppliers": ["lead_time_days"],
}

@dataclass
class WorkbookData:
    """Typed container for the 7 normalized sheets."""

    prod: pd.DataFrame
    bom: pd.DataFrame
    consumption_figs: pd.DataFrame
    consumption: pd.DataFrame
    delivery: pd.DataFrame
    stock: pd.DataFrame
    suppliers: pd.DataFrame
    display_names: dict[str, str] = field(default_factory=dict)
    source: dict = field(default_factory=dict)

    def sheet(self, name: str) -> pd.DataFrame:
        if name not in SHEET_SCHEMAS:
            raise KeyError(name)
        return getattr(self, name)

    def copy(self) -> WorkbookData:
        return WorkbookData(
            **{s: self.sheet(s).copy() for s in SHEET_SCHEMAS},
            display_names=dict(self.display_names),
            source=dict(self.source),
        )

    @property
    def stock_snapshot_date(self) -> pd.Timestamp:
        """The date of the opening stock snapshot (max date in `stock`)."""
        return pd.Timestamp(self.stock["date"].max())

    @property
    def max_plan_date(self) -> pd.Timestamp:
        """Last date covered by the production plan."""
        plan = self.prod.loc[self.prod["production_type"] == "plan", "date"]
        if plan.empty:
            raise SchemaError("prod sheet contains no production_type='plan' rows")
        return pd.Timestamp(plan.max())

    def display(self, normalized: str) -> str:
        """Original casing for a normalized key/label (falls back to the key)."""
        return self.display_names.get(normalized, normalized)

def _normalize_value(value: object) -> object:
    """strip().lower() a key value; integral floats become '1' not '1.0'."""
    if pd.isna(value):
        return value
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip().lower()

def load_workbook(source: str | Path | IO[bytes]) -> WorkbookData:
    """Read, validate schema, and normalize the universal data form.

    Raises SchemaError on: missing sheet, missing required column, or a
    non-null date value that cannot be parsed.
    """
    raw = pd.read_excel(source, sheet_name=None)
    raw = {str(name).strip().lower(): df for name, df in raw.items()}

    missing_sheets = [s for s in SHEET_SCHEMAS if s not in raw]
    if missing_sheets:
        raise SchemaError(f"missing sheet(s): {missing_sheets}")

    display_names: dict[str, str] = {}
    sheets: dict[str, pd.DataFrame] = {}
    for name, required in SHEET_SCHEMAS.items():
        df = raw[name].copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise SchemaError(f"sheet '{name}' is missing column(s): {missing_cols}")

        for col in DATE_COLUMNS.get(name, []):
            parsed = pd.to_datetime(df[col], errors="coerce")
            bad = parsed.isna() & df[col].notna()
            if bad.any():
                raise SchemaError(
                    f"sheet '{name}' column '{col}': {int(bad.sum())} unparseable "
                    f"date(s), e.g. {df.loc[bad, col].iloc[0]!r}"
                )
            df[col] = parsed

        for col in NUMERIC_COLUMNS.get(name, []):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        label_cols = list(NORMALIZED_COLUMNS.get(name, []))
        for col, kind in OPTIONAL_COLUMNS.get(name, {}).items():
            if col not in df.columns:
                continue
            if kind == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                label_cols.append(col)
        for col in label_cols:
            normalized = df[col].map(_normalize_value)
            for norm, orig in zip(normalized, df[col]):
                if isinstance(norm, str) and norm not in display_names and pd.notna(orig):
                    display_names[norm] = str(orig).strip()
            df[col] = normalized
        # rate units: map verified alternate spellings (k/t, p/s, t/d, …) to
        # the three canonical branches; unknown spellings pass through so the
        # validation blocking rule still catches them
        for col in RATE_UOM_COLUMNS.get(name, []):
            if col in df.columns:
                df[col] = df[col].map(
                    lambda v: RATE_UOM_ALIASES.get(v, v) if pd.notna(v) else v)

        df.index.name = "row_id"
        sheets[name] = df

    meta = {"path": str(source) if isinstance(source, (str, Path)) else "<buffer>",
            "row_counts": {s: len(df) for s, df in sheets.items()}}
    return WorkbookData(**sheets, display_names=display_names, source=meta)

def workbook_fingerprint(source: str | Path | IO[bytes]) -> str:
    """Stable content hash used as a cache key."""
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    else:
        pos = source.tell()
        data = source.read()
        source.seek(pos)
    return hashlib.sha256(data).hexdigest()

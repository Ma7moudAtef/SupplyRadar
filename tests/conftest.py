"""Small synthetic fixtures — never the real workbook."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supplyradar.core.loader import WorkbookData


def make_workbook_frames() -> dict[str, pd.DataFrame]:
    """A 2-item, 3-plan-day universal data form with hand-computable numbers.

    ITEM1 'it-1' (ton, kg/ton rate 2)   : 2 ton/day of plan consumption.
    ITEM2 'it-2' (pc, pc/heat rate 0.4) : 2 pc/day of plan consumption.
    """
    prod = pd.DataFrame({
        "date": (list(pd.to_datetime(["2025-12-01"]))
                 + list(pd.date_range("2026-01-01", periods=3))),
        "production_uom1": ["ms"] * 4,
        "production_qty1": [30000.0, 1000.0, 1000.0, 1000.0],
        "production_uom2": ["heat"] * 4,
        "production_qty2": [150.0, 5.0, 5.0, 5.0],
        "output_type": ["f"] * 4,
        "production_line": ["1"] * 4,
        "production_type": ["actual", "plan", "plan", "plan"],
    })
    bom = pd.DataFrame({
        "item_code": ["it-1", "it-2"],
        "item_description": ["Item One", "Item Two"],
        "uom": ["ton", "pc"],
        "unit_price_$": [100.0, 50.0],
        "unit_wt_kg": [1000.0, 20.0],
        "category_level1": ["Raw Material", "Refractory"],
        "category_level2": ["A", "B"],
        "category_level3": ["a", "b"],
    })
    consumption_figs = pd.DataFrame({
        "item_code": ["it-1", "it-2"],
        "std_cons_rate": [2.0, 0.4],
        "std_cons_rate_uom": ["kg/ton", "pc/heat"],
        "output_type": ["f", "f"],
        "production_line": ["1", "1"],
        "daily_need_uom": [None, None],
        "safety_level_days": [2.0, 5.0],
        "replenishment_level_days": [3.0, 5.0],
    })
    months = pd.date_range("2025-05-01", periods=8, freq="MS")
    it1_rates = [2.0, 2.2, 1.9, 2.1, 2.0, 2.3, 2.1, 2.0]  # kg/ton
    it2_rates = [0.4] * 8                                   # pc/heat
    consumption = pd.DataFrame({
        "date": list(months) * 2,
        "item_code": ["it-1"] * 8 + ["it-2"] * 8,
        "cons_qty_base_uom": [r * 30 for r in it1_rates] + [300.0] * 8,
        "cons_qty_ton": [r * 30 for r in it1_rates] + [6.0] * 8,
        "cons_$": [6000.0] * 8 + [15000.0] * 8,
        "output_type": ["f"] * 16,
        "production_line": ["1"] * 16,
        "consumption_type": ["actual"] * 16,
        "cons_rate": it1_rates + it2_rates,
        "cons_rate_uom": ["kg/ton"] * 8 + ["pc/heat"] * 8,
    })
    delivery = pd.DataFrame({
        "item_code": ["it-1"],
        "supplier_name": [None],
        "shipment_lot_no.": ["lot-1"],
        "qty_delivered": [20.0],
        "arrival_date_in_plant": pd.to_datetime(["2026-01-02"]),
        "delivery_label": ["on ship"],
    })
    stock = pd.DataFrame({
        "item_code": ["it-1", "it-2"],
        "uom": ["ton", "pc"],
        "qty": [100.0, 10.0],
        "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        "stock_stage": ["warehouse", "warehouse"],
    })
    suppliers = pd.DataFrame({
        "item_code": ["it-1"],
        "description": ["Item One"],
        "supplier_name": ["acme"],
        "lead_time_days": [None],
    })
    return {
        "prod": prod, "bom": bom, "consumption_figs": consumption_figs,
        "consumption": consumption, "delivery": delivery, "stock": stock,
        "suppliers": suppliers,
    }


@pytest.fixture
def frames() -> dict[str, pd.DataFrame]:
    return make_workbook_frames()


@pytest.fixture
def data(frames) -> WorkbookData:
    return WorkbookData(**frames)


@pytest.fixture
def workbook_path(tmp_path, frames) -> Path:
    """The same fixture written to a real xlsx (with messy casing) for loader tests."""
    messy = {k: df.copy() for k, df in frames.items()}
    messy["bom"]["uom"] = ["Ton", "PC"]
    messy["bom"]["item_code"] = ["IT-1", "it-2"]
    messy["stock"]["uom"] = ["ton", "pc"]
    messy["stock"]["item_code"] = ["IT-1 ", "it-2"]
    messy["prod"]["production_type"] = ["Actual", "plan", "PLAN ", "plan"]
    path = tmp_path / "form.xlsx"
    with pd.ExcelWriter(path) as writer:
        for name, df in messy.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return path

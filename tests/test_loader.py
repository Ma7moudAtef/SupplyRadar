import pandas as pd
import pytest

from supplyradar.core.loader import SchemaError, load_workbook


def test_loads_and_normalizes(workbook_path):
    data = load_workbook(workbook_path)
    # casing/whitespace normalized at load time
    assert set(data.stock["item_code"]) == {"it-1", "it-2"}
    assert set(data.bom["uom"]) == {"ton", "pc"}
    assert set(data.prod["production_type"]) == {"actual", "plan"}
    # display map keeps original casing
    assert data.display_names["it-1"] == "IT-1"
    assert data.display_names["ton"] == "Ton"
    # dates parsed
    assert pd.api.types.is_datetime64_any_dtype(data.prod["date"])
    assert data.stock_snapshot_date == pd.Timestamp("2026-01-01")
    assert data.max_plan_date == pd.Timestamp("2026-01-03")
    # optional cons_rate columns: numeric + normalized when present
    assert pd.api.types.is_float_dtype(data.consumption["cons_rate"])
    assert set(data.consumption["cons_rate_uom"].dropna()) <= {
        "kg/ton", "pc/heat", "ton/day"}


def test_missing_sheet_blocks(tmp_path, frames):
    path = tmp_path / "broken.xlsx"
    with pd.ExcelWriter(path) as writer:
        for name, df in frames.items():
            if name != "stock":
                df.to_excel(writer, sheet_name=name, index=False)
    with pytest.raises(SchemaError, match="stock"):
        load_workbook(path)


def test_missing_column_blocks(tmp_path, frames):
    frames["bom"] = frames["bom"].drop(columns=["unit_wt_kg"])
    path = tmp_path / "broken.xlsx"
    with pd.ExcelWriter(path) as writer:
        for name, df in frames.items():
            df.to_excel(writer, sheet_name=name, index=False)
    with pytest.raises(SchemaError, match="unit_wt_kg"):
        load_workbook(path)


def test_unparseable_date_blocks(tmp_path, frames):
    frames["stock"] = frames["stock"].astype({"date": str})
    frames["stock"].loc[0, "date"] = "not a date"
    path = tmp_path / "broken.xlsx"
    with pd.ExcelWriter(path) as writer:
        for name, df in frames.items():
            df.to_excel(writer, sheet_name=name, index=False)
    with pytest.raises(SchemaError, match="date"):
        load_workbook(path)

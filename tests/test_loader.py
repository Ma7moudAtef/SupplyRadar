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


def test_rate_uom_aliases_map_to_canonical(tmp_path, frames):
    """Abbreviated rate units (k/t, p/s, t/d) are a VERIFIED relabel of the
    three canonical branches and must normalize to them; plain single-letter
    uoms (t/p/k) likewise. Unknown spellings still pass through untouched so
    validation's blocking rule catches them."""
    frames["consumption_figs"]["std_cons_rate_uom"] = ["k/t", "p/s"]
    frames["consumption"]["cons_rate_uom"] = (
        ["k/t"] * 8 + ["p/s"] * 8)
    frames["bom"]["uom"] = ["T", "P"]
    frames["stock"]["uom"] = ["t", "p"]
    path = tmp_path / "aliased.xlsx"
    with pd.ExcelWriter(path) as writer:
        for name, df in frames.items():
            df.to_excel(writer, sheet_name=name, index=False)
    data = load_workbook(path)
    assert data.consumption_figs["std_cons_rate_uom"].tolist() == [
        "kg/ton", "pc/heat"]
    assert set(data.consumption["cons_rate_uom"]) == {"kg/ton", "pc/heat"}
    # plain uoms normalize via the uom module at use time
    from supplyradar.core.uom import normalize_uom
    assert [normalize_uom(u) for u in data.stock["uom"]] == ["ton", "pc"]

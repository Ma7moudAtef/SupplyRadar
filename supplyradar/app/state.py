"""Streamlit state and cached computation. The ONLY module that may bridge
streamlit and core/viz. All heavy work is cached on (file hash, base_date);
widget interactions never trigger a recompute.

Also owns the planner's forecast SWITCH: which items drive the stock
projection from the AMCIP forecast instead of the standard plan rates.
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # streamlit runs pages from their own dir
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st
import yaml

from supplyradar.core.forecast import (
    ItemForecast,
    forecast_projected_consumption,
    run_all_items,
    run_item_forecast,
)
from supplyradar.core.loader import WorkbookData, load_workbook
from supplyradar.core.prep_consumption import append_to_actuals, build_projected_consumption
from supplyradar.core.prep_stock import build_stock_projection, stage_entry_dates, summarize_risk
from supplyradar.core.validation import ValidationReport, apply_exclusions, validate
from supplyradar.viz.pipeline_chart import palette_from_delivery

CONFIG_DIR = REPO_ROOT / "config"

def load_defaults() -> dict:
    path = CONFIG_DIR / "defaults.yaml"
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    return {}

def load_palette_config() -> tuple[dict, list]:
    path = CONFIG_DIR / "palette.yaml"
    if path.exists():
        cfg = yaml.safe_load(path.read_text()) or {}
        return cfg.get("stages") or {}, cfg.get("delivery_cycle") or []
    return {}, []

@st.cache_data(show_spinner="Reading workbook…")
def load_and_validate(file_bytes: bytes) -> tuple[WorkbookData, ValidationReport]:
    data = load_workbook(io.BytesIO(file_bytes))
    return data, validate(data)

@st.cache_data(show_spinner="Projecting consumption and stock…")
def compute_projection(file_bytes: bytes, base_date_iso: str) -> dict:
    """Full prep pipeline, cached on (file content, base_date)."""
    data, report = load_and_validate(file_bytes)
    clean = apply_exclusions(data, report)
    base_date = pd.Timestamp(base_date_iso)
    projected = build_projected_consumption(clean, base_date)
    stages, cycle = load_palette_config()
    return {
        "fingerprint": hashlib.sha256(file_bytes).hexdigest(),
        "clean": clean,
        "display_names": data.display_names,
        "projected": projected,
        "full_consumption": append_to_actuals(clean, projected),
        "projection": build_stock_projection(clean, projected, base_date),
        "palette": palette_from_delivery(clean.delivery, stages or None,
                                         cycle or None),
        "base_date": base_date,
    }

def get_bundle() -> dict | None:
    """The computed bundle for the current workbook/base_date, or None."""
    return st.session_state.get("sr_bundle")

def require_data() -> dict | None:
    """Common empty-state guard for pages. Returns the bundle or renders help."""
    bundle = get_bundle()
    if bundle is None:
        st.info("No data loaded yet. Use **Data** in the sidebar: upload the "
                "workbook (input_data_form.xlsx) and pick a base date.")
    return bundle

# ------------------------------------------------------- item display labels

def item_labels(bundle: dict) -> dict[str, str]:
    """item_code -> what planners see: the bom item_description.

    item_code stays the join key everywhere; only the display changes.
    Items missing from bom (history-only codes) fall back to their code, and
    duplicated descriptions are disambiguated with the code.
    """
    bom = bundle["clean"].bom
    desc = bom.set_index("item_code")["item_description"].astype(str).str.strip()
    dup = desc.duplicated(keep=False)
    labels = {}
    display = bundle["display_names"]
    for code, d in desc.items():
        base = d if d and d.lower() != "nan" else display.get(code, code)
        labels[code] = f"{base} [{display.get(code, code)}]" if dup.get(code) \
            else base
    return labels

def label_of(bundle: dict, code: str) -> str:
    return item_labels(bundle).get(code, bundle["display_names"].get(code, code))

# ------------------------------------------------------- forecast + switch

@st.cache_data(show_spinner="Running the forecast competition…")
def cached_item_forecast(fingerprint: str, item_code: str, horizon: int,
                         override_model: str | None, override_window: int | None,
                         _data: WorkbookData) -> ItemForecast:
    return run_item_forecast(_data, item_code, horizon,
                             override_model=override_model,
                             override_window=override_window)

@st.cache_data(show_spinner="Running the competition for every item…")
def cached_run_all(fingerprint: str, horizon: int,
                   _data: WorkbookData) -> pd.DataFrame:
    return run_all_items(_data, horizon)

def switched_forecasts() -> dict[str, ItemForecast]:
    """item_code -> the ItemForecast the planner switched the projection to."""
    return st.session_state.setdefault("sr_switched", {})

def set_switch(item_code: str, forecast: ItemForecast | None) -> None:
    switched = switched_forecasts()
    if forecast is None:
        switched.pop(item_code, None)
    else:
        switched[item_code] = forecast
    st.session_state.pop("sr_effective", None)  # invalidate

def effective_tables(bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(projected_consumption, stock_projection) honoring the switch.

    Items switched to forecast contribute AMCIP-driven rows
    (consumption_type='forecast'); everything else keeps the standard plan
    rows. With no switches this is exactly the cached default.
    """
    switched = switched_forecasts()
    if not switched:
        return bundle["projected"], bundle["projection"]

    cache_key = (bundle["fingerprint"], str(bundle["base_date"].date()),
                 tuple(sorted((i, f.horizon_months, f.generated_at)
                              for i, f in switched.items())))
    cached = st.session_state.get("sr_effective")
    if cached is not None and cached[0] == cache_key:
        return cached[1], cached[2]

    clean = bundle["clean"]
    base_date = bundle["base_date"]
    default = bundle["projected"]
    kept = default.loc[~default["item_code"].isin(switched)]
    forecast_rows = forecast_projected_consumption(clean, switched, base_date)
    projected = pd.concat([kept, forecast_rows], ignore_index=True)
    projection = build_stock_projection(clean, projected, base_date)
    st.session_state["sr_effective"] = (cache_key, projected, projection)
    return projected, projection

# ------------------------------------------------------- risk

def risk_tables(bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summarize_risk output, stage_entry_dates) on the EFFECTIVE projection."""
    _, projection = effective_tables(bundle)
    key = (bundle["fingerprint"], str(bundle["base_date"].date()),
           tuple(sorted(switched_forecasts())))
    cached = st.session_state.get("sr_risk_tables")
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    risk = summarize_risk(projection)
    entries = stage_entry_dates(projection)
    st.session_state["sr_risk_tables"] = (key, risk, entries)
    return risk, entries

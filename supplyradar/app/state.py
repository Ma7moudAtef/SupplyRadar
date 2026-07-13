"""Streamlit state and cached computation. The ONLY module that may bridge
streamlit and core/viz. All heavy work is cached on (file hash, base_date);
widget interactions never trigger a recompute.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # streamlit runs pages from their own dir
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st
import yaml

from supplyradar.core.loader import WorkbookData, load_workbook
from supplyradar.core.prep_consumption import append_to_actuals, build_projected_consumption
from supplyradar.core.prep_stock import build_stock_projection, summarize_risk
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

def get_risk(within_days: int) -> pd.DataFrame | None:
    bundle = get_bundle()
    if bundle is None:
        return None
    key = f"sr_risk_{within_days}"
    if key not in st.session_state:
        st.session_state[key] = summarize_risk(
            bundle["projection"], within_days=within_days)
    return st.session_state[key]

def require_data() -> dict | None:
    """Common empty-state guard for pages. Returns the bundle or renders help."""
    bundle = get_bundle()
    if bundle is None:
        st.info("No data loaded yet. Use **Data** in the sidebar: upload the "
                "workbook (input_data_form.xlsx) and pick a base date.")
    return bundle

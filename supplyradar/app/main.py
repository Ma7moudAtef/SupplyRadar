"""SupplyRadar entry point.

STARTUP FLOW — the only two inputs the user provides are the workbook and
base_date. Everything else is derived. base_date is bounded by the stock
snapshot date and the last plan date; outside that window the opening
warehouse quantity would silently corrupt every downstream number.

Run with:  streamlit run supplyradar/app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from supplyradar.app import state
from supplyradar.app.components.validation_panel import render_validation_panel
from supplyradar.core.loader import SchemaError

st.set_page_config(page_title="SupplyRadar", page_icon="📡", layout="wide")

def _data_sidebar() -> None:
    """Workbook + base_date intake; computes and stores the bundle."""
    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader(
        "Universal data form (.xlsx)", type=["xlsx"],
        help="The 7-sheet input_data_form workbook.")
    path_text = st.sidebar.text_input(
        "…or a path on this machine", placeholder="/data/input_data_form.xlsx")

    file_bytes: bytes | None = None
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
    elif path_text.strip():
        p = Path(path_text.strip()).expanduser()
        if p.exists():
            file_bytes = p.read_bytes()
        else:
            st.sidebar.error(f"No file at {p}")

    if file_bytes is None:
        st.session_state.pop("sr_bundle", None)
        return

    try:
        data, report = state.load_and_validate(file_bytes)
    except SchemaError as exc:
        st.sidebar.error(f"Schema error — cannot use this workbook: {exc}")
        st.session_state.pop("sr_bundle", None)
        st.session_state["sr_schema_error"] = str(exc)
        return

    snapshot = data.stock_snapshot_date.date()
    max_plan = data.max_plan_date.date()
    base_date = st.sidebar.date_input(
        "Base date (projection start)", value=snapshot,
        min_value=snapshot, max_value=max_plan,
        help=f"Allowed window: {snapshot} (stock snapshot) … {max_plan} "
             "(last plan date).")
    if not (snapshot <= base_date <= max_plan):
        st.sidebar.error(
            f"Base date must be between {snapshot} and {max_plan}. The opening "
            "warehouse quantity is only valid as of the snapshot date; "
            "projecting from any other date corrupts every downstream number.")
        return
    if base_date > snapshot:
        st.sidebar.warning(
            f"Opening stock is dated {snapshot}; consumption between "
            f"{snapshot} and {base_date} is assumed already reflected.")

    st.session_state["sr_report"] = report
    if not report.blocking:
        bundle = state.compute_projection(file_bytes, base_date.isoformat())
        st.session_state["sr_bundle"] = bundle
        st.sidebar.success(
            f"{bundle['projection']['item_code'].nunique()} items projected "
            f"{base_date} → "
            f"{pd.Timestamp(bundle['projection']['date'].max()).date()}")
    else:
        st.session_state.pop("sr_bundle", None)

_data_sidebar()

pages_dir = Path(__file__).parent / "pages"
nav = st.navigation([
    st.Page(pages_dir / "1_pipeline.py", title="Pipeline Status", icon="📡",
            default=True),
    st.Page(pages_dir / "2_consumption.py", title="Consumption History", icon="📊"),
    st.Page(pages_dir / "3_forecast.py", title="Forecast", icon="🔮"),
    st.Page(pages_dir / "4_cost.py", title="Cost", icon="💰"),
])

report = st.session_state.get("sr_report")
if report is not None and report.blocking:
    render_validation_panel(report)
else:
    nav.run()

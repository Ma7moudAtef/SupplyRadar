"""SupplyRadar entry point.

STARTUP FLOW — the only two inputs the user provides are the workbook and
base_date. Everything else is derived. base_date is bounded by the stock
snapshot date and the last plan date; outside that window the opening
warehouse quantity would silently corrupt every downstream number.

PERFORMANCE — this script reruns on every widget interaction on any page, so
the sidebar must be near-free on the rerun path:
- the workbook bytes are read + sha256-fingerprinted ONCE per new file and
  kept in session_state (no per-click disk read or 2 MB hash);
- load/validate/project run ONLY when (fingerprint, base_date) actually
  changed — otherwise the stored bundle is reused untouched;
- the data intake itself sits in an Apply form ("Load data"), so changing the
  date or picking a file does nothing until the user confirms (or the
  auto-apply toggle is on).

Run with:  streamlit run supplyradar/app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from supplyradar.app import state
from supplyradar.app.components import apply
from supplyradar.app.components.validation_panel import render_validation_panel
from supplyradar.core.loader import SchemaError

st.set_page_config(page_title="SupplyRadar", page_icon="📡", layout="wide")

# Bundled sample workbook. Every user sees this dataset until they upload (or
# point at) their own — the upload always takes priority over the default.
DEFAULT_WORKBOOK_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "default_input_public.xlsx")

def _resolve_file(uploaded, path_text: str) -> tuple[bytes | None, bool]:
    """Turn the intake widgets into workbook bytes.

    Priority: upload > explicit path > bundled default. Returns
    (bytes_or_None, using_default). Only called when the intake form was
    applied or on the very first run — never on ordinary reruns.
    """
    if uploaded is not None:
        return uploaded.getvalue(), False
    if path_text.strip():
        p = Path(path_text.strip()).expanduser()
        if p.exists():
            return p.read_bytes(), False
        st.sidebar.error(f"No file at {p}")
        return None, False
    if DEFAULT_WORKBOOK_PATH.exists():
        return DEFAULT_WORKBOOK_PATH.read_bytes(), True
    return None, False

def _compute_if_changed(file_bytes: bytes, base_date) -> None:
    """(Re)compute the bundle ONLY when (workbook, base_date) changed.

    The fingerprint is computed here — once per applied intake — and the
    (fingerprint, base_date) pair is remembered in session_state, so ordinary
    reruns skip everything including the cache lookups.
    """
    fingerprint = state.fingerprint_bytes(file_bytes)
    key = (fingerprint, str(base_date))
    if st.session_state.get("sr_bundle_key") == key and \
            st.session_state.get("sr_bundle") is not None:
        return  # nothing changed: zero work on this rerun

    # schema errors block; anything else becomes the validation report banner
    try:
        data, report = state.load_and_validate(fingerprint, file_bytes)
    except SchemaError as exc:
        st.sidebar.error(f"Schema error — cannot use this workbook: {exc}")
        st.session_state.pop("sr_bundle", None)
        st.session_state.pop("sr_bundle_key", None)
        st.session_state["sr_schema_error"] = str(exc)
        return

    # base_date guardrail: the opening stock is only valid from the snapshot
    # date through the last planned day — reject anything outside it
    snapshot = data.stock_snapshot_date.date()
    max_plan = data.max_plan_date.date()
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
    if report.blocking:
        st.session_state.pop("sr_bundle", None)
        st.session_state.pop("sr_bundle_key", None)
        return
    st.session_state["sr_bundle"] = state.compute_projection(
        fingerprint, base_date.isoformat(), file_bytes)
    st.session_state["sr_bundle_key"] = key
    # a new dataset invalidates per-dataset session caches
    for k in ("sr_effective", "sr_risk_tables", "sr_labels"):
        st.session_state.pop(k, None)

def _data_sidebar() -> None:
    """Workbook + base_date intake behind a 'Load data' Apply form."""
    st.sidebar.header("Data")

    bundle = state.get_bundle()
    # default date shown in the widget: the loaded bundle's date, else today's
    # bundled-default snapshot (resolved after first load)
    date_default = bundle["base_date"].date() if bundle else None

    with st.sidebar:
        with apply.panel("data_intake"):
            uploaded = st.file_uploader(
                "Universal data form (.xlsx)", type=["xlsx"],
                help="The 7-sheet input_data_form workbook. Leave empty to "
                     "use the bundled default dataset.")
            path_text = st.text_input(
                "…or a path on this machine",
                placeholder="/data/input_data_form.xlsx",
                help="Point at a workbook already on this machine instead of "
                     "uploading. Leave empty to use the bundled default "
                     "dataset.")
            base_date = st.date_input(
                "Base date (projection start)", value=date_default,
                help="First day of the projection. Must lie between the stock "
                     "snapshot date and the last production-plan date; the "
                     "allowed window is enforced on load.")
            applied = apply.button(
                "Load data",
                help="Read the chosen workbook and (re)build the projection "
                     "with this base date.")

    first_run = state.get_bundle() is None and \
        "sr_schema_error" not in st.session_state
    if not (applied or first_run):
        # ordinary rerun: show status of what's loaded, do NO work
        _sidebar_status()
        return

    file_bytes, using_default = _resolve_file(uploaded, path_text)
    if file_bytes is None:
        st.session_state.pop("sr_bundle", None)
        st.session_state.pop("sr_bundle_key", None)
        return
    if using_default:
        st.sidebar.caption(
            "Showing the bundled **default dataset**. Upload your own "
            "workbook above to replace it.")

    if base_date is None:
        # first automatic load: default to the workbook's own snapshot date
        try:
            data, _ = state.load_and_validate(
                state.fingerprint_bytes(file_bytes), file_bytes)
            base_date = data.stock_snapshot_date.date()
        except SchemaError as exc:
            st.sidebar.error(f"Schema error — cannot use this workbook: {exc}")
            return
    _compute_if_changed(file_bytes, base_date)
    _sidebar_status()

def _sidebar_status() -> None:
    """One-line summary of the currently loaded projection."""
    bundle = state.get_bundle()
    if bundle is None:
        return
    st.sidebar.success(
        f"{bundle['projection']['item_code'].nunique()} items projected "
        f"{bundle['base_date'].date()} → "
        f"{pd.Timestamp(bundle['projection']['date'].max()).date()}")

# ---- sidebar: intake form + the global apply-mode switch --------------------
_data_sidebar()
apply.render_mode_toggle()

# ---- navigation: the three modules -----------------------------------------
pages_dir = Path(__file__).parent / "pages"
nav = st.navigation([
    st.Page(pages_dir / "1_pipeline.py", title="Pipeline Status", icon="📡",
            default=True),
    st.Page(pages_dir / "2_forecast.py", title="Forecast", icon="🔮"),
    st.Page(pages_dir / "3_cost.py", title="Cost", icon="💰"),
])

# a blocking schema report replaces the page; anything else renders normally
report = st.session_state.get("sr_report")
if report is not None and report.blocking:
    render_validation_panel(report)
else:
    nav.run()

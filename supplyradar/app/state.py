"""Streamlit state and cached computation. The ONLY module that may bridge
streamlit and core/viz.

PERFORMANCE MODEL (why the app stays light on every rerun):
- The workbook is fingerprinted (sha256) ONCE when its bytes first arrive; all
  cache keys downstream are that small fingerprint string, never the 2 MB of
  raw bytes (hashing bytes on every rerun was measured at ~3 ms/click and,
  more importantly, forced Streamlit to treat the bytes as a cache-key arg).
- Every @st.cache_data / @st.cache_resource function takes the fingerprint as
  its hashed key and the big objects as underscore-prefixed (unhashed) args.
- The built pipeline FIGURE is cached with st.cache_resource keyed on a cheap
  tuple; a rerun with unchanged applied choices costs ~0 ms instead of the
  ~3 s full rebuild that made the app feel like it "restarts on every click".
- All caches carry max_entries so a long-running shared server cannot grow
  unbounded across many uploaded workbooks.

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
from supplyradar.core.forecast.preprocessing import build_rate_series
from supplyradar.core.loader import WorkbookData, load_workbook
from supplyradar.core.prep_consumption import append_to_actuals, build_projected_consumption
from supplyradar.core.prep_stock import build_stock_projection, stage_entry_dates, summarize_risk
from supplyradar.core.timing import stage as timing_stage
from supplyradar.core.validation import ValidationReport, apply_exclusions, validate
from supplyradar.viz.pipeline_chart import build_pipeline_chart, palette_from_delivery

CONFIG_DIR = REPO_ROOT / "config"

def load_defaults() -> dict:
    """UI defaults from config/defaults.yaml (risk window, lane pixels, …)."""
    path = CONFIG_DIR / "defaults.yaml"
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    return {}

def load_palette_config() -> tuple[dict, list]:
    """Stage colors + delivery color cycle from config/palette.yaml."""
    path = CONFIG_DIR / "palette.yaml"
    if path.exists():
        cfg = yaml.safe_load(path.read_text()) or {}
        return cfg.get("stages") or {}, cfg.get("delivery_cycle") or []
    return {}, []

def fingerprint_bytes(file_bytes: bytes) -> str:
    """Content hash of a workbook. Computed ONCE per new file (in main.py's
    sidebar) and passed around as the cache key everywhere after that."""
    return hashlib.sha256(file_bytes).hexdigest()

@st.cache_data(show_spinner="Reading workbook…", max_entries=4)
def load_and_validate(fingerprint: str, _file_bytes: bytes
                      ) -> tuple[WorkbookData, ValidationReport]:
    """Parse + schema-check + normalize the workbook, then run validation.

    Keyed on the small fingerprint string; the raw bytes are an unhashed arg
    (leading underscore) so Streamlit never re-hashes 2 MB per lookup.
    """
    with timing_stage("load_and_validate"):
        data = load_workbook(io.BytesIO(_file_bytes))
        return data, validate(data)

@st.cache_data(show_spinner="Projecting consumption and stock…", max_entries=6)
def compute_projection(fingerprint: str, base_date_iso: str,
                       _file_bytes: bytes) -> dict:
    """Full prep pipeline -> the page 'bundle', cached on (fingerprint, date).

    Runs once per (workbook, base_date); every page then reads the bundle from
    session_state without recomputing anything.
    """
    with timing_stage("compute_projection"):
        data, report = load_and_validate(fingerprint, _file_bytes)
        clean = apply_exclusions(data, report)
        base_date = pd.Timestamp(base_date_iso)
        projected = build_projected_consumption(clean, base_date)
        stages, cycle = load_palette_config()
        return {
            "fingerprint": fingerprint,
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

    Memoized per workbook in session_state — pages call this on every rerun
    and the map never changes for a given file.
    """
    memo = st.session_state.get("sr_labels")
    if memo is not None and memo[0] == bundle["fingerprint"]:
        return memo[1]
    bom = bundle["clean"].bom
    desc = bom.set_index("item_code")["item_description"].astype(str).str.strip()
    dup = desc.duplicated(keep=False)
    labels = {}
    display = bundle["display_names"]
    for code, d in desc.items():
        base = d if d and d.lower() != "nan" else display.get(code, code)
        labels[code] = f"{base} [{display.get(code, code)}]" if dup.get(code) \
            else base
    st.session_state["sr_labels"] = (bundle["fingerprint"], labels)
    return labels

def label_of(bundle: dict, code: str) -> str:
    return item_labels(bundle).get(code, bundle["display_names"].get(code, code))

# ------------------------------------------------------- forecast + switch

@st.cache_data(show_spinner="Reading production streams…", max_entries=64)
def available_streams(fingerprint: str, item_code: str,
                      _data: WorkbookData) -> tuple[list[str], list[str]]:
    """Output types and production lines present for an item (for the stream
    selectors), derived from the finest rate series."""
    series = build_rate_series(_data, items=[item_code])
    outs = sorted({k[1] for k in series})
    lines = sorted({k[2] for k in series})
    return outs, lines

@st.cache_data(show_spinner="Running the forecast competition…", max_entries=64)
def cached_item_forecast(fingerprint: str, item_code: str, horizon: int,
                         override_model: str | None, override_window: int | None,
                         by_output: tuple[str, ...] | None,
                         by_line: tuple[str, ...] | None,
                         _data: WorkbookData) -> ItemForecast:
    """The AMCIP competition for one material at one grouping — the expensive
    call on the Forecast page (seconds). Keyed on every choice that changes
    the result; the workbook rides along unhashed."""
    with timing_stage(f"item_forecast:{item_code}"):
        return run_item_forecast(_data, item_code, horizon,
                                 by_output=by_output, by_line=by_line,
                                 override_model=override_model,
                                 override_window=override_window)

@st.cache_data(show_spinner="Running the competition for every item…",
               max_entries=4)
def cached_run_all(fingerprint: str, horizon: int,
                   _data: WorkbookData) -> pd.DataFrame:
    """Fleet-wide competition (minutes at 138 items) — bounded to a few
    entries; each is only a small summary frame but the compute is dear."""
    return run_all_items(_data, horizon)

@st.cache_data(show_spinner="Running the competition for the selected materials…",
               max_entries=8)
def cached_run_selected(fingerprint: str, horizon: int, items: tuple[str, ...],
                        _data: WorkbookData) -> pd.DataFrame:
    """Competition over a chosen subset of materials (hand pick or a category
    group). Cached on the exact item set so re-selecting is instant."""
    return run_all_items(_data, horizon, items=list(items))

# ------------------------------------------------------- pipeline figure cache

@st.cache_resource(show_spinner="Drawing the pipeline chart…", max_entries=16)
def cached_pipeline_figure(cache_key: tuple, _projection: pd.DataFrame,
                           _bom: pd.DataFrame, _palette: dict,
                           _display_names: dict, items: tuple[str, ...],
                           granularity: str, lane_px: int, max_tick_lanes: int):
    """The built Plotly figure, cached zero-copy.

    Building the 138-lane figure was measured at ~3 s and used to run on EVERY
    widget interaction — the single biggest cause of the app feeling heavy.
    st.cache_resource returns the SAME object (no pickle round-trip, unlike
    st.cache_data which measured ~0.5 s/read at this figure size), so a rerun
    with unchanged applied choices costs ~0 ms.

    cache_key is a CHEAP tuple — (fingerprint, base_date, switch-state key) —
    plus the hashed display params below; the large frames ride along unhashed
    (leading underscore). fingerprint+base_date+switch fully determine the
    projection, so the key is sound. The returned figure must be treated as
    read-only (st.plotly_chart only reads it).
    """
    with timing_stage(f"build_pipeline_chart:{len(items)}items"):
        return build_pipeline_chart(
            _projection.loc[_projection["item_code"].isin(items)], _bom,
            stage_palette=_palette, items=list(items), granularity=granularity,
            display_names=_display_names, lane_px=lane_px,
            max_tick_lanes=max_tick_lanes)

def switch_state_key() -> tuple:
    """A small tuple identifying the current forecast-switch state, used in
    cache keys wherever the effective projection matters."""
    return tuple(sorted((i, f.horizon_months, f.generated_at)
                        for i, f in switched_forecasts().items()))

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
                 switch_state_key())
    cached = st.session_state.get("sr_effective")
    if cached is not None and cached[0] == cache_key:
        return cached[1], cached[2]

    clean = bundle["clean"]
    base_date = bundle["base_date"]
    default = bundle["projected"]
    # replace ONLY the (item, output_type, line) sub-combos the forecast covers;
    # streams the planner did not forecast (e.g. F when only B was chosen) keep
    # their standard plan rows
    covered = {(i, o, ln) for i, f in switched.items()
               for (o, ln) in f.covered_subcombos}
    row_sub = list(zip(default["item_code"], default["output_type"],
                       default["production_line"]))
    keep_mask = [s not in covered for s in row_sub]
    kept = default.loc[keep_mask]
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

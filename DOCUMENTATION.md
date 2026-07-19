# SupplyRadar — Code Documentation

A file-by-file guide to the entire repository: what each file does, the key
functions inside it, and how data flows between them. Read `README.md` first
for the product overview; this document is for anyone working on the code.

---

## 1. Big picture

```
Excel workbook (7 sheets)
        │  loader.py           read + schema-check + normalize
        ▼
   WorkbookData
        │  validation.py       rule engine -> ValidationReport (flag, never delete)
        ▼
  clean WorkbookData
        │  prep_consumption.py plan × rates -> daily projected consumption
        ▼
 projected consumption ──────────────┐
        │  prep_stock.py             │  core/forecast/*  (AMCIP engine)
        ▼                            ▼
 daily stock projection      per-material rate forecasts
        │                            │
        ▼                            ▼
 viz/pipeline_chart.py       viz/forecast_chart.py     viz/cost_chart.py
        │                            │                        │
        └────────────► app/ (Streamlit shell) ◄───────────────┘
```

Architecture rule enforced everywhere: **if a function needs
`import streamlit`, it does not belong in `core/` or `viz/`.** Core and viz
are pure Python — testable without a browser; the `app/` layer is a thin
shell that wires widgets to cached calls.

---

## 2. Repository layout

| Path | Role |
|---|---|
| `supplyradar/core/` | Pure-python engines (the real product) |
| `supplyradar/viz/` | Pure Plotly figure factories |
| `supplyradar/app/` | Streamlit shell: pages, state, components |
| `supplyradar/prep/` | Headless CLI wrapper around core |
| `config/` | YAML configuration (palette, UOM table, UI defaults) |
| `data/` | Bundled default workbook (anonymized public dataset) shown before any upload |
| `tests/` | Pytest suite on synthetic fixtures (never the real workbook) |
| `.streamlit/config.toml` | Pinned light theme with dark text |
| `requirements.txt` | Pinned direct dependencies (what the suite is tested on) |
| `pyproject.toml` | Package metadata + ruff/mypy/pytest config |

---

## 3. `supplyradar/core/` — the engines

### `core/loader.py`
Reads the 7-sheet "universal data form" into a typed `WorkbookData`
dataclass (one DataFrame per sheet + a display-name map + source metadata).

- `SHEET_SCHEMAS` — the required columns per sheet; a missing sheet/column
  raises `SchemaError` (the only kind of error that blocks the app).
- `NORMALIZED_COLUMNS` — every key/label column is normalized
  `strip().lower()` at load time ('Ton'/'ton'/'PC'/'pc' become one value);
  `display_names` keeps the original casing for the UI.
- `OPTIONAL_COLUMNS` — newer workbooks carry `cons_rate` / `cons_rate_uom`
  on the consumption sheet; older ones without them still load.
- `load_workbook(source)` — the entry point; parses dates (unparseable
  date = SchemaError), coerces numerics, builds the display map.
- `RATE_UOM_ALIASES` — verified alternate spellings of the three rate units
  (`k/t`→kg/ton, `p/s`→pc/heat, `t/d`→ton/day), proven by row-for-row value
  comparison against the long-form workbook. Unknown spellings still hit
  validation's blocking rule. Plain single-letter uoms (`t`/`p`/`k`) are
  aliases in `core/uom.py`.
- `WorkbookData.stock_snapshot_date` / `.max_plan_date` — the two dates that
  bound the allowed `base_date`.

### `core/validation.py`
The rule engine that runs BEFORE any calculation. Produces a
`ValidationReport` — a list of `ValidationIssue(severity, sheet, rule,
row_ids, message, action)`. Policy: only schema-level breaks block
(`severity='error'`); everything else is a warning/info, and rows are only
excluded (`action='drop'`) where keeping them would corrupt calculations
(e.g. consumption rows whose (output_type, line) combo exists nowhere).
Notable deliberate choices:
- duplicate consumption keys are **flagged, not dropped** — in this data they
  are transactions that net together (including negative adjustment rows);
- `apply_exclusions(data, report)` returns a clean copy for the engines while
  the original data stays intact for the report.

### `core/uom.py`
ALL unit conversion lives here — one explicit table, no guessing.
- `normalize_uom` — any spelling of ton/kg/pc to canonical form.
- `convert(qty, from, to, unit_wt_kg=...)` — mass↔mass via kg factors,
  mass↔pc via the item's unit weight. `pc → mass` with a missing/zero weight
  yields 0 (the source data's own convention, flagged in validation);
  `mass → pc` without a weight raises (division would be undefined).

### `core/prep_consumption.py`
Turns the production PLAN into daily projected consumption per
(item, output_type, line), matching the consumption sheet's schema exactly.
Three confirmed rate branches — `kg/ton` (follows planned tonnage),
`pc/heat` (follows planned heats), `ton/day` (flat per calendar day) — any
other unit raises. `check_base_date` enforces the
[snapshot .. last-plan-date] window. A stock item without a rate for a combo
the plan produces raises `ContractError` (fail loud, never silent zeros).

### `core/prep_stock.py`
The FIFO supply-bucket waterfall producing the daily stock projection:
bucket 0 = opening warehouse qty, then each delivery lot ordered by arrival.
For each (item, day): `qty = available − cum_demand` (negative = shortage),
`stock_stage` = the bucket demand is currently eating into ('gap' once demand
exceeds everything that will ever arrive). Also emits the time-varying
thresholds (`daily_need` = forward 30-day mean of projected consumption;
`safety_qty` and `overstock_qty` = the day-counts from consumption_figs ×
daily_need). `summarize_risk` sorts items by time-to-gap;
`stage_entry_dates` feeds the per-stage risk windows.
Accepts `consumption_type` in {'plan','forecast'} so switched materials can
be driven by the AMCIP forecast.

### `core/forecast/` — the AMCIP engine
Adaptive Material Consumption Intelligence Platform: forecasts the monthly
consumption RATE per production-stream group, then reconstitutes demand as
rate × planned production.

| File | What it does |
|---|---|
| `preprocessing.py` | Builds monthly `RateSeries` per (item, output_type, line) from the consumption sheet's `cons_rate` columns (or derives the rate from monthly actual production for older workbooks). Carries the production driver per month — the weight for weighted-average models — and data-quality flags (`flag_rate_series`). |
| `behavior.py` | Statistical profile per series (CV, trend, seasonality on the DETRENDED series, stationarity, outliers, structural break on detrended residuals) + a classification (Stable / Trending / Seasonal / Highly Variable / Random / Intermittent / Structural Change). Insight only — never picks the model. |
| `memory.py` | Candidate lookback windows {2,3,6,12,24,all}; heavy MLE models compete only on the two longest (short windows starve them while multiplying fit cost). |
| `models.py` | The model library: Naive, Moving Average, **Weighted Moving Average (weights each month's rate by that month's production quantity)**, Seasonal Naive, SES, Holt, damped ETS, Holt-Winters, ARIMA, SARIMA, Croston for intermittent series. Every model degrades gracefully (a fit failure skips the pipeline); forecasts are clipped at zero. |
| `cross_validation.py` | Rolling-origin one-step-ahead validation — chronological, never a random split — scoring MAE/RMSE/MAPE/sMAPE/MASE per (model, window) pipeline. |
| `selector.py` | Ranks pipelines on error + stability; within a 5% tie the SIMPLEST model wins (Occam). `confidence_score` maps sMAPE to 0-100. |
| `explanation.py` | Plain-language "why this model won / why the runners-up lost" + the recommendation (automatic vs manual review). |
| `engine.py` | Orchestration. `run_item_forecast(by_output=…, by_line=…)` implements the **tree of choices**: an empty axis is combined into ONE aggregated forecast (total consumption ÷ total production — a production-weighted rate), chosen values are broken out. Each `ComboForecast` carries the sub-combos it `covers`; `forecast_projected_consumption` applies a group's rate to every covered stream's planned production. Also: `expected_monthly_consumption`, `build_comparison_table` (history + top-3 winners for the editable table), `run_all_items` (fleet competition, optionally over a subset). |

### `core/timing.py`
Env-gated timing harness (`SUPPLYRADAR_TIMING=1`): `with stage("name"):`
logs wall-clock ms to stderr. Zero overhead when off. Used around workbook
load, projection build, forecasts and chart build so performance claims are
measured, not guessed.

---

## 4. `supplyradar/viz/` — figure factories (pure Plotly)

### `viz/pipeline_chart.py`
The signature multi-lane stage-encoded inventory area chart.
- One lane per item; area height = projected stock; fill color = the supply
  bucket currently feeding the plant; geometry and color are independent.
- Stage vocabulary is read from the data; `gap` is ALWAYS red (the one rule
  no palette can override); delivery labels get cycle colors in ETA order
  (`palette_from_delivery` — computed once so filtering never repaints).
- Per-lane normalization (`display_y = offset + value/lane_max`), PCHIP
  smoothing (no overshoot → no phantom stock on delivery step-ups),
  spline-smoothed threshold lines, merged same-stage polygons
  (< 10 traces/item), shortage drawn below zero in gap red, hover only on
  the lane under the cursor.
- Performance: numpy arrays with NaT/NaN separators take Plotly's fast
  validation path; interpolation density adapts to lane count.

### `viz/forecast_chart.py`
Rate history + forecast + 80/95% confidence band, with expected material
consumption bars below (two stacked panels, never a dual axis); optional
manual-override overlay. Dark axis/legend ink throughout.

### `viz/cost_chart.py`
Stacked monthly projected-spend by category + a totals-by-category bar.
Neutral categorical palette deliberately distinct from the reserved stage
colors.

---

## 5. `supplyradar/app/` — the Streamlit shell

### `app/main.py`
Entry point (`streamlit run supplyradar/app/main.py`).
- Sidebar intake behind a **"Load data" Apply form**: workbook upload / path
  / base_date. On ordinary reruns the sidebar does NO work — it compares a
  stored (fingerprint, base_date) key and shows status.
- `_resolve_file` — upload > path > bundled default (`data/…xlsx`).
- `_compute_if_changed` — fingerprints the bytes ONCE, recomputes the bundle
  only when (fingerprint, base_date) actually changed, enforces the
  base-date window, and invalidates the per-dataset session caches.
- Renders the global **"Auto-apply changes"** toggle and the three-page
  navigation; a blocking schema report replaces the page.

### `app/state.py`
The ONLY module bridging Streamlit and core/viz. Owns the caching model:
- `fingerprint_bytes` — sha256 of the workbook, computed once per file.
- `load_and_validate` / `compute_projection` — `st.cache_data` keyed on the
  small fingerprint (the 2 MB bytes ride along unhashed).
- `cached_pipeline_figure` — **`st.cache_resource` (zero-copy)** for the
  built Plotly figure, keyed on the applied choices; a rerun with unchanged
  choices costs ~1 ms instead of the ~3 s rebuild.
- `cached_item_forecast` / `cached_run_all` / `cached_run_selected` /
  `available_streams` — forecast-side caches, all with `max_entries`.
- `switched_forecasts` / `set_switch` / `effective_tables` — the forecast
  switch: replaces ONLY the (item, output_type, line) sub-combos a forecast
  `covers` with forecast-driven rows; everything else keeps plan rows.
- `risk_tables`, `item_labels` (memoized per fingerprint), `require_data`.

### `app/components/apply.py`
The Apply-button system (see its module docstring for the full rationale).
- `panel(key)` — a choice panel: an `st.form` in manual mode (composing
  choices causes NO rerun until Apply), a plain container in auto mode.
- `button(label)` — the panel's Apply button (absent in auto mode).
- `action_button(label)` — click-gated in BOTH modes (used for expensive
  actions like the fleet competition).
- `render_mode_toggle()` — the sidebar "Auto-apply changes" switch
  (default off = manual Apply).

### `app/components/samples.py`
Sample-data downloads offered in the sidebar: `build_sample_csv` (one CSV —
per table a `# table:` marker, the header and the first 10 rows) and
`build_data_dictionary` (a TXT explaining every table and every column of the
universal data form, with the current file's row counts). Built from the
loaded workbook so the sample always matches the working dataset.

### `app/components/tutorial.py`
Per-module tutorial mode. `tutorial.begin(page, title)` renders the page
title with a "📘 Tutorial" toggle button; when active, every
`tutorial.tip(page, title, text)` call renders a numbered amber callout box
explaining the section that follows it. Pure Streamlit (styled markdown), no
JS; tips cost nothing when tour mode is off.

### `app/components/tables.py`
`highlight_rows` — highlighted table rows keep explicit DARK text so a pale
highlight can never hide the font.

### `app/components/validation_panel.py`
Renders the ValidationReport: schema errors block with a download link;
warnings become a banner + expandable findings table.

### `app/pages/1_pipeline.py` — Pipeline Status (landing page)
Headline answer → risk-settings Apply panel (global gap/safety windows +
one window per supply stage) → filter Apply panel (category, sub-category,
stage, only-show, hide, at-risk-only, granularity) → the cached pipeline
chart → "how to read" → at-risk table (with which rule fired) → raw data.

### `app/pages/2_forecast.py` — Forecast (AMCIP dashboard)
"Run forecast" Apply panel (material, horizon, and the two break-out
selectors implementing the tree of choices) → per-group tabs: intelligence
card, recommendation, manual-override Apply panel, rate chart, the
history + top-3 + "Your projection" editable table, explanation, competition
table → the projection switch → fleet competition behind an explicit
"Run competition" action button (materials and/or whole categories).

### `app/pages/3_cost.py` — Cost
Total projected spend + share on at-risk items, spend over time by category,
totals by category, spend on at-risk materials. Uses the switch-aware
effective consumption.

### `supplyradar/prep/run.py`
Headless CLI: `python -m supplyradar.prep.run --workbook … --base-date …
--out …` → parquet outputs + validation_report.json + console summary.

---

## 6. `config/` — YAML, not code

| File | Contents |
|---|---|
| `palette.yaml` | Stage anchors (warehouse green, gap red — gap is not overridable) + the delivery-label color cycle assigned in ETA order. |
| `uom.yaml` | UOM aliases + mass factors (extends the built-in table). |
| `defaults.yaml` | UI knobs: at-risk window, lane pixels, tick-lane cap. |

## 7. `tests/`

| File | Covers |
|---|---|
| `conftest.py` | The synthetic 2-item workbook fixture (in-memory + written xlsx with messy casing). |
| `test_uom.py` | Conversion table + the three rate-branch computations, hand-checked. |
| `test_loader.py` | Normalization, display map, optional columns, schema errors block. |
| `test_validation.py` | Drop-and-count of orphan combos, flag-not-drop of netting rows, blocking rules. |
| `test_prep_consumption.py` | Golden hand-computed projection, ton/day calendar branch, missing-combo raises, base-date window. |
| `test_prep_stock.py` | qty-identity property test, exact stage-transition dates, shrinking daily-need window, risk ordering, stage entry dates. |
| `test_forecast_engine.py` | Rate-series construction (netting, uom backfill, derivation), model library (incl. production-weighted WMA), chronological no-leakage CV, Occam tie-break, trending e2e, override pinning, the grouping tree of choices, aggregated switch coverage, comparison table. |
| `test_pipeline_chart.py` | Trace-count bound, exact stage-boundary dates, gap-always-red, negative rendering, ordering, legend uniqueness. |
| `test_perf_cache.py` | Figure cache: identical call = no rebuild + same object; fingerprint stability. |
| `test_app_smoke.py` | Boot on the bundled default, load-form flow, page walk, upload override, designed empty state. |
| `test_samples.py` | Sample CSV carries every table capped at 10 rows; the data dictionary documents every column. |

## 8. Performance model (why the app is not "heavy")

Measured on the bundled 138-item workbook:

| Path | Before | After |
|---|---|---|
| Any widget click on Pipeline (chart rebuild) | ~3.0 s + 0.7 s serialize | **~1 ms** figure-cache hit (serialize only when the chart actually changes/renders) |
| Composing choices (multiselects, toggles, dates) | full script rerun per click | **no rerun at all** until Apply (manual mode) |
| Sidebar on every rerun | file read + 2 MB sha256 ×3 | dict compare (~µs) |
| Changing base_date mid-thought | immediate ~6 s recompute | recompute only on "Load data" |

The three mechanisms, in order of impact:
1. **Apply forms** (`components/apply.py`) — interactions inside a panel do
   not rerun the script until the user commits.
2. **Zero-copy figure cache** (`state.cached_pipeline_figure`) — the ~3 s
   chart build runs once per distinct applied view.
3. **Fingerprint-first cache keys** — no per-click hashing of workbook
   bytes; every cache is bounded with `max_entries`.

Set `SUPPLYRADAR_TIMING=1` to log stage timings and verify any of this.

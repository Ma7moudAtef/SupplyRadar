# SupplyRadar

Supply-chain visibility for industrial manufacturers. One Excel workbook in,
one answer out: **which material runs out, when, and which supply bucket is
feeding the plant in the meantime.**

The signature view is the **Pipeline Status** chart: one lane per item, area
height = projected stock, fill color = the supply bucket currently being
consumed (warehouse → each incoming delivery lot → red `gap` when the pipeline
runs dry). Geometry and color are independent — the top curve stays continuous
across color changes.

**Working on the code?** See [`DOCUMENTATION.md`](DOCUMENTATION.md) — a
file-by-file guide to every module, the data flow, the caching/performance
model, and the Apply-button system.

## Quick start

```bash
pip install -r requirements.txt
make run                        # = streamlit run supplyradar/app/main.py
```

On Windows (no `make`), run the same command directly from the repo root:

```bat
python -m streamlit run supplyradar/app/main.py
```

The app is OS-independent (Windows / macOS / Linux) — run it locally or on
any web server; always start it from the repository root so the bundled
default dataset and `.streamlit/config.toml` are picked up.

The app ships with a bundled sample dataset (`data/default_input_public.xlsx`)
so it renders immediately with no setup — open it and the Pipeline Status
dashboard is already populated. Upload your own workbook (or paste a path) in
the sidebar to replace it at any time; the upload always takes priority over
the default. Base date is the only other input; everything else is derived
from the data.

Headless prep (parquet + validation report + console summary):

```bash
python -m supplyradar.prep.run --workbook input_data_form.xlsx \
    --base-date 2026-07-11 --out ./out
```

Tests and lint:

```bash
make test                       # pytest (39 tests, >90% coverage on core/viz)
make lint                       # ruff
```

## Architecture

```
supplyradar/
  core/          # pure python, zero streamlit imports — the real product
    loader.py            # read + schema-check + normalize the 7-sheet workbook
    validation.py        # rule engine -> ValidationReport (never auto-deletes)
    uom.py               # ALL unit conversion lives here (explicit table)
    prep_consumption.py  # production plan -> projected consumption per item
    prep_stock.py        # FIFO waterfall -> daily stock projection + thresholds
    forecast/            # AMCIP adaptive forecast engine
      preprocessing.py       # consumption sheet -> monthly rate series
      behavior.py            # stats + behavior classification (insight only)
      memory.py              # adaptive lookback candidates
      models.py              # Naive..SARIMA + Croston model library
      cross_validation.py    # rolling-origin CV (chronological, never random)
      selector.py            # error + stability + Occam's-razor selection
      explanation.py         # why the winner won, why competitors lost
      engine.py              # orchestration + forecast-driven consumption rows
  viz/           # pure plotly figure factories, zero streamlit imports
    pipeline_chart.py    # the signature stage-encoded inventory area chart
    forecast_chart.py    # rate history + forecast + confidence band
    cost_chart.py        # projected spend over time / by category
  prep/
    run.py               # CLI wrapper around core
  app/           # thin streamlit shell: calls core + viz, renders, holds state
    main.py  state.py  components/
    pages/               # 1 Pipeline Status · 2 Forecast · 3 Cost
config/          # stage palette, uom table, UI defaults — YAML, not code
tests/           # synthetic fixtures only; the real workbook is never required
```

Rule enforced throughout: if a function needs `import streamlit`, it does not
belong in `core/` or `viz/`.

## The data contract

Input is the 7-sheet "universal data form": `prod`, `bom`, `consumption_figs`,
`consumption`, `delivery`, `stock`, `suppliers`. All key/label matching is
case- and whitespace-insensitive (`Ton`/`ton`, `PC`/`pc` are the same value; a
display-name map preserves original casing for the UI). No business values are
hardcoded — lines, output types, stages and UOMs are read from the data.

**Severity policy** — only a broken *schema* blocks the app: missing sheet,
missing column, unparseable dates, or an unknown `std_cons_rate_uom`.
Everything else is a warning: the affected rows are excluded where necessary,
counted, and shown in the validation panel — a single bad row never prevents a
dashboard from rendering.

Notable data realities the pipeline handles by design:

- `consumption` combos with no match in `prod`/`consumption_figs`
  (e.g. `output_type = 'C'`) are **dropped and counted** — no mapping is
  invented.
- Multiple consumption rows per (date, item, output_type, line) are
  **transactions that net together** (including negative adjustment entries).
  They are flagged and summed at the reporting grain, never deleted —
  deduplicating or dropping negatives would inflate actual history.
- Delivery lots with a **past-due ETA** (arrival before the stock snapshot)
  are treated as available from the base date and flagged.
- `pc` items with missing/zero `unit_wt_kg` follow the source data's own
  convention: their ton-denominated figures are zero (flagged).
- `suppliers.lead_time_days` is loaded, exposed, and ignored (out of scope for
  v1 — no ETA prediction). `consumption_figs.daily_need_uom` is always derived,
  never read.

## The projection model

For every item in `stock` and every date in `[base_date .. horizon_end]`
(horizon = last day of the month containing the last plan date):

1. **Projected consumption** from the daily production plan and
   `consumption_figs` rates. Exactly three confirmed rate branches:
   `kg/ton` (follows mass produced), `pc/heat` (follows heats),
   `ton/day` (flat per calendar day). Any other unit raises.
2. **Stock projection** as a FIFO waterfall: bucket 0 is the opening warehouse
   quantity; each delivery lot is a bucket available from its arrival date.
   `qty(D) = available(D) − cum_demand(D)` (negative = projected shortage) and
   `stock_stage(D)` is the bucket demand is currently eating into — `gap` once
   demand exceeds everything that will ever arrive.
3. **Thresholds** are stored in days and converted to time-varying quantities:
   `daily_need(D)` = forward-looking 30-day mean of projected consumption
   (window shrinks near the horizon, never zero-padded),
   `safety_qty = safety_days × daily_need`,
   `overstock_qty = (safety_days + replenishment_days) × daily_need`.
   Three reference levels only: zero, safety, overstock.

`base_date` is bounded by the stock snapshot date and the last plan date — the
opening warehouse quantity is only valid as of the snapshot, so projecting
from any other date is rejected rather than silently corrupted.

## Reference-workbook walkthrough

With the reference `input_data_form.xlsx` and base date `2026-07-11`:

```
horizon            : 2026-07-11 .. 2027-12-31  (539 days)
items projected    : 138
projected cons rows: 297,528  (+14,159 actual rows kept)
excluded rows      : {'consumption': 35}    # the orphan output_type='C' combo
validation issues  : 0 errors, 9 warnings, 2 info
at risk (30 days)  : 18 items; 12 hit gap within 30 days
```

The whole prep runs in ~5 s; the 138-lane pipeline chart builds in under 2 s
and stays interactive (≈6 traces per item: merged same-stage polygons, one
outline, one hover layer, plus figure-wide reference lines).

## The Adaptive Forecast Engine (AMCIP)

The Forecast page runs an adaptive competition per material, from the
`consumption` sheet (`cons_rate`, `cons_rate_uom`, `date`, `item_code`; older
workbooks without those columns get the rate derived from monthly actual
production):

1. **Variable**: the normalized monthly consumption rate per
   (item, output_type, production_line). Demand reconstitutes afterwards as
   `forecast rate × planned production`.
2. **Behavior analysis** classifies each series (Stable / Trending / Seasonal /
   Highly Variable / Random / Intermittent / Structural Change) — insight
   only, never the model picker.
3. **Adaptive memory**: candidate lookbacks {2, 3, 6, 12, 24, all} months are
   pipeline parameters; the effective memory is whatever the winner used.
4. **Model competition**: Naive, Seasonal Naive, Moving Average, Weighted MA,
   SES, Holt, ETS (damped), Holt-Winters, ARIMA, SARIMA, Croston for
   intermittent series. Prophet/XGBoost/LightGBM are spec-optional and not in
   v1. Heavy MLE models compete on long lookbacks only.
5. **Rolling-origin CV** (chronological, one-step-ahead) scores every pipeline
   on MAE / RMSE / MAPE / sMAPE / MASE; selection weighs error, stability and
   simplicity (ties go to the simplest model).
6. **Confidence**: 80/95% intervals from validation residuals + a 0-100
   confidence score; every selection ships with a plain-language explanation
   and a recommendation (automatic vs manual review).
7. **Manual override**: pin the model and/or lookback and compare against the
   automatic pick side by side.
8. **Stream grouping (tree of choices)**: choose how to break a material's
   forecast down. Leaving an axis empty combines it into one; picking values
   breaks them out. Output B with no line = **one** forecast for B (its lines
   aggregated by total consumption ÷ total production); output B with lines 1
   and 2 = **two**. A combined forecast drives every stream it covers in the
   projection (an aggregated B rate lands on B line 1 and B line 2).
9. **The switch**: per material, toggle the stock projection from standard
   plan rates to the forecast — the Pipeline and Cost pages follow instantly
   and label the forecast-driven materials. Streams you did not forecast keep
   their standard plan rate.

## UI conventions

- Planners see **item descriptions** everywhere; `item_code` stays the
  internal join key.
- Pipeline hover shows only the lane under the cursor.
- Risk windows are adjustable: global gap / below-safety days plus a separate
  window per supply stage ("flag if consuming from Not Paid within N days"),
  with an at-risk-only toggle and per-item hide.
- The former Consumption History page is retired: history lives in Forecast,
  money lives in Cost.

## Design notes on the chart

- Stage vocabulary is read from the data at runtime; `warehouse` is green and
  delivery labels get shades of blue from a light→dark ramp, spread evenly in
  ETA order (nearest supply lightest, furthest darkest) so no supply stage
  resembles the warehouse green. `gap` is **always red** — the one palette
  rule that cannot be overridden (`config/palette.yaml`).
- Each lane is normalized to its own maximum (`display_y = offset +
  value / lane_max`), because 138 items in tons, pieces and kg span orders of
  magnitude; tick labels show real quantities (0 black, safety green,
  overstock red, mirrored on both edges).
- The top boundary is PCHIP-smoothed (no overshoot → no phantom stock on
  delivery step-ups); stage labels are never interpolated and color changes
  snap to the exact transition date.
- Safety/overstock are drawn as time-varying Scatter series, not constant
  shapes — they move with the production plan.
- Negative quantity (shortage) renders below the zero line in the gap color,
  clamped for display at a fraction of the lane height so a deep shortage
  cannot invade the lane below; the hover always carries the true value.
- Default lane order is time-to-gap ascending — the most urgent material is
  the top lane.

# SupplyRadar

Supply-chain visibility for industrial manufacturers. One Excel workbook in,
one answer out: **which material runs out, when, and which supply bucket is
feeding the plant in the meantime.**

The signature view is the **Pipeline Status** chart: one lane per item, area
height = projected stock, fill color = the supply bucket currently being
consumed (warehouse → each incoming delivery lot → red `gap` when the pipeline
runs dry). Geometry and color are independent — the top curve stays continuous
across color changes.

## Quick start

```bash
pip install -r requirements.txt
make run                        # = streamlit run supplyradar/app/main.py
```

In the app: upload `input_data_form.xlsx` (or paste a path), pick the base
date, done. Those are the only two inputs; everything else is derived from the
data.

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
  viz/           # pure plotly figure factories, zero streamlit imports
    pipeline_chart.py    # the signature stage-encoded inventory area chart
    consumption_chart.py # actual vs plan
    cost_chart.py        # projected spend over time / by category
  prep/
    run.py               # CLI wrapper around core
  app/           # thin streamlit shell: calls core + viz, renders, holds state
    main.py  state.py  components/  pages/
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

## Design notes on the chart

- Stage vocabulary is read from the data at runtime; delivery labels get their
  colors in ETA order. `gap` is **always red** — the one palette rule that
  cannot be overridden (`config/palette.yaml`).
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

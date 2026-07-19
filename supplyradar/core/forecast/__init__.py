"""Adaptive Material Consumption Intelligence Platform (AMCIP).

Forecasts the normalized monthly CONSUMPTION RATE per
(item_code, output_type, production_line) from the `consumption` sheet, then
reconstitutes material demand as rate x planned production. Multiple
(model, lookback) pipelines compete under rolling-origin cross validation;
the winner is selected on error + stability + simplicity, every decision is
explained, and planners can override model / lookback / horizon.
"""

from .engine import (  # noqa: F401
                     ComboForecast,
                     ItemForecast,
                     TopForecast,
                     forecast_projected_consumption,
                     run_all_items,
                     run_item_forecast,
)

"""CLI: run the full data-preparation pipeline.

    python -m supplyradar.prep.run --workbook path/to/input_data_form.xlsx \
        --base-date 2026-07-11 --out ./out

Emits: consumption_projected.parquet, stock_projection.parquet,
validation_report.json, and a one-page console summary.
Exit codes: 0 ok, 2 blocked by schema errors.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from supplyradar.core.loader import SchemaError, load_workbook
from supplyradar.core.prep_consumption import append_to_actuals, build_projected_consumption
from supplyradar.core.prep_stock import build_stock_projection, summarize_risk
from supplyradar.core.validation import apply_exclusions, validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="supplyradar-prep", description=__doc__)
    parser.add_argument("--workbook", required=True, help="path to input_data_form.xlsx")
    parser.add_argument("--base-date", required=True, help="YYYY-MM-DD projection start")
    parser.add_argument("--out", default="./out", help="output directory")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    try:
        data = load_workbook(args.workbook)
    except SchemaError as exc:
        print(f"SCHEMA ERROR — cannot continue: {exc}", file=sys.stderr)
        return 2

    report = validate(data)
    (out_dir / "validation_report.json").write_text(report.to_json(indent=2))
    if report.blocking:
        print("BLOCKED by schema-level validation error(s):", file=sys.stderr)
        for issue in report.errors:
            print(f"  [{issue.sheet}] {issue.rule}: {issue.message}", file=sys.stderr)
        return 2

    clean = apply_exclusions(data, report)
    base_date = pd.Timestamp(args.base_date)

    projected = build_projected_consumption(clean, base_date)
    full_consumption = append_to_actuals(clean, projected)
    full_consumption.to_parquet(out_dir / "consumption_projected.parquet", index=False)

    projection = build_stock_projection(clean, projected, base_date)
    projection.to_parquet(out_dir / "stock_projection.parquet", index=False)

    risk = summarize_risk(projection)
    elapsed = time.perf_counter() - t0

    print("=" * 72)
    print("SupplyRadar prep summary")
    print("=" * 72)
    print(f"workbook           : {args.workbook}")
    print(f"base_date          : {base_date.date()}   "
          f"(stock snapshot: {data.stock_snapshot_date.date()})")
    print(f"horizon            : {projection['date'].min().date()} .. "
          f"{projection['date'].max().date()}  ({projection['date'].nunique()} days)")
    print(f"items projected    : {projection['item_code'].nunique()}")
    print(f"projected cons rows: {len(projected):,}  "
          f"(+{len(clean.consumption):,} actual rows kept)")
    print(f"stock projection   : {len(projection):,} rows")
    counts = report.excluded_counts()
    print(f"excluded rows      : {counts if counts else 'none'}")
    print(f"validation issues  : {len(report.errors)} error(s), "
          f"{len(report.warnings)} warning(s), "
          f"{len(report.issues) - len(report.errors) - len(report.warnings)} info")
    at_risk = risk.loc[risk["at_risk"]]
    gap_30 = risk.loc[risk["time_to_gap_days"] <= 30]
    print(f"at risk (30 days)  : {len(at_risk)} item(s); "
          f"{len(gap_30)} hit gap within 30 days")
    if not at_risk.empty:
        print("most urgent        :")
        for _, r in at_risk.head(5).iterrows():
            ttg = (f"gap in {int(r['time_to_gap_days'])} d"
                   if r["time_to_gap_days"] != float("inf") else "below safety")
            print(f"    {r['item_code']:<20} {ttg}")
    print(f"elapsed            : {elapsed:.2f}s")
    print(f"outputs            : {out_dir}/")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

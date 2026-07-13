import pandas as pd

from supplyradar.core.validation import apply_exclusions, validate


def test_clean_fixture_produces_no_blocking_issues(data):
    report = validate(data)
    assert not report.blocking


def test_unmatched_combo_dropped_and_counted(data):
    extra = data.consumption.iloc[[0]].assign(output_type="c")
    data.consumption = pd.concat([data.consumption, extra], ignore_index=True)
    report = validate(data)
    rules = {i.rule: i for i in report.issues}
    issue = rules["unmatched_output_type_line_combo"]
    assert issue.severity == "warning"
    assert issue.action == "drop"
    assert len(issue.row_ids) == 1
    clean = apply_exclusions(data, report)
    # dropped from computation, still present in the original
    assert len(clean.consumption) == len(data.consumption) - 1
    assert not report.blocking  # never blocks the app


def test_unknown_rate_uom_blocks(data):
    data.consumption_figs.loc[0, "std_cons_rate_uom"] = "liters/day"
    report = validate(data)
    assert report.blocking
    assert any(i.rule == "unknown_std_cons_rate_uom" and i.severity == "error"
               for i in report.issues)


def test_duplicate_consumption_flagged_not_dropped(data):
    data.consumption = pd.concat(
        [data.consumption, data.consumption.iloc[[0]]], ignore_index=True)
    report = validate(data)
    issue = next(i for i in report.issues if i.rule == "duplicate_consumption_key")
    assert issue.action == "flag"  # transactions net together; never deleted
    clean = apply_exclusions(data, report)
    assert len(clean.consumption) == len(data.consumption)


def test_negative_stock_excluded(data):
    data.stock.loc[0, "qty"] = -5.0
    report = validate(data)
    clean = apply_exclusions(data, report)
    assert len(clean.stock) == len(data.stock) - 1


def test_uom_conflict_reported(data):
    data.stock.loc[0, "uom"] = "pc"  # bom says ton
    report = validate(data)
    assert any(i.rule == "uom_conflict_stock_vs_bom" for i in report.issues)


def test_missing_membership_reported(data):
    data.stock = pd.concat([data.stock, pd.DataFrame([{
        "item_code": "ghost", "uom": "ton", "qty": 1.0,
        "date": pd.Timestamp("2026-01-01"), "stock_stage": "warehouse"}])],
        ignore_index=True)
    report = validate(data)
    rules = [i.rule for i in report.issues]
    assert "item_missing_from_bom" in rules
    assert "item_missing_from_consumption_figs" in rules

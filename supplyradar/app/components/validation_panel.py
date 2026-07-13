"""Validation report panel. Schema errors block; everything else is a banner
with an excluded-row count and a download link — the dashboard renders anyway.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from supplyradar.core.validation import ValidationReport


def render_validation_panel(report: ValidationReport) -> bool:
    """Render the panel. Returns True if the app may continue (no schema error)."""
    if report.blocking:
        st.error("The workbook has schema-level problems and cannot be processed:")
        for issue in report.errors:
            st.error(f"**{issue.sheet} / {issue.rule}** — {issue.message}")
        st.download_button("Download full validation report",
                           report.to_json(indent=2),
                           file_name="validation_report.json", mime="application/json")
        return False

    warnings = report.warnings
    excluded = report.excluded_counts()
    n_excluded = sum(excluded.values())
    if warnings or n_excluded:
        st.warning(
            f"Data quality: {len(warnings)} warning(s); "
            f"{n_excluded} row(s) excluded from calculations "
            f"({excluded if excluded else 'none'}). The dashboard renders on the "
            "remaining data."
        )
    with st.expander(f"Validation report — {len(report.issues)} finding(s)"):
        if report.issues:
            st.dataframe(pd.DataFrame([{
                "severity": i.severity, "sheet": i.sheet, "rule": i.rule,
                "rows": len(i.row_ids), "action": i.action, "message": i.message,
            } for i in report.issues]), width="stretch", hide_index=True)
        else:
            st.write("No findings. The workbook is clean.")
        st.download_button("Download validation_report.json",
                           report.to_json(indent=2),
                           file_name="validation_report.json", mime="application/json")
    return True

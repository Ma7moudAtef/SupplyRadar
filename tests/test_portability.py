"""Cross-platform regression guards.

The app must run on Windows, macOS and Linux alike. Python's strftime
delegates to the OS C runtime: the no-padding modifier ('%-d' on glibc,
'%#d' on Windows) is NOT portable and crashes with 'ValueError: Invalid
format string' on the other platform. Plotly tickformat / hovertemplate
strings are exempt — they are rendered in the browser by d3-time-format,
which supports '%-' everywhere.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# a strftime( call whose format string carries a platform-specific modifier
_BAD_STRFTIME = re.compile(r"strftime\(\s*[\"'][^\"']*%[-#]")


def test_no_platform_specific_strftime_directives():
    offenders = []
    for py in (REPO / "supplyradar").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        # collapse newlines so multi-line strftime("...") calls are caught too
        if _BAD_STRFTIME.search(text.replace("\n", " ")):
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, (
        f"platform-specific strftime modifier (%- or %#) found in: {offenders}; "
        "build the string from Timestamp attributes instead (see "
        "pipeline_chart._fmt_date)")


def test_fmt_date_is_portable_and_unpadded():
    import pandas as pd

    from supplyradar.viz.pipeline_chart import _fmt_date

    assert _fmt_date(pd.Timestamp("2025-02-03")) == "Feb 3, 2025"
    assert _fmt_date(pd.Timestamp("2026-12-25")) == "Dec 25, 2026"

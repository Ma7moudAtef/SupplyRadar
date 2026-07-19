"""Shared table styling helpers.

A highlighted row must keep DARK text — a pale highlight with the theme's
default (possibly light) font would make the row unreadable. These helpers set
both the background and an explicit dark font on highlighted rows.
"""

from __future__ import annotations

import pandas as pd

HIGHLIGHT_BG = "#d8ecdb"        # pale green
HIGHLIGHT_FG = "#14532d"        # dark green text — always legible on the pale bg
INK = "#1a2b4c"                 # default dark ink for every other cell

def highlight_rows(df: pd.DataFrame, mask, *, columns: list[str] | None = None):
    """Return a pandas Styler: `mask`-true rows get the pale highlight with
    dark bold text; all other cells get the default dark ink."""
    cols = columns or list(df.columns)

    def _row_style(row: pd.Series) -> list[str]:
        if bool(mask.loc[row.name]):
            return [f"background-color: {HIGHLIGHT_BG}; color: {HIGHLIGHT_FG}; "
                    "font-weight: 600"] * len(row)
        return [f"color: {INK}"] * len(row)

    return df[cols].style.apply(_row_style, axis=1)

"""Sample-download builders: the CSV carries every table with at most 10 rows,
and the data dictionary documents every column of every sheet."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supplyradar.app.components.samples import (  # noqa: E402
    SAMPLE_ROWS,
    build_data_dictionary,
    build_sample_csv,
)
from supplyradar.core.loader import SHEET_SCHEMAS  # noqa: E402


def test_sample_csv_has_every_table_capped_at_ten_rows(data):
    csv = build_sample_csv(data)
    for sheet in SHEET_SCHEMAS:
        marker = f"# ===== table: {sheet} "
        assert marker in csv, f"missing table marker for {sheet}"
        # section body: between this marker and the next marker/blank end
        section = csv.split(marker, 1)[1].split("# =====")[0]
        lines = [ln for ln in section.strip().splitlines() if ln.strip()]
        # marker remainder + header + at most SAMPLE_ROWS data rows
        n_data_rows = len(lines) - 2  # ') =====' remainder line + header
        assert n_data_rows <= SAMPLE_ROWS


def test_data_dictionary_documents_every_column(data):
    txt = build_data_dictionary(data)
    for sheet, cols in SHEET_SCHEMAS.items():
        assert f"TABLE: {sheet}" in txt
        for col in cols:
            assert col in txt, f"{sheet}.{col} missing from the dictionary"

"""End-to-end smoke test: boot the Streamlit app on the synthetic workbook,
load it through the sidebar path input, and walk every page.
UI logic is intentionally thin — this asserts wiring, not behavior.
"""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

MAIN = str(Path(__file__).resolve().parents[1] / "supplyradar" / "app" / "main.py")


def test_app_walks_every_page(workbook_path):
    at = st_testing.AppTest.from_file(MAIN, default_timeout=60)
    at.run()
    assert not at.exception

    at.sidebar.text_input[0].set_value(str(workbook_path))
    at.run()
    assert not at.exception
    assert at.sidebar.success  # "N items projected ..."
    assert at.title[0].value == "Pipeline Status"
    # the one-line answer renders
    assert any("go to gap" in m.value for m in at.markdown)

    for page in ("2_consumption", "3_forecast", "4_cost"):
        at.switch_page(f"pages/{page}.py")
        at.run()
        assert not at.exception, page


def test_pages_show_empty_state_without_data():
    at = st_testing.AppTest.from_file(MAIN, default_timeout=60)
    at.run()
    assert not at.exception
    assert at.info  # designed empty state, not a stack trace

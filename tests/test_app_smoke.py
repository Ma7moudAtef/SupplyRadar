"""End-to-end smoke test: boot the Streamlit app on the synthetic workbook,
load it through the sidebar path input, and walk every page.
UI logic is intentionally thin — this asserts wiring, not behavior.
"""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

MAIN = str(Path(__file__).resolve().parents[1] / "supplyradar" / "app" / "main.py")


def _click(at, label):
    """Click a button (incl. form submit buttons) by its label."""
    btn = [b for b in at.button if b.label == label][0]
    btn.set_value(True)


def test_app_walks_every_page(workbook_path):
    at = st_testing.AppTest.from_file(MAIN, default_timeout=60)
    at.run()
    assert not at.exception

    # the intake sits behind the 'Load data' Apply form now
    at.sidebar.text_input[0].set_value(str(workbook_path))
    _click(at, "Load data")
    at.run()
    assert not at.exception
    assert at.sidebar.success  # "N items projected ..."
    assert at.title[0].value == "Pipeline Status"
    # the one-line answer renders
    assert any("go to gap" in m.value for m in at.markdown)

    for page in ("2_forecast", "3_cost"):
        at.switch_page(f"pages/{page}.py")
        at.run()
        assert not at.exception, page


def test_app_uses_bundled_default_dataset_without_upload():
    """No upload, no path override: the bundled default workbook loads and
    the dashboard renders immediately — this is the actual out-of-the-box
    experience every user gets."""
    at = st_testing.AppTest.from_file(MAIN, default_timeout=60)
    at.run()
    assert not at.exception
    assert at.sidebar.success  # "N items projected ..."
    assert any("default dataset" in c.value for c in at.sidebar.caption)
    assert at.title[0].value == "Pipeline Status"


def test_uploaded_workbook_overrides_the_default(workbook_path):
    at = st_testing.AppTest.from_file(MAIN, default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(workbook_path))
    _click(at, "Load data")
    at.run()
    assert not at.exception
    assert at.sidebar.success
    # the "using the bundled default" banner must disappear once overridden
    assert not any("default dataset" in c.value for c in at.sidebar.caption)


def test_pages_show_empty_state_without_data():
    """Regression guard for the empty-state code path: if the bundled
    default were ever missing, the app must show a designed empty state,
    not a stack trace. AppTest re-executes main.py from scratch each run, so
    DEFAULT_WORKBOOK_PATH is recomputed from the file on disk — the only way
    to exercise the "missing" branch is to move the real file aside."""
    default_path = (Path(__file__).resolve().parents[1] / "supplyradar"
                    / ".." / "data" / "default_input_public.xlsx").resolve()
    backup_path = default_path.with_suffix(".xlsx.bak")
    assert default_path.exists(), "bundled default dataset is missing"
    default_path.rename(backup_path)
    try:
        at = st_testing.AppTest.from_file(MAIN, default_timeout=60)
        at.run()
        assert not at.exception
        assert at.info  # designed empty state, not a stack trace
    finally:
        backup_path.rename(default_path)

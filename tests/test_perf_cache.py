"""Performance-regression guards for the caching layer.

The app's "heavy" complaint came from rebuilding the pipeline figure on every
rerun. These tests pin the fix: a second call with identical arguments must
NOT rebuild (call-counter assertion) and must return the very same object
(st.cache_resource is zero-copy).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("streamlit")

from tests.test_pipeline_chart import make_projection  # noqa: E402


def test_pipeline_figure_cache_hits_without_rebuild(monkeypatch):
    from supplyradar.app import state as S

    calls = {"n": 0}
    real = S.build_pipeline_chart

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(S, "build_pipeline_chart", counting)
    S.cached_pipeline_figure.clear()

    proj = make_projection(n_items=3)
    items = tuple(sorted(proj["item_code"].unique()))
    key = ("fp-test", "2026-01-01", ())

    f1 = S.cached_pipeline_figure(key, proj, None, {}, {}, items,
                                  "day", 110, 20)
    f2 = S.cached_pipeline_figure(key, proj, None, {}, {}, items,
                                  "day", 110, 20)
    assert calls["n"] == 1, "second identical call must be a cache hit"
    assert f1 is f2, "cache_resource must return the same object (zero-copy)"

    # a different applied choice (item subset) is a different cache entry
    f3 = S.cached_pipeline_figure(key, proj, None, {}, {}, items[:2],
                                  "day", 110, 20)
    assert calls["n"] == 2
    assert f3 is not f1


def test_fingerprint_is_stable_and_cheap():
    from supplyradar.app import state as S

    blob = b"workbook-bytes" * 1000
    assert S.fingerprint_bytes(blob) == S.fingerprint_bytes(blob)
    assert S.fingerprint_bytes(blob) != S.fingerprint_bytes(blob + b"x")

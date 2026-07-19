"""Per-module tutorial mode: highlight each section with an explainer box.

Each page renders a "📘 Tutorial" button next to its title. Clicking it turns
tour mode on for that page: every section then gets a highlighted, numbered
callout box directly above it explaining what the section does and how to use
it. Clicking the button again (now "✖ Exit tutorial") turns the boxes off.

Implementation notes:
- Pure Streamlit — the callouts are styled markdown blocks, so they work in
  every deployment with no JS injection.
- The step counter lives in session_state and is reset at the top of each
  script run (`begin`), so steps stay numbered in reading order.
"""

from __future__ import annotations

import streamlit as st

# Pale amber callouts with a strong left accent and DARK text (readability
# rule: never light font on a light background).
_BOX_CSS = (
    "background-color:#fff7e0;border-left:6px solid #d97706;"
    "border-radius:6px;padding:10px 14px;margin:6px 0 10px 0;"
    "color:#4a3a10;font-size:0.95rem;"
)

def _flag_key(page: str) -> str:
    return f"sr_tour_{page}"

def begin(page: str, title: str) -> bool:
    """Render the page title + the Tutorial toggle button; start the tour
    step counter. Returns True when tour mode is active.

    Use instead of st.title on pages that offer a tutorial:
        active = tutorial.begin("pipeline", "Pipeline Status")
    """
    flag = _flag_key(page)
    st.session_state.setdefault(flag, False)
    st.session_state[f"sr_tour_step_{page}"] = 0  # reset numbering each run

    c1, c2 = st.columns([5, 1])
    c1.title(title)
    label = "✖ Exit tutorial" if st.session_state[flag] else "📘 Tutorial"
    if c2.button(label, key=f"btn_{flag}",
                 help="Walk through this module: every section gets a "
                      "highlighted box explaining what it does."):
        st.session_state[flag] = not st.session_state[flag]
        st.rerun()

    if st.session_state[flag]:
        st.markdown(
            f"<div style='{_BOX_CSS}'><b>📘 Tutorial mode is ON.</b> "
            "The numbered boxes below explain each section of this module. "
            "Click <b>✖ Exit tutorial</b> to hide them.</div>",
            unsafe_allow_html=True)
    return bool(st.session_state[flag])

def tip(page: str, title: str, text: str) -> None:
    """A numbered explainer box for the section that follows it.

    Renders nothing unless the page's tour mode is on, so sprinkling tips
    through a page costs nothing in normal use.
    """
    if not st.session_state.get(_flag_key(page), False):
        return
    step_key = f"sr_tour_step_{page}"
    st.session_state[step_key] = int(st.session_state.get(step_key, 0)) + 1
    n = st.session_state[step_key]
    st.markdown(
        f"<div style='{_BOX_CSS}'><b>📍 Step {n} — {title}.</b> {text}</div>",
        unsafe_allow_html=True)

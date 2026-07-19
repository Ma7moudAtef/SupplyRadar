"""Choice-panel apply system: batch widget changes behind an Apply button.

Why this exists: Streamlit reruns the whole page script on EVERY widget
interaction. With reactive widgets, each click on a multiselect or toggle
triggered a full rerun (and, before caching, a full chart rebuild) — the "app
recomputes on every change" heaviness users complained about.

The fix is `st.form`: widgets inside a form do NOT trigger reruns while the
user is composing choices; their values only reach the script when the form's
submit (Apply) button is pressed. Every choice panel in the app goes through
`panel(...)` below so this behavior is uniform.

Two modes, chosen by the sidebar "Auto-apply changes" toggle:
- MANUAL (default): each panel is a real st.form with an Apply button; nothing
  recomputes until the user clicks it.
- AUTO: panels are plain containers, widgets are reactive again, and the apply
  button is not rendered — every change applies immediately, for users who
  prefer not to click.

Widget keys are identical in both modes, so switching modes preserves every
current selection.
"""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st

# Session key for the global auto-apply preference (False = manual Apply).
AUTO_APPLY_KEY = "sr_auto_apply"

def auto_apply_enabled() -> bool:
    """True when the user opted into reactive (no-button) mode."""
    return bool(st.session_state.get(AUTO_APPLY_KEY, False))

def render_mode_toggle() -> None:
    """The sidebar switch between manual Apply buttons and auto-apply.

    Default is MANUAL: choices take effect only when their panel's Apply
    button is clicked. Rendered once, in the sidebar, by main.py.
    """
    st.sidebar.toggle(
        "Auto-apply changes", key=AUTO_APPLY_KEY, value=False,
        help="Off (default): each choice panel has an Apply button and your "
             "changes only take effect when you click it. On: every change "
             "applies immediately without clicking.")

@contextmanager
def panel(key: str):
    """A choice panel. Use as `with apply.panel('filters'): <widgets> ...
    apply.button('Apply filters')`.

    In manual mode this is an st.form (interactions inside cause NO rerun);
    in auto mode it is a plain container (reactive widgets).
    """
    if auto_apply_enabled():
        with st.container():
            yield
    else:
        # clear_on_submit=False keeps the chosen values visible after Apply.
        with st.form(f"form_{key}", clear_on_submit=False, border=False):
            yield

def button(label: str, *, help: str | None = None, key: str | None = None) -> bool:
    """The panel's Apply button. Must be called INSIDE a `panel(...)` block.

    Manual mode: renders the form's submit button, returns True on the rerun
    it triggered. Auto mode: renders nothing and returns True (changes are
    already live). Callers usually don't need the return value — the widget
    values themselves are the applied state — but it is returned for panels
    that want to react to the click itself.
    """
    if auto_apply_enabled():
        return True
    return st.form_submit_button(
        label, help=help or "Apply the choices above.", type="primary")

def action_button(label: str, *, help: str | None = None,
                  key: str | None = None) -> bool:
    """A click-gated ACTION inside a panel (e.g. 'Run competition').

    Unlike `button`, an action must never fire automatically: in auto-apply
    mode it renders a normal st.button and still waits for the click; in
    manual mode it doubles as the form's submit button (one click applies the
    choices AND starts the action). Returns True only on the click's rerun.
    """
    if auto_apply_enabled():
        return st.button(label, help=help, key=key, type="primary")
    return st.form_submit_button(label, help=help, type="primary")

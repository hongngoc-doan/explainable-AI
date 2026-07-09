"""
unipark_bridge.py
=================

Bridge between Tivian/Unipark EFS and the Streamlit experiment apps.

The Tivian project sends participants in with two parameters:

    https://xai.ntk.fyi/1/?pid=<LFDN>&tic=<TIC>

  * `pid` is the participant's `lfdn` (running number). We log this as the
    participant ID -- it's what you'll merge on against the Unipark export.
  * `tic` is Tivian's return ticket. We never display it; we just keep it
    in session and substitute it into the return URL on completion.

The matching Tivian "Embedded text" link template is:

    https://xai.ntk.fyi/1/?pid=#lfdn#&tic=#tic#
    https://xai.ntk.fyi/2/?pid=#lfdn#&tic=#tic#
    https://xai.ntk.fyi/3/?pid=#lfdn#&tic=#tic#

The Tivian return URL (provided in the EFS configuration) is fixed and
substitutes the same tic value back in:

    https://ww3.unipark.de/uc/StudierendeMitRechten/3220/ospe.php?return_tic=<TIC>

Behaviour:

  * If `tic` and `pid` are both in the URL -> auto-login. The participant
    ID is `pid` (= lfdn), and the return URL is built from `tic`.
  * If only one of them is present -> use what we got, and synthesise a
    reasonable fallback for the missing one (so a slightly malformed link
    still works for piloting).
  * If neither is present -> show a small manual login form, so the apps
    remain usable for development, debugging, and pilot testing outside
    the Tivian flow. The manual session won't have a return URL, so the
    completion screen falls back to the "close this tab" instruction.
"""

from __future__ import annotations
import time
import streamlit as st


# URL parameter names that match the Tivian "Embedded text" template
PARAM_TIC = "tic"   # Tivian's return ticket
PARAM_PID = "pid"   # lfdn -- the participant ID we merge on

# Tivian's return endpoint. {tic} is substituted with the value we received
# in `?tic=`. If the survey moves to a different EFS project, this is the
# only constant that needs to change.
RETURN_URL_TEMPLATE = (
    "https://ww3.unipark.de/uc/StudierendeMitRechten/3220/ospe.php"
    "?return_tic={tic}"
)


def _qp_get(key: str, default: str = "") -> str:
    """Read a single query-parameter value as a string. Tolerates the new
    dict-like st.query_params (Streamlit >= 1.30) and the legacy
    experimental_get_query_params API."""
    try:
        val = st.query_params.get(key, default)
        if isinstance(val, list):
            val = val[-1] if val else default
        return str(val) if val is not None else default
    except Exception:
        try:
            qp = st.experimental_get_query_params()  # type: ignore[attr-defined]
            v = qp.get(key, [default])
            return str(v[0]) if isinstance(v, list) and v else default
        except Exception:
            return default


def _do_login(pid: str, tic: str, condition: str) -> None:
    """Common login routine shared by URL-bootstrap and manual login paths."""
    st.session_state["logged_in"]   = True
    st.session_state["username"]    = pid     # logged as Participant ID for merging
    st.session_state["tic"]         = tic     # used to build the return URL
    st.session_state["condition"]   = condition.upper()
    # Per-participant random seed for case-order shuffling. Stable within a
    # session because session_state persists across reruns.
    seed_basis = (pid or tic or "anonymous") + str(time.time())
    st.session_state["session_seed"] = abs(hash(seed_basis)) % (2**31)


def bootstrap_session_from_url(condition: str) -> None:
    """
    Auto-login from the Tivian URL parameters if present; otherwise fall
    through to a small manual login form. Must be called once near the top
    of the Streamlit app, after session_state has been initialised.

    `condition` is "A" / "B" / "C" and is stored in session_state for
    downstream logging. There is no per-condition URL check -- each
    Streamlit URL (/1/, /2/, /3/) is condition-locked at deployment.
    """
    if st.session_state.get("logged_in", False):
        return  # already in a session, nothing to do

    tic = _qp_get(PARAM_TIC).strip()
    pid = _qp_get(PARAM_PID).strip()

    # --- Path 1: came from Tivian, both params present ---------------------
    if pid and tic:
        _do_login(pid=pid, tic=tic, condition=condition)
        return

    # --- Path 2: partial parameters (tolerate either alone) ----------------
    # If only `pid` arrived, log them in but they won't have a return URL.
    # If only `tic` arrived, use it as both ID and return ticket so the
    # session is still linkable to Unipark (the tic still encodes the lfdn).
    if pid and not tic:
        _do_login(pid=pid, tic="", condition=condition)
        return
    if tic and not pid:
        _do_login(pid=tic, tic=tic, condition=condition)
        return

    # --- Path 3: no URL parameters -- manual login form --------------------
    # This path keeps the app usable for piloting, debugging, and any
    # participant who needs to enter their ID by hand. Manual sessions
    # have no return URL, so the completion screen falls back to the
    # "close this tab" instruction.
    st.markdown(
        f"""
        <div style='text-align:center;margin-top:14vh;'>
          <span style='font-size:3rem;'>🏢</span>
          <h1 style='margin:10px 0 4px;font-weight:800;
                     background:linear-gradient(135deg,#7c3aed,#0284c7);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            Retention Specialist Copilot
          </h1>
          <p style='color:#64748b;font-size:1rem;margin:0;'>
            System {condition.upper()} -- please sign in to continue
          </p>
          <p style='color:#94a3b8;font-size:.82rem;margin:14px auto 0;max-width:420px;'>
            If you reached this page from the Unipark survey, please
            return to the survey link and click through again -- your
            participant ID should be passed automatically.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1, 1])
    with col:
        with st.form("manual_login_form"):
            user = st.text_input("Participant ID", placeholder="e.g. P001")
            submit = st.form_submit_button(
                "Access Workspace", use_container_width=True, type="primary"
            )
            if submit and user.strip():
                _do_login(pid=user.strip(), tic="", condition=condition)
                st.rerun()
    st.stop()


def get_return_url() -> str | None:
    """Build the Tivian return URL using the participant's ticket. Returns
    None if no ticket is in session (manual-login sessions have none)."""
    tic = st.session_state.get("tic", "")
    if not tic:
        return None
    return RETURN_URL_TEMPLATE.format(tic=tic)


def render_return_button(label: str = "✅ Return to Survey") -> None:
    """
    Render the completion-screen call to action. If we have a return ticket,
    show a big "Return to Survey" button that takes the participant back
    to the Unipark survey. If we don't (e.g. manual-login pilot session),
    fall back to the original "close this tab" instruction so the app still
    behaves sensibly outside the Tivian flow.
    """
    url = get_return_url()
    if url:
        st.markdown(
            f"""
            <div style='text-align:center;margin:24px auto 0;max-width:600px;'>
              <a href="{url}" target="_self" rel="noopener"
                 style="display:inline-block;background:linear-gradient(135deg,#7c3aed,#0284c7);
                        color:#fff;padding:14px 32px;border-radius:12px;
                        text-decoration:none;font-weight:700;font-size:1.05rem;
                        box-shadow:0 4px 14px rgba(124,58,237,.35);">
                {label}
              </a>
              <p style='color:#64748b;font-size:.85rem;margin:14px 0 0;'>
                Clicking this button takes you back to the survey to complete
                the remaining questions. Your decisions have been saved.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style='background:linear-gradient(135deg,#f0f9ff,#ede9fe);
                        border:1.5px solid #bae6fd;border-radius:16px;
                        padding:28px 32px;text-align:center;
                        max-width:600px;margin:0 auto;'>
              <span style='font-size:2.2rem;'>📋</span>
              <h2 style='color:#0f172a;margin:10px 0 6px;font-weight:700;
                         font-size:1.25rem;'>Task Complete</h2>
              <p style='color:#475569;margin:0 0 10px;font-size:.95rem;
                        line-height:1.6;'>Your decisions have been recorded.
                You may close this tab now.</p>
              <p style='color:#94a3b8;font-size:.8rem;font-weight:500;margin:0;'>
                (No survey return URL was provided for this session.)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

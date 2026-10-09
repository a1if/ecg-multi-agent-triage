"""ECG multi-agent triage: Streamlit frontend. A thin client of the API service.

Run (with the API running):  streamlit run frontend/app.py      env: API_URL (default http://127.0.0.1:8000)

How Streamlit works, in one paragraph: the script runs top to bottom on every interaction (a click, a typed
question) and redraws the page. Anything that must survive a rerun lives in ``st.session_state``. A
``@st.fragment(run_every=...)`` re-runs just one part of the page on a timer, which is how the live agent trace
updates without redrawing everything else.
"""
import streamlit as st

import api
import ui

st.set_page_config(page_title="ECG multi-agent triage", page_icon="🫀", layout="wide")

pages = st.navigation([
    st.Page("views/worklist.py", title="Worklist", icon="📋", default=True),
    st.Page("views/run.py", title="Recording", icon="🫀", url_path="run"),
    st.Page("views/about.py", title="How it works", icon="🧭", url_path="about"),
])

with st.sidebar:
    st.caption(ui.DISCLAIMER)
    try:
        st.session_state["status"] = api.status()
    except api.ApiError:
        st.session_state["status"] = None
    s = st.session_state["status"]
    if s:
        st.caption(f"Receiver: **{s['receiver']}** · planner: {s['planner']} · questions: {s['classifier']}"
                   f" · uploads: {'on' if s['uploads'] else 'off'}")

pages.run()

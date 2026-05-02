import time
import streamlit as st
from contextlib import contextmanager


@contextmanager
def profile_step(label):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start

    if "profile_logs" not in st.session_state:
        st.session_state.profile_logs = []

    st.session_state.profile_logs.append({
        "step": label,
        "seconds": round(elapsed, 3)
    })


def clear_profile_logs():
    st.session_state.profile_logs = []


def get_profile_logs():
    return st.session_state.get("profile_logs", [])
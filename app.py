import streamlit as st
from src.frontend.layout import render_layout
from src.frontend.components import render_form, render_results


def main():
    st.set_page_config(
        page_title="Symphony DX - Medical Triage",
        page_icon="🩺",
        layout="wide",
    )

    # Initialize session state
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "submission_count" not in st.session_state:
        st.session_state.submission_count = 0

    render_layout()

    # Two-column layout: input on left, results on right
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        render_form()

    with col_right:
        render_results(st.session_state.prediction, st.session_state.submission_count)


if __name__ == "__main__":
    main()
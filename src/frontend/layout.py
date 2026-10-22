import streamlit as st

def render_layout():
    """Render the main page layout with clean header and sidebar."""

    # ── Page header ──────────────────────────────
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
            <h1 style="margin: 0; font-size: 2.5rem;">🩺 Symphony DX</h1>
            <p style="color: #666; font-size: 1.05rem; margin-top: 0.4rem;">
                AI-Powered Medical Triage
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Sidebar ──────────────────────────────────
    with st.sidebar:
        st.markdown("### About")
        st.write(
            "Symphony DX is a research project exploring how "
            "patient information affects AI-based medical triage accuracy."
        )

        st.markdown("---")
        st.markdown("### How It Works")
        st.markdown(
            "1. Describe your symptoms on the left\n"
            "2. Click **Get Triage Recommendation**\n"
            "3. See your result on the right"
        )

        st.markdown("---")
        st.caption(
            "⚠️ **Disclaimer:** This app is for research purposes only. "
            "It is not a substitute for professional medical advice. "
            "Always consult a qualified healthcare provider."
        )
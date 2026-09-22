import joblib
import streamlit as st
from datetime import datetime


# ─────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained baseline model (cached so it loads only once)."""
    return joblib.load("models/baseline_model.pkl")


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict_triage(symptoms_text):
    """Predict ESI triage level from symptom text."""
    model = load_model()

    pred = int(model.predict([symptoms_text])[0])
    proba = model.predict_proba([symptoms_text])[0]
    confidence = float(max(proba))

    esi_to_app = {
        1: "emergency",
        2: "emergency",
        3: "urgent_care",
        4: "doctor",
        5: "self_care",
    }

    return {
        "triage_level": esi_to_app.get(pred, "urgent_care"),
        "esi_score": pred,
        "confidence": confidence,
        "symptoms": symptoms_text,
        "model_type": "Baseline (Symptoms Only)",
    }


# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
def render_form():
    """Render the symptom input form (left column)."""

    st.markdown("#### 📝 Describe Your Symptoms")
    st.caption("Provide as much detail as you can — duration, severity, "
               "and any related symptoms.")

    with st.form(key="symptom_form", clear_on_submit=False):
        symptoms = st.text_area(
            "Your symptoms",
            placeholder="e.g., I've had a severe headache for two days, "
                        "along with nausea and sensitivity to light.",
            height=200,
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "Get Triage Recommendation",
            type="primary",
            use_container_width=True,
        )

        if submitted:
            if symptoms.strip():
                prediction = predict_triage(symptoms.strip())
                prediction["timestamp"] = datetime.now().strftime("%H:%M:%S")
                st.session_state.prediction = prediction
                st.session_state.submission_count += 1
                st.rerun()
            else:
                st.error("Please describe your symptoms before submitting.")


# ─────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────
def render_results(prediction, submission_count):
    """Display prediction results (right column)."""

    # No results yet
    if not prediction:
        st.markdown("#### 📋 Triage Recommendation")
        st.info(
            "👈 Enter your symptoms on the left and click "
            "**Get Triage Recommendation** to see your result here."
        )
        return

    # ── Triage categories ────────────────────────
    triage_levels = {
        "self_care": {
            "label": "Self-Care",
            "emoji": "🟢",
            "advice": "Manage at home with rest and over-the-counter remedies.",
            "color": "#28a745",
        },
        "doctor": {
            "label": "See a Doctor",
            "emoji": "🟡",
            "advice": "Schedule an appointment with your primary care physician.",
            "color": "#ffc107",
        },
        "urgent_care": {
            "label": "Urgent Care",
            "emoji": "🟠",
            "advice": "Visit urgent care within 24 hours.",
            "color": "#fd7e14",
        },
        "emergency": {
            "label": "Emergency",
            "emoji": "🔴",
            "advice": "Seek emergency medical care immediately.",
            "color": "#dc3545",
        },
    }

    urgency_names = {
        1: "Immediate",
        2: "High",
        3: "Moderate",
        4: "Low",
        5: "Very Low",
    }

    level = prediction.get("triage_level", "urgent_care")
    info = triage_levels[level]
    esi = prediction.get("esi_score", 3)
    urgency = urgency_names.get(esi, "Moderate")
    confidence = prediction.get("confidence", 0.0)

    # ── Compact "new result" indicator (small text, no box) ──
    st.markdown(
        f"""
        <div style="
            color: #28a745;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        ">
            ✅ New result · #{submission_count} · {prediction.get('timestamp', '')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Single result card ───────────────────────
    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {info['color']};
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        ">
            <h2 style="margin: 0; color: {info['color']}; font-size: 1.7rem;">
                {info['emoji']} {info['label']}
            </h2>
            <p style="margin: 0.6rem 0 0 0; color: #333; font-size: 1.05rem;">
                {info['advice']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Prominent metrics row ────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 0.9rem 1rem;
                text-align: center;
            ">
                <div style="font-size: 0.8rem; color: #666;
                            text-transform: uppercase;
                            letter-spacing: 0.5px;">
                    Urgency
                </div>
                <div style="font-size: 1.4rem; font-weight: 700;
                            color: {info['color']}; margin-top: 0.25rem;">
                    {urgency}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 0.9rem 1rem;
                text-align: center;
            ">
                <div style="font-size: 0.8rem; color: #666;
                            text-transform: uppercase;
                            letter-spacing: 0.5px;">
                    Confidence
                </div>
                <div style="font-size: 1.4rem; font-weight: 700;
                            color: #333; margin-top: 0.25rem;">
                    {confidence:.0%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Educational expander ─────────────────────
    with st.expander("ℹ️ What do these levels mean?"):
        st.markdown(
            """
            **How we classify urgency**

            Symphony DX estimates how urgently you may need care based on the
            symptoms you describe:

            | Level | Urgency | What it means |
            |---|---|---|
            | 🔴 | Immediate | Life-threatening — needs care right now |
            | 🔴 | High | Potentially serious — go to the ER |
            | 🟠 | Moderate | See a provider within 24 hours |
            | 🟡 | Low | Schedule a doctor's appointment |
            | 🟢 | Very Low | Manage at home with self-care |

            **Note:** This tool provides an *estimate* based on the symptoms
            you describe. It does not replace professional medical judgment.
            """
        )

    # ── Disclaimer ───────────────────────────────
    st.warning(
        "⚠️ **Disclaimer:** This app is for research purposes only. "
        "It is not a substitute for professional medical advice."
    )
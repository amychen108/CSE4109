import joblib
import re
import pandas as pd
import streamlit as st
from datetime import datetime

import sklearn.compose._column_transformer as column_transformer


if not hasattr(column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass

    column_transformer._RemainderColsList = _RemainderColsList


# ─────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load available models, falling back to the text-only baseline."""
    models = {}
    for name in ["m1", "m2", "m3", "m4"]:
        try:
            models[name] = joblib.load(f"models/{name}.pkl")
        except Exception as error:
            if name == "m1":
                try:
                    models[name] = joblib.load("models/demo1_model.pkl")
                    st.warning(
                        "Personalized models are unavailable in this environment. "
                        "Using the text-only baseline model."
                    )
                except Exception:
                    models[name] = None
            else:
                models[name] = None
            if name != "m1":
                st.info(f"Model {name} is unavailable: {error}")
    return models


# ─────────────────────────────────────────────
# Feature Extraction Helpers
# ─────────────────────────────────────────────
def count_meds(meds_text):
    if not meds_text or not meds_text.strip():
        return 0
    parts = re.split(r"[;\n,]", meds_text)
    return len([p for p in parts if p.strip()])


def extract_med_flags(meds_text):
    txt = (meds_text or "").lower()
    return {
        "has_insulin":   int(bool(re.search(r"insulin", txt))),
        "has_anticoag":  int(bool(re.search(r"heparin|warfarin|apixaban|rivaroxaban|enoxaparin|dabigatran", txt))),
        "has_statin":    int(bool(re.search(r"atorvastatin|rosuvastatin|simvastatin|pravastatin|lovastatin", txt))),
        "has_bp_med":    int(bool(re.search(r"lisinopril|losartan|metoprolol|amlodipine|atenolol|hydrochlorothiazide|carvedilol|valsartan|olmesartan|propranolol", txt))),
        "has_steroid":   int(bool(re.search(r"prednisone|methylprednisolone|hydrocortisone|dexamethasone", txt))),
        "has_inhaler":   int(bool(re.search(r"albuterol|fluticasone|ipratropium|budesonide|salmeterol|tiotropium", txt))),
        "has_opioid":    int(bool(re.search(r"oxycodone|hydrocodone|morphine|fentanyl|tramadol|codeine", txt))),
        "has_diuretic":  int(bool(re.search(r"furosemide|bumetanide|spironolactone|torsemide", txt))),
    }


def select_model_name(has_demo, has_meds, has_vitals):
    if has_vitals:
        return "m4"
    if has_meds:
        return "m3"
    if has_demo:
        return "m2"
    return "m1"


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict_triage(inputs):
    models = load_models()

    symptoms = inputs.get("symptoms", "").strip()
    age      = inputs.get("age")
    gender   = inputs.get("gender") or None
    race     = inputs.get("race") or None
    meds     = inputs.get("meds") or ""
    temp     = inputs.get("temp")
    hr       = inputs.get("hr")
    rr       = inputs.get("rr")
    spo2     = inputs.get("spo2")
    sbp      = inputs.get("sbp")
    dbp      = inputs.get("dbp")

    # Determine which model to use based on what user provided
    has_demo   = any([age is not None, gender, race])
    has_meds   = count_meds(meds) > 0
    has_vitals = any([temp is not None, hr is not None, rr is not None,
                      spo2 is not None, sbp is not None, dbp is not None])

    model_name = select_model_name(has_demo, has_meds, has_vitals)
    model = models.get(model_name)
    if model is None:
        if model_name == "m1":
            message = "The text-only baseline model is unavailable."
        else:
            message = (
                f"Model {model_name.upper()} is unavailable in this environment. "
                "Install the XGBoost OpenMP runtime, then restart the app."
            )
        st.error(message)
        return None

    # Fill defaults so pipeline never crashes on NaN
    row = {
        "HPI": symptoms,
        "age": age if age is not None else 65,
        "gender": gender if gender else "Female",
        "race": race if race else "WHITE",
    }

    # Meds
    row["num_meds"] = count_meds(meds)
    row.update(extract_med_flags(meds))

    # Vitals + derived
    row["temp"] = temp if temp is not None else 37.0
    row["hr"]   = hr   if hr   is not None else 80
    row["rr"]   = rr   if rr   is not None else 16
    row["spo2"] = spo2 if spo2 is not None else 98
    row["sbp"]  = sbp  if sbp  is not None else 120
    row["dbp"]  = dbp  if dbp  is not None else 80

    row["map"]            = (row["sbp"] + 2 * row["dbp"]) / 3
    row["shock_index"]    = row["hr"] / row["sbp"] if row["sbp"] else 1.0
    row["pulse_pressure"] = row["sbp"] - row["dbp"]

    df_row = pd.DataFrame([row])

    # Predict
    try:
        raw_pred = int(model.predict(df_row)[0])
        proba = model.predict_proba(df_row)[0]
    except Exception:
        # Fallback for legacy demo1_model.pkl (list input, LR labels 1–4)
        raw_pred = int(model.predict([symptoms])[0])
        proba = model.predict_proba([symptoms])[0]

    # Demo 2 pipelines (ColumnTransformer) use XGBoost with 0-indexed labels
    is_demo2 = "prep" in model.named_steps
    esi = raw_pred + 1 if is_demo2 else raw_pred

    confidence = float(max(proba))

    esi_to_app = {
        1: "emergency", 2: "emergency",
        3: "urgent_care", 4: "doctor", 5: "self_care",
    }

    model_labels = {
        "m1": "M1 — Baseline (Symptoms Only)",
        "m2": "M2 — + Demographics",
        "m3": "M3 — + Medical History",
        "m4": "M4 — Full Personalization",
    }

    return {
        "triage_level":  esi_to_app.get(esi, "urgent_care"),
        "esi_score":     esi,
        "confidence":    confidence,
        "symptoms":      symptoms,
        "model_type":    model_labels[model_name],
        "model_name":    model_name,
    }


# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
def render_form():
    st.markdown("#### 📝 Describe Your Symptoms")
    st.caption(
        "Start with your symptoms. Optionally add patient information, "
        "medical history, or vital signs for a more personalized recommendation."
    )

    with st.form(key="symptom_form", clear_on_submit=False):
        # ── Required: symptoms ──
        symptoms = st.text_area(
            "Your symptoms *",
            placeholder="e.g., I've had a severe headache for two days, "
                        "along with nausea and sensitivity to light.",
            height=140,
        )

        # ── Optional: demographics ──
        with st.expander("👤 Patient Information (optional)"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input(
                    "Age", min_value=0, max_value=120,
                    value=None, step=1,
                    help="Leave empty if you prefer not to say",
                )
            with c2:
                gender = st.selectbox(
                    "Gender", ["", "Female", "Male", "Other"], index=0,
                )
            race = st.selectbox(
                "Race", ["", "WHITE", "BLACK", "ASIAN",
                         "HISPANIC", "OTHER"], index=0,
            )

        # ── Optional: medical history ──
        with st.expander("💊 Medical History (optional)"):
            meds = st.text_area(
                "Past or current medications (one per line)",
                placeholder="e.g.,\nMetoprolol 50mg daily\n"
                            "Atorvastatin 20mg\nInsulin sliding scale",
                height=100,
            )

        # ── Optional: vitals ──
        with st.expander("🩺 Vital Signs (optional)"):
            c1, c2, c3 = st.columns(3)
            with c1:
                temp = st.number_input("Temperature (°C)", min_value=30.0,
                                       max_value=45.0, value=None,
                                                                             step=0.1)
                hr = st.number_input("Heart Rate (bpm)", min_value=20,
                                     max_value=250, value=None,
                                                                         step=1)
            with c2:
                rr = st.number_input("Resp. Rate (breaths/min)", min_value=4,
                                     max_value=60, value=None,
                                                                         step=1)
                spo2 = st.number_input("SpO₂ (%)", min_value=50, max_value=100,
                                                                             value=None, step=1)
            with c3:
                sbp = st.number_input("Systolic BP (mmHg)", min_value=40,
                                      max_value=300, value=None,
                                                                            step=1)
                dbp = st.number_input("Diastolic BP (mmHg)", min_value=20,
                                      max_value=200, value=None,
                                                                            step=1)

        submitted = st.form_submit_button(
            "Get Triage Recommendation",
            type="primary",
            use_container_width=True,
        )

        if submitted:
            if not symptoms.strip():
                st.error("Please describe your symptoms before submitting.")
            else:
                inputs = {
                    "symptoms": symptoms.strip(),
                    "age": age, "gender": gender, "race": race,
                    "meds": meds,
                    "temp": temp, "hr": hr, "rr": rr,
                    "spo2": spo2, "sbp": sbp, "dbp": dbp,
                }
                prediction = predict_triage(inputs)
                if prediction:
                    prediction["timestamp"] = datetime.now().strftime("%H:%M:%S")
                    st.session_state.prediction = prediction
                    st.session_state.submission_count += 1
                    st.rerun()


# ─────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────
def render_results(prediction, submission_count):
    if not prediction:
        st.markdown("#### 📋 Triage Recommendation")
        st.info(
            "👈 Enter your symptoms on the left and click "
            "**Get Triage Recommendation** to see your result here."
        )
        return

    triage_levels = {
        "self_care":   {"label": "Self-Care",   "emoji": "🟢",
                        "advice": "Manage at home with rest and over-the-counter remedies.",
                        "color": "#28a745"},
        "doctor":      {"label": "See a Doctor", "emoji": "🟡",
                        "advice": "Schedule an appointment with your primary care physician.",
                        "color": "#ffc107"},
        "urgent_care": {"label": "Urgent Care",  "emoji": "🟠",
                        "advice": "Visit urgent care within 24 hours.",
                        "color": "#fd7e14"},
        "emergency":   {"label": "Emergency",    "emoji": "🔴",
                        "advice": "Seek emergency medical care immediately.",
                        "color": "#dc3545"},
    }

    urgency_names = {1: "Immediate", 2: "High", 3: "Moderate",
                     4: "Low", 5: "Very Low"}

    level      = prediction.get("triage_level", "urgent_care")
    info       = triage_levels[level]
    esi        = prediction.get("esi_score", 3)
    urgency    = urgency_names.get(esi, "Moderate")
    confidence = prediction.get("confidence", 0.0)

    # ── New result indicator ──
    st.markdown(
        f"""
        <div style="color: #28a745; font-size: 0.85rem;
                    font-weight: 600; margin-bottom: 0.5rem;">
            ✅ New result · #{submission_count} · {prediction.get('timestamp', '')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Result card ──
    st.markdown(
        f"""
        <div style="border-left: 6px solid {info['color']};
                    background-color: #f8f9fa; border-radius: 8px;
                    padding: 1.25rem 1.5rem; margin-bottom: 1rem;">
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

    # ── Metrics ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #e0e0e0;
                        border-radius: 8px; padding: 0.9rem 1rem; text-align: center;">
                <div style="font-size: 0.8rem; color: #666;
                            text-transform: uppercase; letter-spacing: 0.5px;">
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
    with c2:
        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #e0e0e0;
                        border-radius: 8px; padding: 0.9rem 1rem; text-align: center;">
                <div style="font-size: 0.8rem; color: #666;
                            text-transform: uppercase; letter-spacing: 0.5px;">
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

    # ── Model badge ──
    st.markdown(
        f"""
        <div style="margin-top: 0.75rem; padding: 0.5rem 0.75rem;
                    background-color: #e8f4f8; border-radius: 6px;
                    font-size: 0.85rem; color: #333;">
            <strong>Model used:</strong> {prediction.get('model_type', 'M1 — Baseline')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Educational expander ──
    with st.expander("ℹ️ What do these levels mean?"):
        st.markdown(
            """
            **How we classify urgency**

            | Level | Urgency | What it means |
            |---|---|---|
            | 🔴 | Immediate | Life-threatening — needs care right now |
            | 🔴 | High | Potentially serious — go to the ER |
            | 🟠 | Moderate | See a provider within 24 hours |
            | 🟡 | Low | Schedule a doctor's appointment |
            | 🟢 | Very Low | Manage at home with self-care |

            **About personalization:** The model used depends on how much
            information you provide. Adding patient details, medical history,
            or vitals enables a more personalized recommendation.
            """
        )

    st.warning(
        "⚠️ **Disclaimer:** This app is for research purposes only. "
        "It is not a substitute for professional medical advice."
    )
"""
app.py  —  Heart Failure Prediction  |  Streamlit Web Application
-----------------------------------------------------------------
Run:  streamlit run app.py
Requires: models/rf_heart_failure.pkl and models/scaler.pkl
          (generate them first with:  python train_model.py)
"""

import numpy as np
import pandas as pd
import joblib
import os
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #c0392b;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 5px solid #c0392b;
        margin-bottom: 1rem;
    }
    .result-survived {
        background: #d4edda;
        border-left: 6px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        font-size: 1.3rem;
        font-weight: 600;
        color: #155724;
    }
    .result-deceased {
        background: #f8d7da;
        border-left: 6px solid #dc3545;
        border-radius: 10px;
        padding: 1.5rem;
        font-size: 1.3rem;
        font-weight: 600;
        color: #721c24;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        border-bottom: 2px solid #c0392b;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("models", "rf_heart_failure.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

# ── Feature metadata ───────────────────────────────────────────────────────────
FEATURES = [
    "age", "anaemia", "CPK", "diabetes", "ejection_fraction",
    "high_blood_pressure", "platelets", "serum_creatinine",
    "serum_sodium", "sex", "smoking", "time"
]

FEATURE_META = {
    "age":               {"label": "Age", "unit": "years",              "type": "numeric", "min": 40,   "max": 95,   "default": 60,    "step": 1},
    "anaemia":           {"label": "Anaemia",                            "type": "binary"},
    "CPK":               {"label": "Creatinine Phosphokinase (CPK)", "unit": "mcg/L",    "type": "numeric", "min": 20.0, "max": 8000.0,"default": 250.0, "step": 10.0},
    "diabetes":          {"label": "Diabetes",                           "type": "binary"},
    "ejection_fraction": {"label": "Ejection Fraction", "unit": "%",     "type": "numeric", "min": 10,   "max": 80,   "default": 38,    "step": 1},
    "high_blood_pressure":{"label": "High Blood Pressure",               "type": "binary"},
    "platelets":         {"label": "Platelets", "unit": "kiloplatelets/mL","type": "numeric","min": 25.0,"max": 850.0,"default": 262.0, "step": 5.0},
    "serum_creatinine":  {"label": "Serum Creatinine", "unit": "mg/dL",  "type": "numeric", "min": 0.5,  "max": 9.4,  "default": 1.1,   "step": 0.1},
    "serum_sodium":      {"label": "Serum Sodium", "unit": "mEq/L",      "type": "numeric", "min": 113,  "max": 148,  "default": 136,   "step": 1},
    "sex":               {"label": "Sex (1 = Male, 0 = Female)",         "type": "binary"},
    "smoking":           {"label": "Smoking",                            "type": "binary"},
    "time":              {"label": "Follow-up Period", "unit": "days",   "type": "numeric", "min": 4,    "max": 285,  "default": 115,   "step": 1},
}

# Normal clinical reference ranges for annotation
NORMAL_RANGES = {
    "CPK":               (10, 120),
    "ejection_fraction": (55, 70),
    "platelets":         (150, 400),
    "serum_creatinine":  (0.6, 1.2),
    "serum_sodium":      (135, 145),
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def random_patient():
    """Generate a synthetic patient with plausible clinical values."""
    rng = np.random.default_rng()
    return {
        "age":                float(rng.integers(40, 95)),
        "anaemia":            int(rng.integers(0, 2)),
        "CPK":                float(rng.choice([
                                  rng.uniform(20, 120),     # normal
                                  rng.uniform(120, 7861),   # elevated
                              ], p=[0.4, 0.6])),
        "diabetes":           int(rng.integers(0, 2)),
        "ejection_fraction":  float(rng.integers(14, 80)),
        "high_blood_pressure":int(rng.integers(0, 2)),
        "platelets":          float(rng.uniform(25, 850)),
        "serum_creatinine":   float(rng.choice([
                                  rng.uniform(0.5, 1.2),    # normal
                                  rng.uniform(1.2, 9.4),    # elevated
                              ], p=[0.45, 0.55])),
        "serum_sodium":       float(rng.integers(113, 148)),
        "sex":                int(rng.integers(0, 2)),
        "smoking":            int(rng.integers(0, 2)),
        "time":               float(rng.integers(4, 285)),
    }


def build_feature_vector(vals: dict) -> np.ndarray:
    return np.array([[vals[f] for f in FEATURES]])


def predict(vals: dict):
    vec = build_feature_vector(vals)
    vec_scaled = scaler.transform(vec)
    prob = model.predict_proba(vec_scaled)[0]      # [p_survived, p_deceased]
    pred = int(np.argmax(prob))
    return pred, prob


def gauge_chart(probability: float, label: str, color: str):
    """Draw a semicircular gauge showing death probability."""
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw=dict(aspect="equal"))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.1, 1.3)

    # Background arc
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#e0e0e0", linewidth=18, solid_capstyle="round")

    # Filled arc up to probability
    fill_end = np.pi - probability * np.pi
    theta_fill = np.linspace(np.pi, fill_end, 300)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, linewidth=18, solid_capstyle="round")

    # Needle
    angle = np.pi - probability * np.pi
    ax.plot([0, 0.75 * np.cos(angle)], [0, 0.75 * np.sin(angle)], color="#2c3e50", linewidth=2.5)
    ax.add_patch(plt.Circle((0, 0), 0.06, color="#2c3e50", zorder=5))

    ax.text(0, 0.2, f"{probability*100:.1f}%", ha="center", va="center",
            fontsize=22, fontweight="bold", color=color)
    ax.text(0, -0.05, label, ha="center", va="center",
            fontsize=10, color="#555")
    ax.text(-1.1, -0.05, "0%", ha="center", fontsize=8, color="#888")
    ax.text(1.1, -0.05, "100%", ha="center", fontsize=8, color="#888")

    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    return fig


def importance_chart(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    colors = ["#c0392b" if v > importances.median() else "#aec6cf" for v in importances.values]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.barh(importances.index, importances.values, color=colors, edgecolor="white")
    ax.set_xlabel("Importance", fontsize=9)
    ax.set_title("Feature Importances\n(Random Forest)", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    high_patch = mpatches.Patch(color="#c0392b", label="Above median")
    low_patch  = mpatches.Patch(color="#aec6cf", label="Below median")
    ax.legend(handles=[high_patch, low_patch], fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">Heart Failure Survival Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Clinical decision support tool using a Random Forest ensemble model trained on the UCI Heart Failure Clinical Records dataset (Chicco & Jurman, 2020).</p>', unsafe_allow_html=True)

if model is None:
    st.error(
        "**Model not found.** Please run `python train_model.py` first to generate "
        "`models/rf_heart_failure.pkl` and `models/scaler.pkl`."
    )
    st.stop()

# ── Sidebar: model info ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About the Model")
    st.markdown("""
**Algorithm**: Random Forest (300 trees)
**Training**: SMOTE-balanced, 70/30 split
**Dataset**: 299 patients, 12 features

**Performance (held-out test set)**
- Accuracy: ~83%
- ROC-AUC: ~0.90

**Key Predictors**
1. Follow-up time
2. Serum creatinine
3. Ejection fraction
4. Age
5. CPK

---
*This tool is for educational and research purposes only and does not constitute medical advice.*
    """)

    st.markdown("---")
    st.markdown("**Reference**")
    st.caption("Chicco D, Jurman G (2020). BMC Med Inform Decis Mak 20, 16. https://doi.org/10.1186/s12911-020-1023-5")

# ── Two-column layout ──────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1], gap="large")

# ── LEFT: Input form ───────────────────────────────────────────────────────────
with left_col:
    st.markdown('<p class="section-header">Patient Clinical Parameters</p>', unsafe_allow_html=True)

    # Random patient button
    if st.button("🎲  Generate Random Synthetic Patient", use_container_width=True):
        st.session_state["patient"] = random_patient()

    if "patient" not in st.session_state:
        st.session_state["patient"] = {f: FEATURE_META[f].get("default", 0) for f in FEATURES}

    vals = {}

    with st.form("patient_form"):
        st.markdown("**Demographics & History**")
        c1, c2 = st.columns(2)
        with c1:
            vals["age"] = st.number_input(
                "Age (years)", min_value=40, max_value=95,
                value=int(st.session_state["patient"]["age"]), step=1)
            vals["sex"] = st.selectbox(
                "Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male",
                index=int(st.session_state["patient"]["sex"]))
            vals["smoking"] = st.selectbox(
                "Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                index=int(st.session_state["patient"]["smoking"]))
        with c2:
            vals["diabetes"] = st.selectbox(
                "Diabetes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                index=int(st.session_state["patient"]["diabetes"]))
            vals["anaemia"] = st.selectbox(
                "Anaemia", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                index=int(st.session_state["patient"]["anaemia"]))
            vals["high_blood_pressure"] = st.selectbox(
                "High Blood Pressure", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                index=int(st.session_state["patient"]["high_blood_pressure"]))

        st.markdown("**Cardiac & Renal Biomarkers**")
        c3, c4 = st.columns(2)
        with c3:
            vals["ejection_fraction"] = st.slider(
                "Ejection Fraction (%)", min_value=10, max_value=80,
                value=int(st.session_state["patient"]["ejection_fraction"]))
            vals["serum_creatinine"] = st.number_input(
                "Serum Creatinine (mg/dL)", min_value=0.5, max_value=9.4,
                value=float(st.session_state["patient"]["serum_creatinine"]), step=0.1, format="%.1f")
            vals["serum_sodium"] = st.number_input(
                "Serum Sodium (mEq/L)", min_value=113, max_value=148,
                value=int(st.session_state["patient"]["serum_sodium"]), step=1)
        with c4:
            vals["CPK"] = st.number_input(
                "CPK (mcg/L)", min_value=20.0, max_value=8000.0,
                value=float(st.session_state["patient"]["CPK"]), step=10.0, format="%.1f")
            vals["platelets"] = st.number_input(
                "Platelets (kiloplatelets/mL)", min_value=25.0, max_value=850.0,
                value=float(st.session_state["patient"]["platelets"]), step=5.0, format="%.1f")

        st.markdown("**Follow-up**")
        vals["time"] = st.slider(
            "Follow-up Period (days)", min_value=4, max_value=285,
            value=int(st.session_state["patient"]["time"]))

        submitted = st.form_submit_button("Run Prediction", use_container_width=True, type="primary")

# ── RIGHT: Results ─────────────────────────────────────────────────────────────
with right_col:
    st.markdown('<p class="section-header">Prediction Results</p>', unsafe_allow_html=True)

    if submitted or "last_vals" in st.session_state:
        if submitted:
            st.session_state["last_vals"] = vals

        lv = st.session_state.get("last_vals", vals)
        pred, prob = predict(lv)

        p_survived = prob[0]
        p_deceased = prob[1]

        if pred == 0:
            st.markdown(
                f'<div class="result-survived">✅ Predicted Outcome: <b>Survived</b><br>'
                f'Confidence: {p_survived*100:.1f}%</div>', unsafe_allow_html=True)
            gauge_color = "#28a745"
            gauge_label = "Death Risk"
        else:
            st.markdown(
                f'<div class="result-deceased">⚠️ Predicted Outcome: <b>Deceased</b><br>'
                f'Confidence: {p_deceased*100:.1f}%</div>', unsafe_allow_html=True)
            gauge_color = "#dc3545"
            gauge_label = "Death Risk"

        st.markdown("")

        g_col, i_col = st.columns(2)
        with g_col:
            fig_gauge = gauge_chart(p_deceased, gauge_label, gauge_color)
            st.pyplot(fig_gauge, use_container_width=True)
        with i_col:
            fig_imp = importance_chart(model, FEATURES)
            st.pyplot(fig_imp, use_container_width=True)

        # ── Clinical flag table ────────────────────────────────────────────────
        st.markdown('<p class="section-header">Clinical Flag Summary</p>', unsafe_allow_html=True)

        flag_data = []
        for feat, (lo, hi) in NORMAL_RANGES.items():
            val = lv[feat]
            unit_label = FEATURE_META[feat].get("unit", "")
            if val < lo:
                status = "🔵 Below Normal"
            elif val > hi:
                status = "🔴 Above Normal"
            else:
                status = "🟢 Normal"
            flag_data.append({
                "Feature":       FEATURE_META[feat]["label"].split(" (")[0],
                "Value":         f"{val:.1f} {unit_label}",
                "Normal Range":  f"{lo}–{hi} {unit_label}",
                "Status":        status,
            })

        st.dataframe(
            pd.DataFrame(flag_data).set_index("Feature"),
            use_container_width=True,
            hide_index=False,
        )

        # ── Probability breakdown ──────────────────────────────────────────────
        st.markdown('<p class="section-header">Probability Breakdown</p>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Outcome":     ["Survived", "Deceased"],
            "Probability": [f"{p_survived*100:.2f}%", f"{p_deceased*100:.2f}%"],
            "Confidence Bar": [p_survived, p_deceased],
        })
        st.dataframe(prob_df[["Outcome", "Probability"]].set_index("Outcome"), use_container_width=True)

        fig_bar, ax = plt.subplots(figsize=(5, 1.6))
        ax.barh(["Survived", "Deceased"], [p_survived, p_deceased],
                color=["#28a745", "#dc3545"], edgecolor="white", height=0.5)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False, labelsize=9)
        ax.set_xlabel("Probability", fontsize=9)
        fig_bar.tight_layout()
        st.pyplot(fig_bar, use_container_width=True)

    else:
        st.info("Fill in the patient parameters on the left and click **Run Prediction** to see results.")
        st.markdown("""
**What this app does:**
- Accepts 12 routine clinical measurements
- Predicts 30-day mortality following a heart failure episode
- Displays prediction probability, clinical flags, and feature importances

Click **Generate Random Synthetic Patient** to auto-fill a random patient profile.
        """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**Disclaimer**: This application is for educational and research purposes only. "
    "It does not constitute medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare professional for clinical decisions.  \n"
    "Model: Random Forest | Dataset: UCI Heart Failure Clinical Records | "
    "Reference: Chicco & Jurman (2020)"
)

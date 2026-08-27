from pathlib import Path
import sys
import json
from datetime import date, timedelta
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.predict import predict_batch  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models"
if (
    not (MODEL_DIR / "fraud_model.pkl").exists()
    or not (MODEL_DIR / "severity_model.pkl").exists()
):
    st.error("Models not found. Run: python -m src.models.train")
    st.stop()
if not (MODEL_DIR / "metadata.json").exists():
    st.error("Metadata not found. Run: python -m src.models.train")
    st.stop()

MODEL_METADATA = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
FRAUD_REVIEW_THRESHOLD = float(MODEL_METADATA.get("threshold", 0.40))
FRAUD_REJECT_THRESHOLD = float(MODEL_METADATA.get("reject_threshold", 0.70))
RISK_REVIEW_THRESHOLD = 10000.0
RISK_REJECT_THRESHOLD = 25000.0


class PayloadValidationError(ValueError):
    """Raised when payload validation fails."""


def _parse_numeric(value: object, field_name: str, min_value: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PayloadValidationError(f"{field_name} must be numeric.") from exc
    if parsed <= min_value:
        raise PayloadValidationError(f"{field_name} must be greater than {min_value}.")
    return parsed


def score_claim(payload: dict[str, object], model_dir: Path | str | None = None) -> dict[str, object]:
    claim_amount = _parse_numeric(payload.get("CLAIM_AMOUNT"), "CLAIM_AMOUNT")
    premium_amount = _parse_numeric(payload.get("PREMIUM_AMOUNT"), "PREMIUM_AMOUNT")

    loss_dt = pd.to_datetime(payload.get("LOSS_DT"), errors="coerce")
    report_dt = pd.to_datetime(payload.get("REPORT_DT"), errors="coerce")
    if pd.isna(loss_dt) or pd.isna(report_dt):
        raise PayloadValidationError("LOSS_DT and REPORT_DT must be valid dates.")
    if report_dt < loss_dt:
        raise PayloadValidationError("REPORT_DT cannot be before LOSS_DT.")

    scoring_df = pd.DataFrame([payload])
    prediction = predict_batch(scoring_df)

    fraud_probability = float(prediction.loc[0, "fraud_probability"])
    fraud_probability = max(0.0, min(1.0, fraud_probability))
    predicted_severity = float(prediction.loc[0, "predicted_claim_amount"])
    risk_score = float(prediction.loc[0, "risk_score"])
    rejection_score = fraud_probability * 100.0

    if fraud_probability >= FRAUD_REJECT_THRESHOLD or risk_score >= RISK_REJECT_THRESHOLD:
        classification = "Reject"
        risk_band = "High"
        badge_class = "badge-reject"
    elif fraud_probability >= FRAUD_REVIEW_THRESHOLD or risk_score >= RISK_REVIEW_THRESHOLD:
        classification = "Review"
        risk_band = "Medium"
        badge_class = "badge-review"
    else:
        classification = "Approve"
        risk_band = "Low"
        badge_class = "badge-approve"

    confidence_score = max(55.0, abs(fraud_probability - 0.5) * 120.0 + 55.0)
    transparency_fields = [
        "AGE",
        "MARITAL_STATUS",
        "EMPLOYMENT_STATUS",
        "RISK_SEGMENTATION",
        "HOUSE_TYPE",
        "INCIDENT_STATE",
        "INCIDENT_CITY",
    ]
    present = sum(1 for key in transparency_fields if payload.get(key) not in (None, "", " "))
    transparency_score = 60.0 + (present / len(transparency_fields)) * 40.0

    ratio = claim_amount / premium_amount if premium_amount else 0.0
    reasons = []
    if ratio > 30:
        reasons.append("Claim amount is high compared to premium amount.")
    if str(payload.get("INCIDENT_SEVERITY")) == "Total Loss":
        reasons.append("Incident severity is total loss.")
    if int(payload.get("POLICE_REPORT_AVAILABLE", 0)) == 0:
        reasons.append("No police report was provided.")
    if int(payload.get("ANY_INJURY", 0)) == 1:
        reasons.append("Injury was reported for this claim.")
    if not reasons:
        reasons = ["No dominant manual risk triggers were identified from entered fields."]

    warnings = []
    if ratio > 80:
        warnings.append("Very high claim-to-premium ratio; consider manual review.")

    decision_reason = (
        f"Decision based on fraud probability ({fraud_probability:.1%}) and risk score ({risk_score:,.0f})."
    )

    return {
        "fraud_probability": fraud_probability,
        "predicted_severity": predicted_severity,
        "risk_score": risk_score,
        "rejection_score": rejection_score,
        "confidence_score": min(confidence_score, 99.0),
        "transparency_score": min(transparency_score, 100.0),
        "classification": classification,
        "risk_band": risk_band,
        "badge_class": badge_class,
        "threshold_used": {
            "fraud_review_threshold": FRAUD_REVIEW_THRESHOLD,
            "fraud_reject_threshold": FRAUD_REJECT_THRESHOLD,
            "risk_review_threshold": RISK_REVIEW_THRESHOLD,
            "risk_reject_threshold": RISK_REJECT_THRESHOLD,
        },
        "decision_reason": decision_reason,
        "reasons": reasons[:4],
        "warnings": warnings,
        "payload": payload,
    }


if "last_scoring" not in st.session_state:
    st.session_state["last_scoring"] = None
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None

data_dir = PROJECT_ROOT / "data" / "raw"
insurance_type_options = ["Health", "Auto", "Property"]
incident_state_options = [""]
incident_city_options = [""]
insurance_data_path = data_dir / "insurance_data.csv"
if insurance_data_path.exists():
    try:
        insurance_types_df = pd.read_csv(
            insurance_data_path,
            usecols=["INSURANCE_TYPE", "INCIDENT_STATE", "INCIDENT_CITY"],
        )
        insurance_type_options = (
            insurance_types_df["INSURANCE_TYPE"]
            .dropna()
            .astype("string")
            .sort_values()
            .unique()
            .tolist()
        )
        incident_state_options = [""] + insurance_types_df[
            "INCIDENT_STATE"
        ].dropna().astype("string").sort_values().unique().tolist()
        incident_city_options = [""] + insurance_types_df[
            "INCIDENT_CITY"
        ].dropna().astype("string").sort_values().unique().tolist()
    except Exception:
        insurance_type_options = ["Health", "Auto", "Property"]
        incident_state_options = [""]
        incident_city_options = [""]


st.set_page_config(
    page_title="Insurance Risk Studio",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://streamlit.io",
        "Report a bug": "https://github.com",
        "About": "Insurance Risk Scoring & Analytics Platform",
    },
)

theme = {
    "ink": "#0b1220",
    "subtle": "#5b6474",
    "card": "rgba(255, 255, 255, 0.95)",
    "surface": "#ffffff",
    "accent": "#6366f1",
    "accent_2": "#ec4899",
    "teal": "#14b8a6",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "shadow": "0 20px 50px rgba(15, 23, 42, 0.15)",
    "label": "#1f2937",
    "input_bg": "#ffffff",
    "input_text": "#0b1220",
    "input_placeholder": "#9ca3af",
    "pill_bg": "rgba(99, 102, 241, 0.15)",
    "pill_text": "#4338ca",
    "score_pill_bg": "rgba(20, 184, 166, 0.15)",
    "score_pill_text": "#0f766e",
    "border": "rgba(15, 23, 42, 0.1)",
    "hero_border": "rgba(15, 23, 42, 0.08)",
    "app_bg": (
        "radial-gradient(1400px 700px at 20% 0%, #e0e7ff 0%, rgba(224, 231, 255, 0) 50%),"
        "radial-gradient(1000px 600px at 100% 50%, #fce7f3 0%, rgba(252, 231, 243, 0) 50%),"
        "linear-gradient(180deg, #f8fafc 0%, #f1f5f9 50%, #ffffff 100%)"
    ),
}

theme_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:wght@400&family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
    --ink: {theme['ink']};
    --subtle: {theme['subtle']};
    --card: {theme['card']};
    --surface: {theme['surface']};
    --accent: {theme['accent']};
    --accent-2: {theme['accent_2']};
    --teal: {theme['teal']};
    --success: {theme['success']};
    --warning: {theme['warning']};
    --danger: {theme['danger']};
    --shadow: {theme['shadow']};
    --label: {theme['label']};
    --input-bg: {theme['input_bg']};
    --input-text: {theme['input_text']};
    --input-placeholder: {theme['input_placeholder']};
    --pill-bg: {theme['pill_bg']};
    --pill-text: {theme['pill_text']};
    --score-pill-bg: {theme['score_pill_bg']};
    --score-pill-text: {theme['score_pill_text']};
    --border: {theme['border']};
    --hero-border: {theme['hero_border']};
}}

.stApp {{
    background: {theme['app_bg']};
    color: var(--ink);
}}

* {{
    box-sizing: border-box;
}}
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Space Grotesk', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

/* Hero Section */
.hero {
    display: grid;
    grid-template-columns: 1.3fr 0.7fr;
    gap: 2rem;
    align-items: center;
    padding: 2.5rem 2rem;
    border-radius: 32px;
    background: linear-gradient(135deg, var(--card) 0%, rgba(255, 255, 255, 0.8) 100%);
    box-shadow: var(--shadow);
    border: 1px solid var(--hero-border);
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
    border-radius: 50%;
}

.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 48px;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}

.hero p {
    margin: 0.5rem 0 0 0;
    color: #64748b;
    font-size: 1.1rem;
    line-height: 1.6;
    font-weight: 400;
}

/* Pills and Badges */
.pill {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    background: linear-gradient(135deg, var(--pill-bg) 0%, rgba(99, 102, 241, 0.08) 100%);
    color: var(--pill-text);
    font-weight: 600;
    margin-bottom: 0.8rem;
    border: 1px solid rgba(99, 102, 241, 0.2);
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

.card {
    background: var(--card);
    border-radius: 28px;
    padding: 2rem;
    border: 1px solid var(--hero-border);
    box-shadow: var(--shadow);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.card:hover {
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.2);
    transform: translateY(-2px);
}

/* Form Elements */
.form-section {
    margin-bottom: 1.5rem;
}

.form-section-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Metric Grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1.5rem;
}

.metric-grid-4 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-top: 1.5rem;
}

.summary-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.8rem;
    align-items: center;
    margin-top: 1.2rem;
}

/* Badge Styling */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.83rem;
    font-weight: 600;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}

.badge:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.badge-approve {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.08) 100%);
    color: #047857;
    border-color: rgba(16, 185, 129, 0.3);
}

.badge-review {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(245, 158, 11, 0.08) 100%);
    color: #d97706;
    border-color: rgba(245, 158, 11, 0.3);
}

.badge-reject {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.08) 100%);
    color: #dc2626;
    border-color: rgba(239, 68, 68, 0.3);
}

.badge-neutral {
    background: linear-gradient(135deg, rgba(148, 163, 184, 0.15) 0%, rgba(148, 163, 184, 0.08) 100%);
    color: #475569;
    border-color: rgba(148, 163, 184, 0.3);
}

/* Metric Card */
.metric-card {
    background: linear-gradient(135deg, var(--surface) 0%, rgba(255, 255, 255, 0.6) 100%);
    border-radius: 20px;
    padding: 1.2rem;
    border: 1px solid var(--border);
    min-height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(99, 102, 241, 0.15);
    border-color: rgba(99, 102, 241, 0.2);
}

.metric-card h3 {
    margin: 0 0 0.5rem 0;
    font-size: 0.85rem;
    color: #64748b;
    font-weight: 600;
    line-height: 1.3;
    word-break: break-word;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

.metric-card p {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.2;
}

/* Score Pill */
.score-pill {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    background: linear-gradient(135deg, var(--score-pill-bg) 0%, rgba(20, 184, 166, 0.08) 100%);
    color: var(--score-pill-text);
    font-weight: 600;
    font-size: 0.88rem;
    margin-top: 0.8rem;
    border: 1px solid rgba(20, 184, 166, 0.2);
    letter-spacing: 0.3px;
}

/* Input Styling */
.stNumberInput label,
.stSelectbox label,
.stDateInput label,
.stTextInput label,
.stTextArea label,
.stRadio label,
.stMultiSelect label {
    color: var(--label);
    font-weight: 600;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

.stNumberInput input,
.stDateInput input,
.stTextInput input,
.stTextArea textarea {
    background: var(--input-bg);
    color: var(--input-text);
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 0.7rem 1rem;
    font-size: 0.95rem;
    transition: all 0.2s ease;
}

.stNumberInput input:focus,
.stDateInput input:focus,
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

.stNumberInput input::placeholder,
.stDateInput input::placeholder,
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: var(--input-placeholder);
}

div[data-baseweb="select"] > div {
    background: var(--input-bg);
    color: var(--input-text);
    border: 2px solid var(--border);
    border-radius: 12px;
    transition: all 0.2s ease;
}

div[data-baseweb="select"]:focus-within > div {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

div[data-baseweb="select"] svg {
    fill: var(--input-text);
}

/* Button Styling */
div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #5b5cf1 100%);
    color: #ffffff;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.3px;
    box-shadow: 0 12px 28px rgba(99, 102, 241, 0.3);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    text-transform: uppercase;
    position: relative;
    overflow: hidden;
}

div.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s ease;
}

div.stButton > button:hover::before {
    left: 100%;
}

div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 40px rgba(99, 102, 241, 0.4);
}

div.stButton > button:active,
div.stFormSubmitButton > button:active {
    transform: translateY(-1px);
}

/* Progress Bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
    border-radius: 10px;
    height: 8px;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

/* Expander */
details > summary {
    background: linear-gradient(135deg, var(--card) 0%, rgba(255, 255, 255, 0.8) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.5rem;
    font-weight: 600;
    color: var(--ink);
    cursor: pointer;
    transition: all 0.2s ease;
    user-select: none;
}

details > summary:hover {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(236, 72, 153, 0.05));
    border-color: rgba(99, 102, 241, 0.2);
}

details > summary::-webkit-details-marker {
    color: var(--accent);
}

/* Animations */
.fade-in {
    animation: fadeUp 0.6s ease both;
}

.fade-in-slow {
    animation: fadeUp 1s ease both;
}

@keyframes fadeUp {
    from { 
        opacity: 0; 
        transform: translateY(12px);
    }
    to { 
        opacity: 1; 
        transform: translateY(0);
    }
}

@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

.slide-in {
    animation: slideInLeft 0.5s ease both;
}

/* Responsive Design */
@media (max-width: 1200px) {
    .hero {
        grid-template-columns: 1fr;
        gap: 1.5rem;
    }

    .hero h1 {
        font-size: 40px;
    }

    .metric-grid-4 {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 900px) {
    .hero {
        padding: 1.8rem 1.5rem;
    }

    .hero h1 {
        font-size: 36px;
    }

    .card {
        padding: 1.5rem;
    }

    .metric-grid,
    .metric-grid-4 {
        grid-template-columns: repeat(2, 1fr);
        gap: 0.8rem;
    }
}

@media (max-width: 768px) {
    .hero {
        grid-template-columns: 1fr;
        padding: 1.5rem 1rem;
        border-radius: 24px;
    }

    .hero h1 {
        font-size: 32px;
    }

    .hero p {
        font-size: 1rem;
    }

    .card {
        padding: 1.25rem;
        border-radius: 24px;
    }

    .metric-grid,
    .metric-grid-4 {
        grid-template-columns: 1fr;
        gap: 1rem;
    }

    .metric-card {
        min-height: 110px;
        padding: 1rem;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        padding: 0.7rem 1.5rem;
        font-size: 0.9rem;
    }
}

@media (max-width: 640px) {
    .hero h1 {
        font-size: 28px;
    }

    .hero p {
        font-size: 0.95rem;
    }

    .pill {
        font-size: 0.8rem;
        padding: 0.3rem 0.8rem;
    }

    .card {
        padding: 1rem;
    }

    .metric-card p {
        font-size: 1.2rem;
    }

    .metric-card h3 {
        font-size: 0.75rem;
    }
}

/* Info/Warning/Error Boxes */
[data-testid="stAlert"] {
    border-radius: 16px;
    padding: 1rem 1.25rem;
    border: 1px solid transparent;
}

[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
    font-weight: 500;
}

/* Subheaders */
[data-testid="stHeading"] {
    font-weight: 700;
}

/* Caption */
[data-testid="stCaption"] {
    color: #64748b;
    font-size: 0.9rem;
}

/* Spinner */
.stSpinner > div > div {
    border-color: var(--accent);
}

.stSpinner > div > div:after {
    border-color: var(--accent);
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="hero fade-in">
    <div>
        <span class="pill">✨ Claim scoring dashboard</span>
        <h1>Insurance Risk Studio</h1>
        <p>⚡ Score claim risk, estimate severity, and surface the highest-impact cases with AI-powered insights.</p>
    </div>
    <div>
        <div class="score-pill">🔗 Fraud + Severity combined</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

left, right = st.columns([1.3, 1], gap="large")

with left:
    st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
    st.subheader("📋 Claim Input Form")
    st.caption(
        "🔐 Enter claim details below to generate comprehensive risk, fraud classification, and rejection scores."
    )

    with st.form("risk_form"):
        # Primary Information
        st.markdown(
            "<div class='form-section-title'>💰 Financial Details</div>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            claim_amount = st.number_input(
                "Claim Amount (₹)", min_value=1.0, value=1000.0, step=100.0
            )
        with col2:
            premium_amount = st.number_input(
                "Premium Amount (₹)", min_value=1.0, value=100.0, step=10.0
            )

        st.divider()

        # Incident Details
        st.markdown(
            "<div class='form-section-title'>🚨 Incident Details</div>",
            unsafe_allow_html=True,
        )
        col3, col4 = st.columns(2, gap="medium")
        with col3:
            insurance_type = st.selectbox("Insurance Type", insurance_type_options)
        with col4:
            incident_severity = st.selectbox(
                "Incident Severity", ["Minor Loss", "Major Loss", "Total Loss"]
            )

        col5, col6 = st.columns(2, gap="medium")
        with col5:
            police_report_available = st.selectbox(
                "Police Report Available",
                [0, 1],
                format_func=lambda v: "✅ Yes" if v == 1 else "❌ No",
            )
        with col6:
            any_injury = st.selectbox(
                "Any Injury",
                [0, 1],
                format_func=lambda v: "✅ Yes" if v == 1 else "❌ No",
            )

        st.divider()

        # Dates
        st.markdown(
            "<div class='form-section-title'>📅 Timeline</div>", unsafe_allow_html=True
        )
        col7, col8 = st.columns(2, gap="medium")
        with col7:
            loss_date = st.date_input(
                "Loss Date",
                value=date.today() - timedelta(days=5),
                max_value=date.today(),
            )
        with col8:
            report_date = st.date_input(
                "Report Date", value=date.today(), max_value=date.today()
            )

        # Optional Details
        with st.expander("🔍 Additional Details", expanded=False):
            st.markdown(
                "<div class='form-section-title'>👤 Personal Information</div>",
                unsafe_allow_html=True,
            )
            col9, col10 = st.columns(2, gap="medium")
            with col9:
                age = st.number_input("Age", min_value=18, max_value=100, value=40)
                marital_status = st.selectbox("Marital Status", ["Y", "N"])
                employment_status = st.selectbox("Employment Status", ["Y", "N"])
            with col10:
                risk_segmentation = st.selectbox("Risk Segmentation", ["L", "M", "H"])
                house_type = st.selectbox("House Type", ["Own", "Rent", "Mortgage"])
                incident_state = st.selectbox(
                    "Incident State",
                    incident_state_options,
                    format_func=lambda v: "Select" if v == "" else v,
                )
                incident_city = st.selectbox(
                    "Incident City",
                    incident_city_options,
                    format_func=lambda v: "Select" if v == "" else v,
                )

        st.write("")
        submitted = st.form_submit_button("🚀 Score Claim", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
    st.subheader("📊 Risk Summary & Analysis")
    st.write(
        "💡 Use the button to score this claim. Results persist until you score again."
    )

    if submitted:
        payload = {
            "CLAIM_AMOUNT": claim_amount,
            "PREMIUM_AMOUNT": premium_amount,
            "INSURANCE_TYPE": insurance_type,
            "INCIDENT_SEVERITY": incident_severity,
            "POLICE_REPORT_AVAILABLE": police_report_available,
            "ANY_INJURY": any_injury,
            "LOSS_DT": loss_date.strftime("%Y-%m-%d"),
            "REPORT_DT": report_date.strftime("%Y-%m-%d"),
            "AGE": age,
            "MARITAL_STATUS": marital_status,
            "EMPLOYMENT_STATUS": employment_status,
            "RISK_SEGMENTATION": risk_segmentation,
            "HOUSE_TYPE": house_type,
            "INCIDENT_STATE": incident_state or None,
            "INCIDENT_CITY": incident_city or None,
        }

        try:
            with st.spinner("Scoring claim..."):
                st.session_state["last_scoring"] = score_claim(
                    payload, model_dir=MODEL_DIR
                )
                st.session_state["last_error"] = None
        except PayloadValidationError as exc:
            st.session_state["last_scoring"] = None
            st.session_state["last_error"] = str(exc)

    error_msg = st.session_state.get("last_error")
    if error_msg:
        st.error(f"❌ Error: {error_msg}")

    scoring = st.session_state.get("last_scoring")
    if scoring:
        fraud_prob = float(scoring["fraud_probability"])
        severity = float(scoring["predicted_severity"])
        risk_score = float(scoring["risk_score"])
        rejection_score = float(scoring["rejection_score"])
        confidence_score = float(scoring.get("confidence_score", 75.0))
        transparency_score = float(scoring.get("transparency_score", 85.0))
        classification = str(scoring.get("classification", "Review"))
        band = str(scoring.get("risk_band", "Medium"))
        badge_class = str(scoring.get("badge_class", "badge-review"))
        threshold_used = dict(
            scoring.get(
                "threshold_used",
                {
                    "fraud_review_threshold": 0.40,
                    "fraud_reject_threshold": 0.70,
                    "risk_review_threshold": 10000,
                    "risk_reject_threshold": 25000,
                },
            )
        )
        decision_reason = str(
            scoring.get(
                "decision_reason",
                "Decision is based on fraud probability and risk score thresholds.",
            )
        )
        reasons = list(
            scoring.get(
                "reasons",
                [
                    "No additional explainability reasons available for this older result."
                ],
            )
        )
        warnings = list(scoring.get("warnings", []))

        # Determine classification icon
        icon_map = {"Approve": "✅", "Review": "⚠️", "Reject": "🚫"}
        icon = icon_map.get(classification, "ℹ️")

        st.markdown(
            f"""
<div class="metric-grid slide-in">
    <div class="metric-card">
        <h3>🎯 Fraud Probability</h3>
        <p>{fraud_prob:.2%}</p>
    </div>
    <div class="metric-card">
        <h3>💸 Estimated Payout</h3>
        <p>₹{severity:,.0f}</p>
    </div>
    <div class="metric-card">
        <h3>📊 Risk Score</h3>
        <p>₹{risk_score:,.0f}</p>
    </div>
</div>
<div class="metric-grid-4 slide-in" style="animation-delay: 0.1s;">
    <div class="metric-card">
        <h3>💵 Claim Amount</h3>
        <p>₹{claim_amount:,.0f}</p>
    </div>
    <div class="metric-card">
        <h3>{icon} Classification</h3>
        <p>{classification}</p>
    </div>
    <div class="metric-card">
        <h3>⚡ Fraud Risk %</h3>
        <p>{rejection_score:.1f}%</p>
    </div>
    <div class="metric-card">
        <h3>🎯 Confidence</h3>
        <p>{confidence_score:.0f}/100</p>
    </div>
    <div class="metric-card">
        <h3>🔍 Transparency</h3>
        <p>{transparency_score:.0f}/100</p>
    </div>
</div>
<div class="summary-row slide-in" style="animation-delay: 0.2s;">
    <div class="score-pill">🏆 Risk Band: {band}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("**Fraud Probability Trend**")
        st.progress(min(1.0, fraud_prob))

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.caption(
                "📌 Claim classification uses fraud probability and risk score thresholds to determine approval status."
            )
        with col_info2:
            st.caption(
                f"⚙️ Review: Fraud ≥ {threshold_used['fraud_review_threshold']:.0%} | Reject: Fraud ≥ {threshold_used['fraud_reject_threshold']:.0%}"
            )

        st.write("")
        st.info(f"💡 {decision_reason}")

        if warnings:
            st.warning(f"⚠️ {' | '.join(warnings)}")

        with st.expander(
            f"🔍 Detailed Explainability (Why {classification}?)",
            expanded=classification in {"Review", "Reject"},
        ):
            st.markdown("**Key Factors:**")
            for i, reason in enumerate(reasons, 1):
                st.markdown(f"- **{i}.** {reason}")
    else:
        st.info(
            "📌 No scoring output yet. Fill the form above and click 'Score Claim' to analyze the risk."
        )

    st.write("")
    col_reset_1, col_reset_2 = st.columns([1, 3])
    with col_reset_1:
        if st.button("🔄 Reset Results", use_container_width=True):
            st.session_state["last_scoring"] = None
            st.session_state["last_error"] = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# app.py
# Streamlit UI for CME Churn Prediction API
#
# Run with: streamlit run app.py
# Requires FastAPI running at http://127.0.0.1:8000
#
# Architecture:
# Streamlit (this file) = UI layer — takes input, displays output
# FastAPI (main.py)     = model layer — runs predictions
# These are separate processes — Streamlit calls FastAPI via HTTP

import streamlit as st
import requests
import json

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="CME Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    /* Base */
    .stApp {
        background-color: #0a0e1a;
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 1100px;}

    /* Header */
    .header-bar {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 100%);
        border: 1px solid #1e3a5f;
        border-radius: 4px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .header-bar::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4ff, #0080ff, #00d4ff);
    }
    .header-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #e8f4fd;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-sub {
        font-size: 0.85rem;
        color: #5a8ab0;
        margin-top: 0.4rem;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.05em;
    }

    /* Section labels */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        color: #00d4ff;
        text-transform: uppercase;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e3a5f;
    }

    /* Input card */
    .input-card {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 4px;
        padding: 1.5rem;
        height: 100%;
    }

    /* Result cards */
    .result-high {
        background: linear-gradient(135deg, #2d0a0a, #1a0505);
        border: 1px solid #8b1a1a;
        border-left: 4px solid #ff3b3b;
        border-radius: 4px;
        padding: 1.5rem 2rem;
        margin-bottom: 1rem;
    }
    .result-medium {
        background: linear-gradient(135deg, #2d1a0a, #1a0f05);
        border: 1px solid #8b4a1a;
        border-left: 4px solid #ff8c00;
        border-radius: 4px;
        padding: 1.5rem 2rem;
        margin-bottom: 1rem;
    }
    .result-low {
        background: linear-gradient(135deg, #0a2d1a, #051a0f);
        border: 1px solid #1a8b4a;
        border-left: 4px solid #00d97e;
        border-radius: 4px;
        padding: 1.5rem 2rem;
        margin-bottom: 1rem;
    }
    .prob-number {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 3.5rem;
        font-weight: 600;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .tier-badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.2em;
        padding: 0.3rem 0.8rem;
        border-radius: 2px;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .tier-high  { background: #8b1a1a; color: #ffaaaa; }
    .tier-med   { background: #8b4a1a; color: #ffd0aa; }
    .tier-low   { background: #1a8b4a; color: #aaffcc; }

    /* Reason items */
    .reason-item {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 3px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #a8c8e8;
        font-family: 'IBM Plex Sans', sans-serif;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Metric boxes */
    .metric-box {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 4px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #5a8ab0;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.2rem;
    }

    /* Status indicator */
    .api-status {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: #5a8ab0;
    }
    .dot-green { color: #00d97e; }
    .dot-red   { color: #ff3b3b; }

    /* Streamlit widget overrides */
    .stSlider > div > div > div {
        background: #1e3a5f !important;
    }
    label {
        color: #7ab0d0 !important;
        font-size: 0.8rem !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    .stSelectbox > div > div {
        background: #0d1b2a !important;
        border-color: #1e3a5f !important;
        color: #e8f4fd !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0050a0, #0070cc) !important;
        color: white !important;
        border: none !important;
        border-radius: 3px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0070cc, #0090ff) !important;
        transform: translateY(-1px) !important;
    }

    /* Number inputs */
    .stNumberInput > div > div > input {
        background: #0d1b2a !important;
        border-color: #1e3a5f !important;
        color: #e8f4fd !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"


# ── Helper Functions ───────────────────────────────────────

def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}


def get_prediction(payload: dict):
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )
        if r.status_code == 200:
            return True, r.json()
        else:
            return False, {"error": f"API returned {r.status_code}: {r.text}"}
    except requests.exceptions.ConnectionError:
        return False, {"error": "Cannot connect to API. Is FastAPI running on port 8000?"}
    except Exception as e:
        return False, {"error": str(e)}


def get_tier_color(tier: str) -> str:
    return {"HIGH": "#ff3b3b", "MEDIUM": "#ff8c00", "LOW": "#00d97e"}.get(tier, "#ffffff")


def get_result_class(tier: str) -> str:
    return {"HIGH": "result-high", "MEDIUM": "result-medium", "LOW": "result-low"}.get(tier, "result-low")


def get_tier_badge_class(tier: str) -> str:
    return {"HIGH": "tier-high", "MEDIUM": "tier-med", "LOW": "tier-low"}.get(tier, "tier-low")


# ── Header ─────────────────────────────────────────────────
healthy, health_data = check_api_health()

st.markdown(f"""
<div class="header-bar">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
            <p class="header-title">CME Churn Intelligence</p>
            <p class="header-sub">CLIENT RETENTION RISK ASSESSMENT SYSTEM · v1.0</p>
        </div>
        <div class="api-status">
            <span class="{'dot-green' if healthy else 'dot-red'}">●</span>
            {'API LIVE · ' + health_data.get('model_type', 'MODEL LOADED') if healthy else 'API OFFLINE'}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not healthy:
    st.error("⚠️ FastAPI is not running. Start it with: `uvicorn src.api.main:app --port 8000`")
    st.stop()


# ── Main Layout ────────────────────────────────────────────
col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown('<p class="section-label">▸ Client Profile Input</p>', unsafe_allow_html=True)

    with st.container():

        c1, c2 = st.columns(2)
        with c1:
            credit_score = st.number_input(
                "Credit Score", min_value=300, max_value=900,
                value=650, step=10
            )
            age = st.number_input(
                "Age", min_value=18, max_value=100, value=42
            )
            tenure = st.number_input(
                "Tenure (years)", min_value=0, max_value=10, value=3
            )
            balance = st.number_input(
                "Account Balance (£)", min_value=0.0,
                value=85000.0, step=1000.0, format="%.0f"
            )

        with c2:
            num_products = st.number_input(
                "Number of Products", min_value=1, max_value=4, value=1
            )
            estimated_salary = st.number_input(
                "Estimated Salary (£)", min_value=0.0,
                value=75000.0, step=1000.0, format="%.0f"
            )
            geography = st.selectbox(
                "Geography",
                options=["France", "Germany", "Spain"],
                index=0
            )
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
                index=0
            )

        c3, c4 = st.columns(2)
        with c3:
            has_cr_card = st.selectbox(
                "Has Credit Card",
                options=["Yes", "No"],
                index=0
            )
        with c4:
            is_active = st.selectbox(
                "Active Member",
                options=["Yes", "No"],
                index=1
            )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("▶  RUN RISK ASSESSMENT", type="primary")


with col_result:
    st.markdown('<p class="section-label">▸ Risk Assessment Output</p>', unsafe_allow_html=True)

    if predict_btn:
        # Build payload
        payload = {
            "CreditScore":       credit_score,
            "Age":               age,
            "Tenure":            tenure,
            "Balance":           float(balance),
            "NumOfProducts":     num_products,
            "HasCrCard":         1 if has_cr_card == "Yes" else 0,
            "IsActiveMember":    1 if is_active == "Yes" else 0,
            "EstimatedSalary":   float(estimated_salary),
            "Geography_France":  1 if geography == "France" else 0,
            "Geography_Germany": 1 if geography == "Germany" else 0,
            "Geography_Spain":   1 if geography == "Spain" else 0,
            "Gender":            1 if gender == "Male" else 0
        }

        with st.spinner("Analysing client risk profile..."):
            success, result = get_prediction(payload)

        if success:
            tier  = result['risk_tier']
            prob  = result['churn_probability']
            reasons = result['top_reasons']
            pct   = round(prob * 100, 1)
            color = get_tier_color(tier)
            rc    = get_result_class(tier)
            bc    = get_tier_badge_class(tier)

            # Main result card
            st.markdown(f"""
            <div class="{rc}">
                <p class="prob-number" style="color:{color}">{pct}%</p>
                <span class="tier-badge {bc}">
                    {tier} RISK
                </span>
                <p style="color:#8ab8d8; font-size:0.8rem; margin:0; font-family:'IBM Plex Mono',monospace;">
                    CHURN PROBABILITY SCORE
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Risk drivers
            st.markdown('<p class="section-label" style="margin-top:1.5rem;">▸ Primary Risk Drivers</p>', unsafe_allow_html=True)
            icons = ["◈", "◉", "◌"]
            for i, reason in enumerate(reasons):
                st.markdown(f"""
                <div class="reason-item">
                    <span style="color:{color}; font-family:'IBM Plex Mono',monospace;">{icons[i]}</span>
                    {reason}
                </div>
                """, unsafe_allow_html=True)

            # Metrics row
            st.markdown('<p class="section-label" style="margin-top:1.5rem;">▸ Profile Metrics</p>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">£{int(balance):,}</div>
                    <div class="metric-label">Balance</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{tenure}Y</div>
                    <div class="metric-label">Tenure</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{num_products}</div>
                    <div class="metric-label">Products</div>
                </div>""", unsafe_allow_html=True)

            # Action recommendation
            st.markdown("<br>", unsafe_allow_html=True)
            actions = {
                "HIGH":   "🔴 **Immediate action required.** Assign to senior relationship manager for urgent outreach within 24 hours.",
                "MEDIUM": "🟡 **Proactive monitoring.** Schedule a check-in call within the next 2 weeks. Consider product upgrade offer.",
                "LOW":    "🟢 **Standard engagement.** Include in next quarterly review cycle. No immediate action needed."
            }
            st.info(actions.get(tier, ""))

            # Raw payload expander
            with st.expander("View API request/response"):
                st.markdown("**Request payload:**")
                st.json(payload)
                st.markdown("**API response:**")
                st.json(result)

        else:
            st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")

    else:
        # Placeholder state
        st.markdown("""
        <div style="
            background: #0d1b2a;
            border: 1px dashed #1e3a5f;
            border-radius: 4px;
            padding: 3rem 2rem;
            text-align: center;
            color: #2a4a6a;
        ">
            <p style="font-family:'IBM Plex Mono',monospace; font-size:2rem; margin:0;">◎</p>
            <p style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; letter-spacing:0.15em; margin-top:1rem;">
                AWAITING INPUT
            </p>
            <p style="font-size:0.8rem; margin-top:0.5rem; color:#1e3a5f;">
                Configure client profile and run assessment
            </p>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="
    border-top: 1px solid #1e3a5f;
    padding-top: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
">
    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#2a4a6a; letter-spacing:0.1em;">
        CME CHURN INTELLIGENCE · RANDOMFOREST · MLFLOW TRACKED · KUBERNETES DEPLOYED
    </span>
    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#2a4a6a;">
        MODEL: cme_churn_model · AUC 0.8548
    </span>
</div>
""", unsafe_allow_html=True)
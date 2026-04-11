# src/api/main.py

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import shap
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────
app = FastAPI(
    title="CME Churn Prediction API",
    description="""
    Predicts churn probability for CME Group clients.
    Returns probability score, risk tier, and top driving factors.
    Used by relationship managers to prioritise proactive outreach.
    """,
    version="1.0.0"
)

# ── Global model state ─────────────────────────────────────
# Loaded once at startup, reused for every request.
# Loading from registry on every request would add 1-3s latency.
model = None
feature_columns = None


# ── Input Schema ───────────────────────────────────────────
class CustomerFeatures(BaseModel):
    """
    Input validation via Pydantic.

    Every field has type and range constraints.
    Malformed requests are rejected before reaching the model.
    In a regulated environment like CME, silent bad inputs
    producing bad outputs is unacceptable — fail fast, loudly.
    """
    CreditScore:       int   = Field(..., ge=300, le=900)
    Age:               int   = Field(..., ge=18,  le=100)
    Tenure:            int   = Field(..., ge=0,   le=10)
    Balance:           float = Field(..., ge=0)
    NumOfProducts:     int   = Field(..., ge=1,   le=4)
    HasCrCard:         int   = Field(..., ge=0,   le=1)
    IsActiveMember:    int   = Field(..., ge=0,   le=1)
    EstimatedSalary:   float = Field(..., ge=0)
    Geography_France:  int   = Field(..., ge=0,   le=1)
    Geography_Germany: int   = Field(..., ge=0,   le=1)
    Geography_Spain:   int   = Field(..., ge=0,   le=1)
    Gender:            int   = Field(..., ge=0,   le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "CreditScore": 650,
                "Age": 42,
                "Tenure": 3,
                "Balance": 85000.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 0,
                "EstimatedSalary": 75000.0,
                "Geography_France": 1,
                "Geography_Germany": 0,
                "Geography_Spain": 0,
                "Gender": 1
            }
        }


# ── Response Schema ────────────────────────────────────────
class ChurnPrediction(BaseModel):
    """
    Structured response — typed and documented.

    Returns SHAP-driven reasons not just a score because
    a relationship manager needs to know WHY a client is
    at risk — not just that they are. Actionable insight
    enables a relevant conversation, not a generic call.
    """
    churn_probability: float
    risk_tier:         str
    top_reasons:       list
    model_version:     str


# ── Feature Engineering ────────────────────────────────────
def engineer_features_for_input(data: dict) -> pd.DataFrame:
    """
    Apply identical feature engineering as preprocess.py.

    Critical: model was trained on engineered features.
    If raw features are sent at inference the model receives
    different inputs than it trained on — predictions will be wrong.
    Training and inference MUST apply identical transformations.
    This is one of the most common production ML bugs.
    """
    df = pd.DataFrame([data])

    df['Balance_to_Salary_ratio'] = (
        df['Balance'] / (df['EstimatedSalary'] + 1e-9)
    )
    df['Products_per_Year'] = (
        df['NumOfProducts'] / (df['Tenure'] + 1)
    )
    df['Is_Zero_Balance'] = (df['Balance'] == 0).astype(int)
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    df['Engagement_Score'] = (
        df['IsActiveMember'] + df['NumOfProducts']
    )

    return df


def get_risk_tier(probability: float) -> str:
    """
    Convert probability to business-readable risk tier.

    Thresholds calibrated to the F-beta analysis from training:
    HIGH   > 0.60 — immediate outreach required
    MEDIUM > 0.35 — monitor closely, soft outreach
    LOW   <= 0.35 — standard engagement cadence
    """
    if probability >= 0.6:
        return "HIGH"
    elif probability >= 0.35:
        return "MEDIUM"
    else:
        return "LOW"


def get_shap_reasons(model_obj, feature_df: pd.DataFrame) -> list:
    """
    Generate top 3 human-readable reasons using SHAP values.

    SHAP assigns each feature a contribution to this specific
    prediction. Positive = pushes towards churn.
    We surface the top 3 positive contributors as plain English
    reasons the relationship manager can use in conversation.
    """
    try:
        underlying = (
            model_obj.named_steps['model']
            if hasattr(model_obj, 'named_steps')
            else model_obj
        )

        explainer = shap.TreeExplainer(underlying)

        if hasattr(model_obj, 'named_steps'):
            features_scaled = model_obj.named_steps[
                'scaler'
            ].transform(feature_df)
            shap_values = explainer.shap_values(features_scaled)
        else:
            shap_values = explainer.shap_values(feature_df)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]

        feature_names = feature_df.columns.tolist()
        shap_dict = dict(zip(feature_names, shap_vals))

        sorted_features = sorted(
            shap_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        readable_map = {
            'IsActiveMember':          'Customer is not an active member',
            'Age':                     'Customer age is a risk factor',
            'NumOfProducts':           'Low number of products held',
            'Balance':                 'Account balance pattern is unusual',
            'Balance_to_Salary_ratio': 'Balance relative to salary is a risk signal',
            'Engagement_Score':        'Low overall engagement score',
            'Is_Zero_Balance':         'Account has zero balance — dormant signal',
            'Products_per_Year':       'Low product adoption rate over tenure',
            'CreditScore':             'Credit score is a contributing factor',
            'Tenure':                  'Customer tenure pattern',
            'Age_Group':               'Age group shows elevated churn risk',
            'Geography_Germany':       'Geography is a contributing factor',
            'Geography_France':        'Geography is a contributing factor',
            'Geography_Spain':         'Geography is a contributing factor',
            'Gender':                  'Demographic profile factor',
            'HasCrCard':               'Credit card holding status',
            'EstimatedSalary':         'Salary profile is a factor',
        }

        reasons = [
            readable_map.get(feat, feat)
            for feat, val in sorted_features
            if val > 0
        ]

        return reasons if reasons else ["Risk driven by combination of factors"]

    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return ["Risk score based on behavioural profile"]


# ── Startup ────────────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    global model, feature_columns

    try:
        import os
        import joblib

        # Determine base path based on environment
        # Docker container: files copied to /app
        # Local dev: files in project root
        if os.path.exists('/app/mlruns'):
            base_path = '/app'
        else:
            base_path = '.'

        # Load directly from artifact file
        # Bypasses MLflow registry path resolution which stores
        # absolute Mac paths in mlflow.db — those paths don't
        # exist inside the container.
        # Production equivalent: Azure ML registry resolves
        # this natively without any path issues.
        model_path = (
            f"{base_path}/mlruns/1/models/"
            f"m-91451c74fbf942d992381ec5b9eda578/"
            f"artifacts/model.pkl"
        )

        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully: {type(model).__name__}")

        feature_columns = [
            'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'HasCrCard', 'IsActiveMember',
            'EstimatedSalary', 'Geography_France', 'Geography_Germany',
            'Geography_Spain', 'Balance_to_Salary_ratio',
            'Products_per_Year', 'Is_Zero_Balance',
            'Age_Group', 'Engagement_Score'
        ]

    except Exception as e:
        logger.error(f"Failed to load model: {e}")


# ── Endpoints ──────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """
    Kubernetes liveness and readiness probe.

    Called every 10 seconds by Kubernetes.
    Returns 503 if model not loaded — pod gets no traffic
    until model is ready. Correct behaviour — never serve
    predictions from a pod with no model.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — service unavailable"
        )
    return {
        "status":       "healthy",
        "model_loaded": True,
        "service":      "cme-churn-prediction"
    }


@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer: CustomerFeatures):
    """
    Main prediction endpoint.

    Accepts customer features, returns:
    - churn_probability: float between 0 and 1
    - risk_tier: HIGH / MEDIUM / LOW
    - top_reasons: top 3 SHAP-driven risk factors
    - model_version: which model version produced this prediction

    Called by CRM systems, relationship manager dashboards,
    and automated monitoring pipelines.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        # Convert input to dict
        input_data = customer.model_dump()

        # Apply feature engineering — must match training
        feature_df = engineer_features_for_input(input_data)

        # Reorder columns to match exact training order
        feature_df = feature_df[feature_columns]

        # Predict
        churn_prob = float(
            model.predict_proba(feature_df)[0][1]
        )

        # Risk tier
        risk_tier = get_risk_tier(churn_prob)

        # SHAP reasons
        reasons = get_shap_reasons(model, feature_df)

        logger.info(
            f"Prediction: prob={churn_prob:.3f} "
            f"tier={risk_tier}"
        )

        return ChurnPrediction(
            churn_probability=round(churn_prob, 4),
            risk_tier=risk_tier,
            top_reasons=reasons,
            model_version="cme_churn_model_v1"
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/")
async def root():
    return {
        "service":   "CME Churn Prediction API",
        "version":   "1.0.0",
        "endpoints": {
            "health":  "GET  /health",
            "predict": "POST /predict",
            "docs":    "GET  /docs"
        }
    }
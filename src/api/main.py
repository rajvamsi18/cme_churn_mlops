# src/api/main.py

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
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

# ── App initialisation ─────────────────────────────────────
app = FastAPI(
    title="CME Churn Prediction API",
    description="""
    Predicts probability of customer churn for CME Group clients.
    Returns churn probability score, risk tier, and top driving factors.
    Used by relationship managers to prioritise proactive outreach.
    """,
    version="1.0.0"
)

# ── Global model state ─────────────────────────────────────
# Model is loaded once at startup and reused for every request.
# Loading from MLflow registry every request would be 100x slower.
model = None
feature_columns = None


# ── Input Schema ───────────────────────────────────────────
class CustomerFeatures(BaseModel):
    """
    Input validation schema using Pydantic.
    
    Every field has a type, range constraint, and description.
    If a caller sends CreditScore=99999 or Age=-5, FastAPI
    rejects the request immediately with a clear error message
    before it ever reaches the model.
    
    """
    CreditScore:      int   = Field(..., ge=300, le=900,
                                    description="Customer credit score")
    Age:              int   = Field(..., ge=18,  le=100,
                                    description="Customer age in years")
    Tenure:           int   = Field(..., ge=0,   le=10,
                                    description="Years as customer")
    Balance:          float = Field(..., ge=0,
                                    description="Account balance")
    NumOfProducts:    int   = Field(..., ge=1,   le=4,
                                    description="Number of products held")
    HasCrCard:        int   = Field(..., ge=0,   le=1,
                                    description="Has credit card: 1=yes 0=no")
    IsActiveMember:   int   = Field(..., ge=0,   le=1,
                                    description="Is active member: 1=yes 0=no")
    EstimatedSalary:  float = Field(..., ge=0,
                                    description="Estimated annual salary")
    Geography_France: int   = Field(..., ge=0,   le=1)
    Geography_Germany:int   = Field(..., ge=0,   le=1)
    Geography_Spain:  int   = Field(..., ge=0,   le=1)
    Gender:           int   = Field(..., ge=0,   le=1,
                                    description="Gender: 1=Male 0=Female")

    class Config:
        # Example payload shown in the API docs at /docs
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
    Structured response for every field is documented and typed.
    
    Returning SHAP reasons not just a score, because
    A relationship manager calling this API doesn't need 0.73.
    They need: "this client is high risk primarily because
    they have only 1 product and are not an active member."
    That's an actionable insight they can use in a conversation.
    """
    churn_probability: float
    risk_tier:         str    # HIGH / MEDIUM / LOW
    top_reasons:       list   # top 3 SHAP-driven factors
    model_version:     str


# ── Helper Functions ───────────────────────────────────────
def engineer_features_for_input(data: dict) -> pd.DataFrame:
    """
    Applied the same feature engineering from preprocess.py
    to a single incoming request.
    
    Critical point: the model was trained on engineered features.
    If you send raw features at inference time, the model gets
    different inputs than it was trained on and predictions
    will be wrong. Training and inference MUST apply identical
    transformations.
    """
    df = pd.DataFrame([data])
    
    # Same transformations as preprocess.py
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
    Converted probability score to business-readable risk tier.
    
    Thresholds match the F-beta analysis from training:
    > 0.6 = HIGH  — immediate outreach required
    > 0.35 = MEDIUM — monitor closely, soft outreach
    <= 0.35 = LOW  — standard engagement
    """
    if probability >= 0.6:
        return "HIGH"
    elif probability >= 0.35:
        return "MEDIUM"
    else:
        return "LOW"


def get_shap_reasons(model, feature_df: pd.DataFrame) -> list:
    """
    Generated top 3 human-readable reasons for the prediction.
    
    SHAP assigns each feature a contribution value to this
    specific prediction. Positive = pushes towards churn.
    Negative = pushes away from churn.
    
    I took the top 3 positive contributors and convert them
    to plain English reasons the relationship manager can use.
    """
    try:
        # Get the underlying model if wrapped in sklearn Pipeline
        underlying_model = (
            model.named_steps['model']
            if hasattr(model, 'named_steps')
            else model
        )
        
        explainer = shap.TreeExplainer(underlying_model)
        
        # Get features the model actually used
        if hasattr(model, 'named_steps'):
            # Pipeline — transform features first
            features_transformed = model.named_steps[
                'scaler'
            ].transform(feature_df)
            shap_values = explainer.shap_values(features_transformed)
        else:
            shap_values = explainer.shap_values(feature_df)
        
        # For binary classification, shap_values may be
        # a list [class0_values, class1_values]
        # We want class 1 (churn) values
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]
        
        # Map SHAP values back to feature names
        feature_names = feature_df.columns.tolist()
        shap_dict = dict(zip(feature_names, shap_vals))
        
        # Sort by absolute contribution, take top 3 positive
        sorted_features = sorted(
            shap_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Convert to human readable
        readable_map = {
            'IsActiveMember':         'Customer is not an active member',
            'Age':                    'Customer age is a risk factor',
            'NumOfProducts':          'Low number of products held',
            'Balance':                'Account balance pattern',
            'Balance_to_Salary_ratio':'Balance relative to salary is unusual',
            'Engagement_Score':       'Low overall engagement score',
            'Is_Zero_Balance':        'Account has zero balance (dormant)',
            'Products_per_Year':      'Low product adoption rate',
            'CreditScore':            'Credit score is a contributing factor',
            'Tenure':                 'Customer tenure is a risk factor',
            'Age_Group':              'Age group shows elevated churn risk',
            'Geography_Germany':      'Geography is a contributing factor',
            'Geography_France':       'Geography is a contributing factor',
            'Geography_Spain':        'Geography is a contributing factor',
            'Gender':                 'Demographic profile factor',
            'HasCrCard':              'Credit card holding status',
            'EstimatedSalary':        'Salary profile is a factor',
        }
        
        reasons = [
            readable_map.get(feat, feat)
            for feat, val in sorted_features
            if val > 0  # only positive contributors
        ]
        
        return reasons if reasons else ["Combination of risk factors"]
        
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return ["Risk score based on behavioural profile"]


# ── Startup Event ──────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    """
    Loaded model from MLflow registry when API starts.
    
    Loading a model takes 1-3 seconds. If you loaded it
    on every request, your API would have 1-3 second latency
    on every single call. Load once at startup, reuse forever.
    
    In Kubernetes, if this startup fails the pod fails its
    readiness probe and never receives traffic — which is
    exactly the right behaviour. Don't serve traffic with
    a broken model.
    """
    global model, feature_columns
    
    try:
        model_name = "cme_churn_model"
        model_uri  = f"models:/{model_name}/latest"
        
        logger.info(f"Loading model from MLflow registry: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully")
        
        # These must match exactly what preprocess.py produces
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
        # Don't raise here — let health check report unhealthy
        # rather than crashing the process entirely


# ── Endpoints ──────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """
    Kubernetes liveness and readiness probe endpoint.
    
    Kubernetes calls this every 10 seconds to check if the
    pod is alive. If this returns anything other than 200,
    Kubernetes restarts the pod or stops sending it traffic.
    
    I also checked if model is loaded, a pod with no model
    should not receive prediction requests.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — service unavailable"
        )
    return {
        "status":        "healthy",
        "model_loaded":  True,
        "service":       "cme-churn-prediction"
    }


@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer: CustomerFeatures):
    """
    Main prediction endpoint.
    
    Accepts customer features, returns churn probability,
    risk tier, and top 3 reasons driving the prediction.
    
    Called by: CRM systems, relationship manager dashboards,
    automated monitoring pipelines.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # Step 1 — Convert input to dict
        input_data = customer.model_dump()
        
        # Step 2 — Apply feature engineering
        # Must match training transformations exactly
        feature_df = engineer_features_for_input(input_data)
        
        # Step 3 — Reorder columns to match training order
        feature_df = feature_df[feature_columns]
        
        # Step 4 — Predict
        churn_prob = float(
            model.predict_proba(feature_df)[0][1]
        )
        
        # Step 5 — Risk tier
        risk_tier = get_risk_tier(churn_prob)
        
        # Step 6 — SHAP reasons
        reasons = get_shap_reasons(model, feature_df)
        
        logger.info(
            f"Prediction: prob={churn_prob:.3f} "
            f"tier={risk_tier} "
            f"reasons={reasons}"
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
        "service": "CME Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health":  "GET  /health",
            "predict": "POST /predict",
            "docs":    "GET  /docs"
        }
    }
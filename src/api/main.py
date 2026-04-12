# src/api/main.py

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import shap
import logging
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
# Loading from registry per request = 1-3s latency per call.
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

    Returns SHAP reasons not just a score because a relationship
    manager needs to know WHY a client is at risk, not just that
    they are. Actionable insight enables a relevant conversation.
    """
    churn_probability: float
    risk_tier:         str
    top_reasons:       list
    model_version:     str
    model_type:        str


# ── Feature Engineering ────────────────────────────────────
def engineer_features_for_input(data: dict) -> pd.DataFrame:
    """
    Apply identical transformations as preprocess.py.

    CRITICAL: model was trained on engineered features.
    If raw features are sent at inference the model gets different
    inputs than it trained on — predictions will be wrong.
    Training and inference MUST apply identical transformations.
    This is one of the most common production ML bugs.
    """
    df = pd.DataFrame([data])
    df['Balance_to_Salary_ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1e-9)
    df['Products_per_Year'] = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['Is_Zero_Balance'] = (df['Balance'] == 0).astype(int)
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    df['Engagement_Score'] = df['IsActiveMember'] + df['NumOfProducts']
    return df


def get_risk_tier(probability: float) -> str:
    """
    Convert probability to business-readable risk tier.

    Thresholds calibrated to F-beta analysis from training:
    HIGH   >= 0.60 — immediate outreach required
    MEDIUM >= 0.35 — monitor closely, soft outreach
    LOW    <  0.35 — standard engagement cadence
    """
    if probability >= 0.6:
        return "HIGH"
    elif probability >= 0.35:
        return "MEDIUM"
    else:
        return "LOW"


def get_shap_reasons(model_obj, feature_df: pd.DataFrame) -> list:
    """
    Generate top 3 human-readable SHAP-driven reasons.

    SHAP assigns each feature a contribution to this specific
    prediction. Positive = pushes towards churn.
    We surface top 3 positive contributors as plain English
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
            features_scaled = model_obj.named_steps['scaler'].transform(feature_df)
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
            shap_dict.items(), key=lambda x: x[1], reverse=True
        )[:3]

        readable_map = {
            'IsActiveMember':          'Customer is not an active member',
            'Age':                     'Customer age is a risk factor',
            'NumOfProducts':           'Low number of products held',
            'Balance':                 'Account balance pattern is unusual',
            'Balance_to_Salary_ratio': 'Balance to salary ratio is a risk signal',
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
    """
    Three-strategy model loading — tries each in order,
    stops at first success.

    ─────────────────────────────────────────────────────────
    STRATEGY 1 — MLflow Registry
    ─────────────────────────────────────────────────────────
    Asks MLflow: "give me the latest version of cme_churn_model"
    MLflow registry stores named versions — v1, v2, v3 etc.
    This enables rollback: load version 2 if version 3 degrades.

    Works when: Azure ML production (managed registry, HTTPS endpoint)
    Fails when: Docker on Mac or CI (SQLite db stores machine-specific
                absolute paths that don't exist inside container)

    ─────────────────────────────────────────────────────────
    STRATEGY 2 — best_model_info.json (primary Docker strategy)
    ─────────────────────────────────────────────────────────
    Training writes best_model_info.json with RELATIVE pkl path.
    Example: {"pkl_path": "mlruns/1/models/m-abc/artifacts/model.pkl"}

    At load time: base_path + relative_path = correct absolute path
    Docker: /app + mlruns/1/... = /app/mlruns/1/...
    Local:  .   + mlruns/1/... = ./mlruns/1/...

    Why relative not absolute?
    Absolute paths are machine-specific and break across environments.
    Relative paths resolve correctly anywhere by prepending known base.
    No database lookup, no path mismatch possible.

    Registry still exists for governance/rollback — this is just
    the reliable loading mechanism for containerised environments.

    Works when: any Docker environment (Mac local, CI, production)
    Fails when: best_model_info.json not present or pkl moved

    ─────────────────────────────────────────────────────────
    STRATEGY 3 — File Scan (safety net)
    ─────────────────────────────────────────────────────────
    Scans mlruns/ physically for all model.pkl files.
    Loads the first one it finds — no registry, no JSON needed.

    Works when: any environment where mlruns/ was copied in
    Limitation: doesn't know which model is "best" without metrics
                but better than returning 503 with no model at all

    ─────────────────────────────────────────────────────────
    In production with Azure ML:
    Only Strategy 1 needed — Azure manages paths centrally
    via HTTPS endpoints, not local filesystem paths.
    ─────────────────────────────────────────────────────────
    """
    global model, feature_columns

    # Determine environment — Docker or local
    if os.path.exists('/app/mlruns'):
        base_path = '/app'
        tracking_uri = "sqlite:////app/mlflow.db"
    else:
        base_path = '.'
        tracking_uri = "sqlite:///mlflow.db"

    logger.info(f"Environment base path: {base_path}")
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    loaded_model = None
    model_type_str = "unknown"

    # ── Strategy 1: MLflow Registry ───────────────────────
    try:
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = "models:/cme_churn_model/latest"
        logger.info(f"Strategy 1: trying MLflow registry ({model_uri})")
        loaded_model = mlflow.sklearn.load_model(model_uri)
        model_type_str = type(loaded_model).__name__
        logger.info(f"Strategy 1 SUCCESS: loaded {model_type_str}")

    except Exception as strategy1_error:
        logger.warning(f"Strategy 1 failed: {strategy1_error}")

        # ── Strategy 2: best_model_info.json ──────────────
        try:
            import joblib
            json_path = os.path.join(base_path, "best_model_info.json")
            logger.info(f"Strategy 2: trying JSON pointer ({json_path})")

            if not os.path.exists(json_path):
                raise FileNotFoundError(f"best_model_info.json not found at {json_path}")

            with open(json_path, "r") as f:
                model_info = json.load(f)

            # pkl_path in JSON is relative — prepend base_path
            relative_pkl = model_info['pkl_path']
            # Handle both ./mlruns/... and mlruns/... formats
            relative_pkl = relative_pkl.lstrip('./')
            absolute_pkl = os.path.join(base_path, relative_pkl)

            logger.info(f"Strategy 2: loading pkl from {absolute_pkl}")

            if not os.path.exists(absolute_pkl):
                raise FileNotFoundError(f"pkl not found at {absolute_pkl}")

            loaded_model = joblib.load(absolute_pkl)
            model_type_str = model_info.get('model_type', type(loaded_model).__name__)

            logger.info(
                f"Strategy 2 SUCCESS: loaded {model_type_str} "
                f"(registered v{model_info.get('registered_version', 'unknown')}, "
                f"AUC={model_info.get('auc_roc', 'unknown')})"
            )

        except Exception as strategy2_error:
            logger.warning(f"Strategy 2 failed: {strategy2_error}")

            # ── Strategy 3: File Scan ──────────────────────
            try:
                import joblib
                import glob

                logger.info("Strategy 3: scanning for model pkl files")
                pattern = os.path.join(base_path, "mlruns/**/model.pkl")
                all_pkls = glob.glob(pattern, recursive=True)

                if not all_pkls:
                    raise FileNotFoundError(
                        f"No model.pkl found under {base_path}/mlruns/"
                    )

                logger.info(f"Strategy 3: found {len(all_pkls)} pkl files")

                # Try to pick best by MLflow metrics if readable
                try:
                    mlflow.set_tracking_uri(tracking_uri)
                    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
                    experiments = client.search_experiments()

                    best_auc = -1
                    best_run_id = None

                    for exp in experiments:
                        runs = client.search_runs(
                            experiment_ids=[exp.experiment_id],
                            filter_string="params.gates_passed = 'True'",
                            order_by=["metrics.auc_roc DESC"],
                            max_results=1
                        )
                        if runs:
                            run_auc = runs[0].data.metrics.get('auc_roc', 0)
                            if run_auc > best_auc:
                                best_auc = run_auc
                                best_run_id = runs[0].info.run_id

                    if best_run_id:
                        # Find pkl whose path contains the run id
                        for pkl_path in all_pkls:
                            if best_run_id in pkl_path:
                                loaded_model = joblib.load(pkl_path)
                                model_type_str = type(loaded_model).__name__
                                logger.info(
                                    f"Strategy 3 SUCCESS (metric-ranked): "
                                    f"{model_type_str} AUC={best_auc:.4f}"
                                )
                                break

                except Exception as metric_error:
                    logger.warning(f"Strategy 3 metric ranking failed: {metric_error}")

                # Final fallback — load first available pkl
                if loaded_model is None:
                    loaded_model = joblib.load(all_pkls[0])
                    model_type_str = type(loaded_model).__name__
                    logger.warning(
                        f"Strategy 3 SUCCESS (first available): {model_type_str}"
                    )

            except Exception as strategy3_error:
                logger.error(
                    f"ALL strategies failed. Service will be unhealthy.\n"
                    f"Strategy 1: {strategy1_error}\n"
                    f"Strategy 2: {strategy2_error}\n"
                    f"Strategy 3: {strategy3_error}"
                )
                return

    # Assign to global
    model = loaded_model

    # Feature columns — must match preprocess.py exactly
    # If these don't match training, predictions will be wrong
    feature_columns = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary', 'Geography_France', 'Geography_Germany',
        'Geography_Spain', 'Balance_to_Salary_ratio',
        'Products_per_Year', 'Is_Zero_Balance',
        'Age_Group', 'Engagement_Score'
    ]

    logger.info(
        f"Service ready. Model: {model_type_str}. "
        f"Features: {len(feature_columns)}"
    )


# ── Endpoints ──────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """
    Kubernetes liveness and readiness probe.

    Called every 10 seconds by Kubernetes.
    Returns 503 if model not loaded — pod gets no traffic
    until model is ready. Correct behaviour — never serve
    predictions from a pod with no model loaded.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — service unavailable"
        )
    return {
        "status":       "healthy",
        "model_loaded": True,
        "model_type":   type(model).__name__,
        "service":      "cme-churn-prediction"
    }


@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer: CustomerFeatures):
    """
    Main prediction endpoint.

    Accepts customer features, returns:
    - churn_probability: float 0-1
    - risk_tier: HIGH / MEDIUM / LOW
    - top_reasons: top 3 SHAP-driven risk factors in plain English
    - model_version: which version produced this prediction
    - model_type: which algorithm is currently serving

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

        # Apply feature engineering — must match training exactly
        feature_df = engineer_features_for_input(input_data)

        # Reorder columns to match training order exactly
        feature_df = feature_df[feature_columns]

        # Predict
        churn_prob = float(model.predict_proba(feature_df)[0][1])

        # Risk tier
        risk_tier = get_risk_tier(churn_prob)

        # SHAP reasons
        reasons = get_shap_reasons(model, feature_df)

        logger.info(
            f"Prediction: prob={churn_prob:.3f} tier={risk_tier} "
            f"model={type(model).__name__}"
        )

        return ChurnPrediction(
            churn_probability=round(churn_prob, 4),
            risk_tier=risk_tier,
            top_reasons=reasons,
            model_version="cme_churn_model_v1",
            model_type=type(model).__name__
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
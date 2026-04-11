# src/models/train.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────
PROCESSED_PATH = "data/processed/churn_processed.csv"
MLFLOW_EXPERIMENT = "cme_churn_prediction"
TEST_SIZE = 0.2       # last 20% of rows = holdout test set
RANDOM_STATE = 42

# CI/CD threshold gates
# A model must beat ALL of these to be registered
AUC_THRESHOLD = 0.75
RECALL_THRESHOLD = 0.55


def load_processed_data(path: str):
    """
    Load preprocessed data and do a time-based split.
    
    Since Customer behaviour has temporal patterns, a random split
    mixes future behavioural patterns into training, inflating
    validation scores. Time-based split mirrors production reality
    — the model only ever learned from the past.
    """
    df = pd.read_csv(path)
    
    # Separate features and target
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    
    # Time-based split — last 20% as test set
    # This simulates: train on Jan-Aug, test on Sep-Oct
    split_idx = int(len(df) * (1 - TEST_SIZE))
    
    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Train churn rate: {y_train.mean():.3f}")
    logger.info(f"Test churn rate:  {y_test.mean():.3f}")
    
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred_proba, threshold=0.35):
    """
    Compute all evaluation metrics in one place.
    
    Threshold=0.35 not 0.5, because -
    Default 0.5 is wrong for imbalanced churn data.
    Missing a churner (false negative) costs more than
    a wasted retention call (false positive).
    0.35 weights towards recall
    
    Also need AUC-PR alongside AUC-ROC, because -
    AUC-ROC is optimistic on imbalanced datasets because
    it counts true negatives which dominate. AUC-PR only
    focuses on the positive class — it's the honest metric
    for churn where positives are the minority.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Recall at top decile — the business metric
    # Sort customers by predicted risk, then checking what % of
    # actual churners are in the top 10%
    n_top_decile = int(len(y_true) * 0.1)
    top_decile_idx = np.argsort(y_pred_proba)[::-1][:n_top_decile]
    recall_top_decile = y_true.iloc[top_decile_idx].sum() / y_true.sum()
    
    return {
        'auc_roc':           roc_auc_score(y_true, y_pred_proba),
        'auc_pr':            average_precision_score(y_true, y_pred_proba),
        'recall':            recall_score(y_true, y_pred),
        'precision':         precision_score(y_true, y_pred),
        'f1':                f1_score(y_true, y_pred),
        'recall_top_decile': recall_top_decile
    }


def check_threshold_gates(metrics: dict, model_name: str) -> bool:
    """
    Checking if model meets minimum quality gates.
    
    This is the same logic that will run in GitHub Actions CI/CD.
    If a model doesn't meet these gates it never gets registered
    and never reaches production. Period.
    
    These are like the deployment guardrails
    they exist so a degraded model can never silently
    replace a working one.
    """
    passed = True
    
    if metrics['auc_roc'] < AUC_THRESHOLD:
        logger.warning(
            f"{model_name} FAILED AUC gate: "
            f"{metrics['auc_roc']:.3f} < {AUC_THRESHOLD}"
        )
        passed = False
    
    if metrics['recall'] < RECALL_THRESHOLD:
        logger.warning(
            f"{model_name} FAILED Recall gate: "
            f"{metrics['recall']:.3f} < {RECALL_THRESHOLD}"
        )
        passed = False
    
    if passed:
        logger.info(f"{model_name} PASSED all threshold gates ✓")
    
    return passed


def train_logistic_regression(X_train, y_train):
    """
    Baseline model
    
    This forces good feature engineering because it can't learn
    non-linear interactions on its own. If LR performs well,
    features are strong. If it struggles, I know the
    signal requires non-linear capture — that tells me
    to moving towards tree models.
    
    The Pipeline wrapper bundles scaling + model together
    so the scaler is always applied consistently,
    in training, validation, AND production inference.
    So, forgetting to scale at inference is a classic bug.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # LR needs scaled features
        ('model', LogisticRegression(
            class_weight='balanced',   # handles 80/20 imbalance
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(X_train, y_train):
    """
    Intermediate model, captures non-linear patterns.
    
    No scaling needed as tree models are scale-invariant.
    They split on feature values, not distances.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1           
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    """
    Primary model, expected best performer.
    
    Why I choosed LightGBM over XGBoost or Random Forest?
    1. Histogram-based splitting, Normally 4x faster on large dataset size
    2. Handles categoricals natively (I have already encoded)
    3. scale_pos_weight handles class imbalance without distorting
       probability calibration the way SMOTE would
    
    scale_pos_weight = ratio of negatives to positives
    If 80% are non-churners and 20% are churners:
    scale_pos_weight = 80/20 = 4
    This tells LightGBM to weight churners 4x more heavily
    during training without resampling the data.
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    logger.info(f"LightGBM scale_pos_weight: {scale_pos_weight:.2f}")
    
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1          # suppress LightGBM's own output
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],  # track training loss
    )
    return model


def run_training():
    """
    Main training orchestrator.
    
    This function:
    1. Loads processed data
    2. Trains 3 models
    3. Logs everything to MLflow
    4. Checks threshold gates
    5. Registers the best passing model to MLflow Registry
    
    The MLflow Registry is the handoff point between
    training and deployment. The CI/CD pipeline picks
    the best registered model from here.
    """
    
    # Setting up MLflow — this is where all runs get logged
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info(f"MLflow experiment: {MLFLOW_EXPERIMENT}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(
        PROCESSED_PATH
    )
    
    # Models to train — name, training function, mlflow log function
    model_configs = [
        ("LogisticRegression", train_logistic_regression, 
         mlflow.sklearn.log_model),
        ("RandomForest",       train_random_forest,       
         mlflow.sklearn.log_model),
        ("LightGBM",           train_lightgbm,            
         mlflow.lightgbm.log_model),
    ]
    
    results = []  # collect results to find best model at end
    
    for model_name, train_fn, log_fn in model_configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}...")
        
        # Each model gets its own MLflow run
        # This means I can compare them side by side
        # in the MLflow UI
        with mlflow.start_run(run_name=model_name):
            
            # ── Train ──────────────────────────────────────
            model = train_fn(X_train, y_train)
            
            # ── Predict ────────────────────────────────────
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # ── Metrics ────────────────────────────────────
            metrics = compute_metrics(y_test, y_pred_proba)
            
            # ── Log to MLflow ──────────────────────────────
            # Parameters — what settings did I use
            mlflow.log_param("model_type",   model_name)
            mlflow.log_param("test_size",    TEST_SIZE)
            mlflow.log_param("threshold",    0.35)
            mlflow.log_param("random_state", RANDOM_STATE)
            
            # Metrics — how well did it perform?
            mlflow.log_metrics(metrics)
            
            # Model artifact — save the actual model object
            log_fn(model, artifact_path="model")
            
            # ── Print results ──────────────────────────────
            logger.info(f"  AUC-ROC:           {metrics['auc_roc']:.4f}")
            logger.info(f"  AUC-PR:            {metrics['auc_pr']:.4f}")
            logger.info(f"  Recall:            {metrics['recall']:.4f}")
            logger.info(f"  Precision:         {metrics['precision']:.4f}")
            logger.info(f"  F1:                {metrics['f1']:.4f}")
            logger.info(f"  Recall@TopDecile:  {metrics['recall_top_decile']:.4f}")
            
            # ── Threshold gates ────────────────────────────
            gates_passed = check_threshold_gates(metrics, model_name)
            mlflow.log_param("gates_passed", gates_passed)
            
            # Store run info for best model selection
            run_id = mlflow.active_run().info.run_id
            results.append({
                'model_name':   model_name,
                'model':        model,
                'metrics':      metrics,
                'gates_passed': gates_passed,
                'run_id':       run_id,
                'log_fn':       log_fn
            })
    
    # ── Select and Register Best Model ────────────────────────
    # Only consider models that passed ALL threshold gates
    passing_models = [r for r in results if r['gates_passed']]
    
    if not passing_models:
        logger.error(
            "NO models passed threshold gates. "
            "Deployment blocked. Check your features and data."
        )
        return
    
    # Best model = highest AUC-ROC among passing models
    best = max(passing_models, key=lambda r: r['metrics']['auc_roc'])
    logger.info(f"\nBest model: {best['model_name']} "
                f"(AUC-ROC: {best['metrics']['auc_roc']:.4f})")
    
    # Register to MLflow Model Registry
    # This is the handoff point to deployment
    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name="cme_churn_model"
    )
    logger.info(
        f"Registered model: cme_churn_model "
        f"version {registered.version}"
    )
    logger.info("Training complete. Model ready for deployment.")
    
    return best


if __name__ == "__main__":
    run_training()
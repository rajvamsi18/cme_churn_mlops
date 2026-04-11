# src/models/train.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import logging
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
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
TEST_SIZE = 0.2
RANDOM_STATE = 42

# CI/CD threshold gates — model must beat ALL of these
AUC_THRESHOLD = 0.75
RECALL_THRESHOLD = 0.55


def load_processed_data(path: str):
    """
    Load preprocessed data and perform time-based split.

    """
    df = pd.read_csv(path)

    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # Last 20% of rows = holdout test set
    # Simulates: train on Jan-Aug, evaluate on Sep-Oct
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
    Compute all evaluation metrics.

    Why threshold=0.35 not 0.5?
    Missing a churner costs more than a wasted retention call.
    0.35 weights recall over precision — same logic as F-beta
    threshold tuning in the SoulFoods churn model.

    Why AUC-PR alongside AUC-ROC?
    ROC is optimistic on imbalanced data — includes true negatives
    which dominate. PR curve focuses only on positive class — honest
    metric when churners are the minority.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Recall at top decile — the real business metric
    # Of the top 10% highest-risk customers, what % actually churn?
    n_top = int(len(y_true) * 0.1)
    top_idx = np.argsort(y_pred_proba)[::-1][:n_top]
    recall_top_decile = y_true.iloc[top_idx].sum() / y_true.sum()

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
    Check if model meets minimum quality gates before registration.

    This same logic runs in GitHub Actions CI/CD pipeline.
    If a model fails here it never gets registered and never
    reaches production. No human approval needed, automated gate.
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

    Forces good feature engineering because LR cannot learn
    non-linear interactions on its own. If LR performs well
    your features are strong. If it struggles you know tree
    models are needed, which confirms my algorithm choice.

    Pipeline bundles scaler + model so scaling is ALWAYS applied
    consistently at training, validation, AND inference.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(X_train, y_train):
    """
    Intermediate model, captures non-linear patterns.
    No scaling needed, tree models are scale-invariant.
    They split on feature values not distances.
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
    Primary model, expected best performer on tabular data.

    Why LightGBM over XGBoost or Random Forest?
    1. Histogram-based splitting, faster on this dataset size
    2. scale_pos_weight handles class imbalance without distorting
       probability calibration the way SMOTE would
    3. Consistent performance advantage on mixed feature types

    scale_pos_weight = negatives / positives
    Tells LightGBM to weight churners proportionally more heavily
    during training without resampling the data.
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    logger.info(f"LightGBM scale_pos_weight: {scale_pos_weight:.2f}")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model


def run_training():
    """
    Main training orchestrator.
    
    """
    # Local SQLite MLflow — reliable artifact storage
    # Docker container overrides this via MLFLOW_TRACKING_URI env var
    # pointing to sqlite:////app/mlflow.db (absolute path in container)
    tracking_uri = os.getenv(
        'MLFLOW_TRACKING_URI',
        'sqlite:///mlflow.db'
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"MLflow experiment: {MLFLOW_EXPERIMENT}")

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(
        PROCESSED_PATH
    )

    # Models to train
    model_configs = [
        ("LogisticRegression", train_logistic_regression,
         mlflow.sklearn.log_model),
        ("RandomForest",       train_random_forest,
         mlflow.sklearn.log_model),
        ("LightGBM",           train_lightgbm,
         mlflow.lightgbm.log_model),
    ]

    results = []

    for model_name, train_fn, log_fn in model_configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}...")

        with mlflow.start_run(run_name=model_name):

            # Train
            model = train_fn(X_train, y_train)

            # Predict
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = compute_metrics(y_test, y_pred_proba)

            # Log parameters
            mlflow.log_param("model_type",   model_name)
            mlflow.log_param("test_size",    TEST_SIZE)
            mlflow.log_param("threshold",    0.35)
            mlflow.log_param("random_state", RANDOM_STATE)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model artifact
            # artifact_path="model" is required for register_model
            # to resolve the URI correctly as runs:/{run_id}/model
            log_fn(model, artifact_path="model")

            # Log results
            logger.info(f"  AUC-ROC:          {metrics['auc_roc']:.4f}")
            logger.info(f"  AUC-PR:           {metrics['auc_pr']:.4f}")
            logger.info(f"  Recall:           {metrics['recall']:.4f}")
            logger.info(f"  Precision:        {metrics['precision']:.4f}")
            logger.info(f"  F1:               {metrics['f1']:.4f}")
            logger.info(f"  Recall@TopDecile: {metrics['recall_top_decile']:.4f}")

            # Check threshold gates
            gates_passed = check_threshold_gates(metrics, model_name)
            mlflow.log_param("gates_passed", gates_passed)

            run_id = mlflow.active_run().info.run_id
            results.append({
                'model_name':   model_name,
                'model':        model,
                'metrics':      metrics,
                'gates_passed': gates_passed,
                'run_id':       run_id,
                'log_fn':       log_fn
            })

    # ── Select Best Model ──────────────────────────────────
    passing_models = [r for r in results if r['gates_passed']]

    if not passing_models:
        logger.error(
            "NO models passed threshold gates. "
            "Deployment blocked. Check features and data."
        )
        return None

    best = max(
        passing_models,
        key=lambda r: r['metrics']['auc_roc']
    )
    logger.info(
        f"\nBest model: {best['model_name']} "
        f"(AUC-ROC: {best['metrics']['auc_roc']:.4f})"
    )

    # ── Register Best Model ────────────────────────────────
    # URI format: runs:/{run_id}/model
    # "model" matches the artifact_path used in log_fn above
    model_uri = f"runs:/{best['run_id']}/model"
    logger.info(f"Registering from URI: {model_uri}")

    registered = mlflow.register_model(
        model_uri=model_uri,
        name="cme_churn_model"
    )
    logger.info(
        f"Registered: cme_churn_model "
        f"version {registered.version}"
    )
    logger.info("Training complete. Model ready for deployment.")

    return best


if __name__ == "__main__":
    run_training()
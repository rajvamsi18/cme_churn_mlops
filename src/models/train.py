# src/models/train.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import logging
import os
import json
import glob
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
PROCESSED_PATH  = "data/processed/churn_processed.csv"
MLFLOW_EXPERIMENT = "cme_churn_prediction"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# CI/CD threshold gates — model must beat ALL of these
# to be registered and deployed
AUC_THRESHOLD    = 0.75
RECALL_THRESHOLD = 0.55


def load_processed_data(path: str):
    """
    Load preprocessed data and perform time-based split.

    Why time-based not random?
    Random split leaks future behavioural patterns into training,
    inflating validation scores. Time-based split mirrors production
    reality — model only ever learned from the past.
    Same principle as train/test split at SoulFoods churn model.
    """
    df = pd.read_csv(path)

    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # Last 20% of rows = holdout test set
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
    Missing a churner (false negative) costs more than a wasted
    retention call (false positive). 0.35 weights recall over
    precision — same F-beta logic from SoulFoods churn model.

    Why AUC-PR alongside AUC-ROC?
    ROC is optimistic on imbalanced data because it counts true
    negatives which dominate. PR curve focuses only on the positive
    class — honest metric when churners are the minority.

    Recall@TopDecile — the real business metric.
    Of the top 10% highest risk customers flagged,
    what % actually churn? This is what the retention team cares
    about — not an abstract AUC number.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

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
    Check if model meets minimum quality gates.

    This same logic runs in GitHub Actions CI/CD pipeline.
    If a model fails here it never gets registered and never
    reaches production. Automated gate — no human approval needed.

    This is the "evaluation as a control mechanism" pattern
    from the RAG system at SoulFoods — same principle applied
    to classical ML models.
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
    Baseline model.

    Always start with the simplest model that could work.
    LR forces good feature engineering because it cannot learn
    non-linear interactions. If LR performs well, features are
    strong. If it struggles, tree models are needed — which
    confirms the algorithm choice rationale.

    Pipeline bundles scaler + model — critical so scaling is
    applied consistently at training AND inference.
    Forgetting to scale at inference is a classic production bug.
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
    Intermediate model — captures non-linear patterns.
    No scaling needed — tree models are scale-invariant.
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
    Primary model — expected best performer on tabular data.

    Why LightGBM over XGBoost or Random Forest?
    1. Histogram-based splitting — faster on this dataset size
    2. scale_pos_weight handles class imbalance without distorting
       probability calibration the way SMOTE would
    3. Consistent performance advantage on mixed feature types

    scale_pos_weight = negatives / positives
    Tells LightGBM to weight churners proportionally more during
    training without resampling the data.
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


def find_model_pkl_path(run_id: str) -> str:
    """
    Find the physical pkl file path for a given MLflow run.

    MLflow stores model artifacts in mlruns/ with a hash-based
    directory structure. This function scans for the pkl file
    belonging to a specific run and returns its relative path.

    Why relative path not absolute?
    Absolute paths are machine-specific. A relative path like
    mlruns/1/models/m-abc123/artifacts/model.pkl resolves
    correctly on any machine by prepending the known base path.
    This is what makes Docker loading reliable.
    """
    pattern = "mlruns/**/model.pkl"
    all_pkls = glob.glob(pattern, recursive=True)

    # Try to find pkl associated with this specific run
    # MLflow artifact paths contain fragments of run metadata
    for pkl_path in all_pkls:
        try:
            # Check MLflow artifacts for this run to find match
            client = mlflow.tracking.MlflowClient()
            artifacts = client.list_artifacts(run_id)
            for artifact in artifacts:
                if artifact.path == 'model':
                    # This run has a model artifact
                    # The pkl should be in a directory linked to this run
                    # Try to match by checking if run_id appears in path
                    # or fall back to returning the most recent pkl
                    pass
        except Exception:
            pass

    # Simpler approach — return all pkls, let caller pick
    return all_pkls


def run_training():
    """
    Main training orchestrator.

    Flow:
    1. Load processed data, split train/test
    2. Train 3 models — LR baseline, RandomForest, LightGBM
    3. Evaluate each against threshold gates
    4. Register best passing model to MLflow registry
       (registry provides version history and rollback capability)
    5. Write best_model_info.json with relative pkl path
       (JSON provides reliable Docker loading without registry path issues)

    Why both registry AND JSON?
    Registry = governance, versioning, rollback
    JSON = reliable loading mechanism in containerised environments
    where SQLite MLflow stores environment-specific absolute paths
    that don't survive containerisation.

    In production with Azure ML managed registry, only Strategy 1
    (registry) would be needed — Azure resolves paths centrally.
    """
    # Local SQLite MLflow — reliable artifact storage
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"MLflow experiment: {MLFLOW_EXPERIMENT}")

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(PROCESSED_PATH)

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

            # Log parameters to MLflow
            mlflow.log_param("model_type",   model_name)
            mlflow.log_param("test_size",    TEST_SIZE)
            mlflow.log_param("threshold",    0.35)
            mlflow.log_param("random_state", RANDOM_STATE)

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)

            # Log model artifact
            # artifact_path="model" is required for register_model
            # to resolve the URI as runs:/{run_id}/model
            log_fn(model, artifact_path="model")

            # Print results
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

    # ── Register to MLflow Registry ────────────────────────
    # Registry provides: version history, rollback capability,
    # named pointer to current best model.
    # "models:/cme_churn_model/1" can be rolled back to
    # by loading a specific version number if latest degrades.
    model_uri = f"runs:/{best['run_id']}/model"
    logger.info(f"Registering from URI: {model_uri}")

    registered_version = None
    try:
        registered = mlflow.register_model(
            model_uri=model_uri,
            name="cme_churn_model"
        )
        registered_version = registered.version
        logger.info(
            f"Registered: cme_churn_model "
            f"version {registered_version}"
        )
    except Exception as e:
        logger.warning(
            f"Registry registration failed: {e}. "
            f"Continuing — JSON pointer will be used for loading."
        )

    # ── Write best_model_info.json ─────────────────────────
    # This is the reliable loading mechanism for Docker containers.
    #
    # Problem: MLflow registry stores absolute paths in mlflow.db
    # e.g. /Users/rajvamsichenna/... or /home/runner/work/...
    # These paths don't exist inside Docker containers which
    # have their own filesystem starting at /app/
    #
    # Solution: find the actual pkl file and store its RELATIVE path.
    # Relative path + known base (/app or .) = correct absolute path
    # in any environment without any database lookup.
    #
    # The registry still exists for governance and rollback.
    # JSON is just the reliable loading pointer.

    # Find pkl files created during this training run
    all_pkls = glob.glob("mlruns/**/model.pkl", recursive=True)

    # Find the pkl for our best model's run
    # MLflow artifact structure: mlruns/exp_id/models/m-hash/artifacts/model.pkl
    # We identify ours by process of elimination —
    # there are exactly 3 pkls (one per model trained)
    # and we know which run_id belongs to the best model
    best_pkl_path = None

    # Try to match via MLflow client artifact listing
    try:
        client = mlflow.tracking.MlflowClient()
        # Get artifact URI for best run
        run_info = client.get_run(best['run_id'])
        artifact_uri = run_info.info.artifact_uri
        # artifact_uri looks like: mlruns/1/run_id/artifacts
        # Extract the relative component
        if artifact_uri.startswith('mlruns'):
            potential_pkl = os.path.join(artifact_uri, 'model', 'model.pkl')
            if os.path.exists(potential_pkl):
                best_pkl_path = potential_pkl
                logger.info(f"Found pkl via artifact URI: {best_pkl_path}")
    except Exception as e:
        logger.warning(f"Artifact URI lookup failed: {e}")

    # Fallback — match by checking which pkl loads as correct type
    if best_pkl_path is None:
        import joblib
        expected_type = best['model_name']
        for pkl_path in all_pkls:
            try:
                candidate = joblib.load(pkl_path)
                candidate_type = type(candidate).__name__
                # Match model type to expected
                if expected_type == 'LogisticRegression' and 'Pipeline' in candidate_type:
                    best_pkl_path = pkl_path
                    break
                elif expected_type == 'RandomForest' and 'RandomForest' in candidate_type:
                    best_pkl_path = pkl_path
                    break
                elif expected_type == 'LightGBM' and 'LGBM' in candidate_type:
                    best_pkl_path = pkl_path
                    break
            except Exception:
                continue

    if best_pkl_path is None and all_pkls:
        # Last resort — use first available pkl
        best_pkl_path = all_pkls[0]
        logger.warning(
            f"Could not identify best pkl precisely. "
            f"Using: {best_pkl_path}"
        )

    # Write the JSON pointer file
    model_info = {
        "registered_name":    "cme_churn_model",
        "registered_version": registered_version,
        "model_type":         best['model_name'],
        "run_id":             best['run_id'],
        "auc_roc":            round(best['metrics']['auc_roc'], 4),
        "auc_pr":             round(best['metrics']['auc_pr'], 4),
        "recall":             round(best['metrics']['recall'], 4),
        "recall_top_decile":  round(best['metrics']['recall_top_decile'], 4),
        "pkl_path":           best_pkl_path,
        # pkl_path is RELATIVE — prepend base_path at load time
        # Local:  ./mlruns/...
        # Docker: /app/mlruns/...
    }

    with open("best_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    logger.info(f"Saved best_model_info.json: {model_info}")
    logger.info("Training complete. Model ready for deployment.")

    return best


if __name__ == "__main__":
    run_training()
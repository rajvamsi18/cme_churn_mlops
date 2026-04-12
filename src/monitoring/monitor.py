# src/monitoring/monitor.py
#
# Production Monitoring for CME Churn Prediction Model
#
# Two separate monitoring layers:
#
# LAYER 1 — Data Drift (Leading Indicator)
# ─────────────────────────────────────────
# Monitors whether input feature distributions are shifting
# from what the model was trained on.
# Uses Population Stability Index (PSI) on key features.
# Leading indicator — catches problems BEFORE predictions degrade.
# Key features selected by mean absolute SHAP value on this dataset:
# Age=0.106, Engagement_Score=0.067, Age_Group=0.055,
# NumOfProducts=0.050, IsActiveMember=0.027, Gender=0.026, Balance=0.022
#
# LAYER 2 — Prediction Quality (Lagged Indicator)
# ─────────────────────────────────────────────────
# Monitors whether model predictions are still accurate
# by comparing past predictions against actual outcomes.
# Lagged indicator — needs ground truth (confirmed churn) to arrive.
# Score customers today, check outcomes in 8 weeks.
# Baseline recall loaded from best_model_info.json — not hardcoded.

import pandas as pd
import numpy as np
import json
import logging
import os
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────
# PSI thresholds — industry standard for model drift detection
PSI_NO_CHANGE   = 0.1   # PSI < 0.1:    distribution stable
PSI_MODERATE    = 0.2   # PSI 0.1-0.2: monitor closely
PSI_SIGNIFICANT = 0.2   # PSI > 0.2:   significant shift, investigate

# Recall degradation threshold
# If recall at top decile drops more than 5 percentage points
# from baseline, trigger retraining review
RECALL_DEGRADATION_THRESHOLD = 0.05

# Key features to monitor for drift
# Selected by mean absolute SHAP value computed on this dataset
# (top 7 of 17 features by SHAP importance)
# Monitoring all 17 features would create noise — focus on
# the ones that actually drive predictions
KEY_FEATURES = [
    'Age',               # SHAP=0.106 — strongest predictor
    'Engagement_Score',  # SHAP=0.067 — composite engagement signal
    'Age_Group',         # SHAP=0.055 — age bin captures non-linearity
    'NumOfProducts',     # SHAP=0.050 — product adoption signal
    'IsActiveMember',    # SHAP=0.027 — activity status
    'Gender',            # SHAP=0.026 — demographic factor
    'Balance',           # SHAP=0.022 — financial signal
]


# ── Load Baseline Recall from Training Output ──────────────

def load_baseline_recall(json_path: str = "best_model_info.json") -> float:
    """
    Load baseline recall@top_decile from best_model_info.json.

    Why not hardcode the baseline?
    The baseline is the recall achieved by whichever model won
    training. If we retrain with different parameters and a better
    model wins, the baseline changes. Hardcoding would give wrong
    degradation alerts after any retrain.

    best_model_info.json is written by train.py after every
    training run — it always reflects the current production model.
    Reading from it means monitoring stays in sync with training
    automatically without any manual updates.
    """
    try:
        with open(json_path, 'r') as f:
            info = json.load(f)
        baseline = info.get('recall_top_decile')
        if baseline is None:
            raise KeyError("recall_top_decile not found in JSON")
        logger.info(
            f"Loaded baseline recall@top_decile from {json_path}: "
            f"{baseline:.4f} "
            f"(model: {info.get('model_type', 'unknown')}, "
            f"AUC: {info.get('auc_roc', 'unknown')})"
        )
        return float(baseline)
    except FileNotFoundError:
        logger.warning(
            f"{json_path} not found. Using fallback baseline 0.387. "
            f"Run train.py first to generate this file."
        )
        return 0.387
    except Exception as e:
        logger.warning(f"Could not read baseline from JSON: {e}. Using 0.387.")
        return 0.387


# ── Generate Synthetic Production Data ────────────────────

def generate_synthetic_production_data(
    training_data_path: str,
    n_samples: int = 5000,
    drift_severity: str = "severe",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic production data for monitoring simulation.

    Why synthetic and not the test split?
    Our dataset was split 80/20 — 8000 train, 2000 test.
    Using that same 2000-row test set as "production data"
    would mean monitoring is evaluated on data the model
    already saw during evaluation — not a realistic simulation.

    Instead we generate fresh synthetic data by:
    1. Sampling with replacement from the full dataset
       to get realistic feature distributions
    2. Applying controlled drift to specific features
       to simulate what happens in production over time

    drift_severity options:
    "none"     — clean data, all PSI should be < 0.1 (sanity check)
    "moderate" — mild drift, some features cross 0.1 threshold
    "severe"   — significant drift, features cross 0.2 threshold
    """
    np.random.seed(random_seed)

    # Load full processed dataset as our population
    df = pd.read_csv(training_data_path)

    # Sample with replacement to get n_samples "new" customers
    # This gives us realistic feature correlations and distributions
    production_df = df.sample(
        n=n_samples,
        replace=True,
        random_state=random_seed
    ).reset_index(drop=True)

    # Add small random noise to continuous features
    # to make it feel like genuinely new data
    continuous_features = ['Balance', 'CreditScore', 'EstimatedSalary', 'Age']
    for feat in continuous_features:
        if feat in production_df.columns:
            noise_scale = production_df[feat].std() * 0.05
            production_df[feat] = (
                production_df[feat] +
                np.random.normal(0, noise_scale, n_samples)
            ).clip(production_df[feat].min(), production_df[feat].max())

    # Apply drift based on severity
    if drift_severity == "moderate":
        # Simulate gradual demographic shift — older customer base
        production_df['Age'] = (
            production_df['Age'] +
            np.random.normal(3, 1.5, n_samples)
        ).clip(18, 100)

        # Recalculate Age_Group after shifting Age
        production_df['Age_Group'] = pd.cut(
            production_df['Age'],
            bins=[0, 30, 40, 50, 60, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        # Simulate balance increase — economic growth period
        production_df['Balance'] = (
            production_df['Balance'] * 1.15 +
            np.random.normal(2000, 500, n_samples)
        ).clip(0, None)

        # Recalculate derived features that depend on Balance
        production_df['Balance_to_Salary_ratio'] = (
            production_df['Balance'] /
            (production_df['EstimatedSalary'] + 1e-9)
        )
        production_df['Is_Zero_Balance'] = (
            production_df['Balance'] == 0
        ).astype(int)

        logger.info(
            "Applied moderate drift: Age +3 years avg, "
            "Balance +15%"
        )

    elif drift_severity == "severe":
        # Severe drift — significant distributional shift
        production_df['Age'] = (
            production_df['Age'] +
            np.random.normal(8, 3, n_samples)
        ).clip(18, 100)

        production_df['Age_Group'] = pd.cut(
            production_df['Age'],
            bins=[0, 30, 40, 50, 60, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        production_df['Balance'] = (
            production_df['Balance'] * 1.5 +
            np.random.normal(10000, 2000, n_samples)
        ).clip(0, None)

        production_df['Balance_to_Salary_ratio'] = (
            production_df['Balance'] /
            (production_df['EstimatedSalary'] + 1e-9)
        )
        production_df['Is_Zero_Balance'] = (
            production_df['Balance'] == 0
        ).astype(int)

        logger.info(
            "Applied severe drift: Age +8 years avg, "
            "Balance +50%"
        )

    else:
        logger.info("No drift applied — clean production data")

    logger.info(
        f"Generated {n_samples} synthetic production records "
        f"(drift_severity={drift_severity})"
    )

    return production_df


# ── Layer 1: Data Drift Monitoring ────────────────────────

def calculate_psi(expected: np.ndarray,
                  actual: np.ndarray,
                  buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).

    PSI measures how much a feature's distribution has shifted
    between training (expected) and production (actual).

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    Why PSI over simpler statistics like mean shift?
    PSI captures full distributional shape changes.
    A feature could have the same mean but completely different
    tails — PSI catches this, mean comparison doesn't.
    Same metric used in the SoulFoods churn model monitoring.
    """
    breakpoints = np.nanpercentile(
        expected,
        np.linspace(0, 100, buckets + 1)
    )
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        logger.warning("Not enough unique breakpoints for PSI")
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    # Add epsilon to avoid log(0)
    expected_pct = (expected_counts + 1e-6) / (len(expected) + 1e-6)
    actual_pct   = (actual_counts   + 1e-6) / (len(actual)   + 1e-6)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return round(float(psi), 4)


def run_drift_monitoring(training_df: pd.DataFrame,
                         production_df: pd.DataFrame) -> dict:
    """
    Run PSI drift monitoring on key features.

    Compares current production feature distributions against
    the training distribution baseline.

    In production:
    - Training distribution saved as baseline artifact during training
    - Production distribution computed from last week's API requests
    - Runs weekly via scheduled job
    - Alerts sent to Azure Monitor if PSI > 0.2
    """
    logger.info("=== Layer 1: Data Drift Monitoring ===")

    results = {}
    alerts  = []

    for feature in KEY_FEATURES:
        if feature not in training_df.columns:
            logger.warning(f"Feature {feature} not in training data")
            continue
        if feature not in production_df.columns:
            logger.warning(f"Feature {feature} not in production data")
            continue

        psi = calculate_psi(
            training_df[feature].dropna().values,
            production_df[feature].dropna().values
        )

        if psi < PSI_NO_CHANGE:
            status = "STABLE"
        elif psi < PSI_MODERATE:
            status = "MONITOR"
        else:
            status = "ALERT"
            alerts.append(feature)

        results[feature] = {'psi': psi, 'status': status}
        logger.info(f"  {feature:30s} PSI={psi:.4f}  [{status}]")

    if alerts:
        logger.warning(
            f"DRIFT ALERT: {len(alerts)} features show significant shift: "
            f"{alerts}. Recommend retraining review."
        )
    else:
        logger.info("All features stable. No significant drift detected.")

    results['_summary'] = {
        'timestamp':          datetime.now().isoformat(),
        'features_monitored': len(results) - 1,
        'alerts':             alerts,
        'alert_count':        len(alerts),
        'recommendation':     'retrain' if len(alerts) >= 2 else 'monitor'
    }

    return results


# ── Layer 2: Prediction Quality Monitoring ────────────────

def calculate_recall_at_top_decile(y_true: np.ndarray,
                                   y_pred_proba: np.ndarray) -> float:
    """
    Calculate recall at top decile.

    Of the top 10% of customers ranked by predicted churn risk,
    what % actually churned?

    Same metric used during training evaluation — consistency
    between training and monitoring metrics is critical.
    You cannot monitor degradation if you measure different things.
    """
    n_top = max(1, int(len(y_true) * 0.1))
    top_idx = np.argsort(y_pred_proba)[::-1][:n_top]
    recall = y_true[top_idx].sum() / max(y_true.sum(), 1)
    return round(float(recall), 4)


def run_prediction_quality_monitoring(
    production_df: pd.DataFrame,
    model,
    feature_columns: list,
    baseline_recall: float
) -> dict:
    """
    Monitor prediction quality on synthetic production data.

    In production:
    - Week 1:  model scores customers, scores logged to database
    - Week 9:  churn outcomes confirmed for those customers
    - Weekly job: compare week 1 scores vs week 9 outcomes
    - Alert if recall@top_decile drops > 5pp from baseline

    Why 8-week lag?
    Churn is defined as disengagement within 8 weeks of scoring.
    You need to wait 8 weeks for ground truth to arrive.
    This is the same churn window used during training.

    For simulation: we have labels in our synthetic data
    (sampled from real dataset which has Exited column)
    so we can compute quality metrics immediately.
    """
    logger.info("=== Layer 2: Prediction Quality Monitoring ===")

    # Get features and labels
    if 'Exited' not in production_df.columns:
        logger.warning("No Exited column in production data — skipping quality check")
        return {'status': 'SKIPPED', 'reason': 'No outcome labels available'}

    X_prod = production_df[feature_columns]
    y_true = production_df['Exited'].values

    # Generate predictions
    try:
        scores = model.predict_proba(X_prod)[:, 1]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {'status': 'ERROR', 'reason': str(e)}

    # Calculate metrics
    current_recall = calculate_recall_at_top_decile(y_true, scores)
    degradation    = baseline_recall - current_recall

    from sklearn.metrics import roc_auc_score
    try:
        current_auc = round(roc_auc_score(y_true, scores), 4)
    except Exception:
        current_auc = None

    # Alert logic
    if degradation > RECALL_DEGRADATION_THRESHOLD:
        status = "ALERT"
        recommendation = "trigger_retraining_review"
        logger.warning(
            f"QUALITY ALERT: Recall@TopDecile dropped from "
            f"{baseline_recall:.4f} to {current_recall:.4f} "
            f"({degradation:.4f} degradation > threshold {RECALL_DEGRADATION_THRESHOLD}). "
            f"Retraining review recommended."
        )
    else:
        status = "STABLE"
        recommendation = "continue_monitoring"
        logger.info(
            f"Quality stable: Recall@TopDecile={current_recall:.4f} "
            f"(baseline={baseline_recall:.4f}, "
            f"degradation={degradation:.4f} < threshold {RECALL_DEGRADATION_THRESHOLD})"
        )

    return {
        'timestamp':        datetime.now().isoformat(),
        'n_evaluated':      len(production_df),
        'baseline_recall':  baseline_recall,
        'current_recall':   current_recall,
        'degradation':      round(degradation, 4),
        'auc_roc':          current_auc,
        'status':           status,
        'recommendation':   recommendation,
        'threshold_used':   RECALL_DEGRADATION_THRESHOLD
    }


# ── Monitoring Report ──────────────────────────────────────

def generate_monitoring_report(drift_results: dict,
                                quality_results: dict,
                                output_path: str = "monitoring_report.json"):
    """
    Combine drift and quality results into a single report.

    In production:
    - Metrics sent to Azure Monitor as custom metrics
    - Report stored in monitoring database
    - PagerDuty alert fired if overall_status = ALERT
    - Grafana dashboard updated with rolling metrics
    """
    report = {
        'report_timestamp': datetime.now().isoformat(),
        'model_name':       'cme_churn_model',
        'layer_1_drift':    drift_results,
        'layer_2_quality':  quality_results,
        'overall_status':   'STABLE',
        'action_required':  False
    }

    drift_alerts   = drift_results.get('_summary', {}).get('alert_count', 0)
    quality_status = quality_results.get('status', 'STABLE')

    if drift_alerts >= 2 or quality_status == 'ALERT':
        report['overall_status']     = 'ALERT'
        report['action_required']    = True
        report['recommended_action'] = 'trigger_retraining_review'
    elif drift_alerts == 1:
        report['overall_status']     = 'MONITOR'
        report['recommended_action'] = 'increase_monitoring_frequency'
    else:
        report['recommended_action'] = 'continue_normal_monitoring'

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Monitoring report saved: {output_path}")
    logger.info(f"Overall status:  {report['overall_status']}")
    logger.info(f"Action required: {report['action_required']}")

    return report


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":

    PROCESSED_PATH   = "data/processed/churn_processed.csv"
    MODEL_INFO_PATH  = "best_model_info.json"

    if not os.path.exists(PROCESSED_PATH):
        logger.error(
            f"Processed data not found at {PROCESSED_PATH}. "
            f"Run preprocess.py first."
        )
        exit(1)

    # ── Load baseline recall from training output ──────────
    # Not hardcoded — read from best_model_info.json which
    # train.py writes after every training run.
    # This means monitoring stays in sync with the current
    # production model automatically.
    baseline_recall = load_baseline_recall(MODEL_INFO_PATH)

    # ── Load training data as baseline distribution ────────
    # Use 80% training split — same as model training
    # so PSI measures drift from what the model actually learned
    full_df   = pd.read_csv(PROCESSED_PATH)
    split_idx = int(len(full_df) * 0.8)
    training_df = full_df.iloc[:split_idx].copy()

    logger.info(
        f"Baseline distribution: {len(training_df)} training records"
    )

    # ── Generate synthetic production data ─────────────────
    # 5000 fresh synthetic records — NOT the test split
    # Why not test split?
    # Test split was used for model evaluation — using it for
    # monitoring simulation would be evaluating on already-seen data.
    # Synthetic data gives us genuinely new records with controlled
    # drift applied to test that PSI detection works correctly.
    production_df = generate_synthetic_production_data(
        training_data_path=PROCESSED_PATH,
        n_samples=5000,
        drift_severity="moderate"  # change to "severe" or "none" to test
    )

    # ── Load model ─────────────────────────────────────────
    import joblib, glob

    try:
        with open(MODEL_INFO_PATH, 'r') as f:
            model_info = json.load(f)
        pkl_path = model_info['pkl_path']
        model = joblib.load(pkl_path)
        logger.info(f"Loaded model: {type(model).__name__} from {pkl_path}")
    except Exception as e:
        logger.warning(f"Could not load via JSON: {e}. Scanning mlruns/...")
        pkls = glob.glob("mlruns/**/model.pkl", recursive=True)
        if not pkls:
            logger.error("No model found. Run train.py first.")
            exit(1)
        model = joblib.load(pkls[0])
        logger.info(f"Loaded model via scan: {type(model).__name__}")

    # Feature columns must match training exactly
    feature_columns = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary', 'Geography_France', 'Geography_Germany',
        'Geography_Spain', 'Balance_to_Salary_ratio',
        'Products_per_Year', 'Is_Zero_Balance',
        'Age_Group', 'Engagement_Score'
    ]

    # ── Layer 1: Drift Monitoring ──────────────────────────
    drift_results = run_drift_monitoring(training_df, production_df)

    # ── Layer 2: Quality Monitoring ────────────────────────
    quality_results = run_prediction_quality_monitoring(
        production_df=production_df,
        model=model,
        feature_columns=feature_columns,
        baseline_recall=baseline_recall
    )

    # ── Generate Report ────────────────────────────────────
    report = generate_monitoring_report(
        drift_results,
        quality_results,
        output_path="monitoring_report.json"
    )

    # ── Print Summary ──────────────────────────────────────
    print("\n" + "="*60)
    print("MONITORING SUMMARY")
    print("="*60)
    print(f"Baseline Recall@TopDecile: {baseline_recall:.4f}")
    print(f"Current  Recall@TopDecile: {quality_results.get('current_recall', 'N/A')}")
    print(f"Drift alerts:              {drift_results.get('_summary', {}).get('alert_count', 0)}")
    print(f"Overall status:            {report['overall_status']}")
    print(f"Action required:           {report['action_required']}")
    print(f"Recommendation:            {report['recommended_action']}")
    print("="*60)
    print("\nFull report: monitoring_report.json")
    print("In production: sent to Azure Monitor, triggers PagerDuty if ALERT")
# CME Churn Intelligence — End-to-End MLOps Pipeline

A production-grade MLOps system for customer churn prediction, built to demonstrate the full ML engineering lifecycle from raw data to deployed Kubernetes service. Designed to reflect the technical expectations of Senior AI/ML Engineer roles in financial services.

---

## What This Project Demonstrates

This project goes beyond model training. It covers every stage a production ML system needs:

- **Data pipeline** — ingestion, validation, preprocessing, and feature engineering
- **Model training** — multi-model comparison with automated threshold gates
- **Experiment tracking** — MLflow with hosted DagsHub tracking server
- **Model serving** — FastAPI REST API with Pydantic input validation and SHAP-driven explainability
- **Containerisation** — Docker image built for linux/amd64
- **CI/CD automation** — GitHub Actions pipeline with smoke testing
- **Kubernetes deployment** — Minikube locally, AKS-ready manifests
- **Production monitoring** — PSI drift detection and rolling recall quality checks
- **UI** — Streamlit risk assessment interface for relationship managers

---

## Live Experiment Tracking

All training runs — parameters, metrics, and model artifacts — logged to a hosted MLflow tracking server on DagsHub. Publicly accessible without login.

🔗 **[View Live Experiments on DagsHub](https://dagshub.com/rajvamsi18/cme_churn_mlops.mlflow)**

You can see all three model runs (LogisticRegression, RandomForest, LightGBM) with their full metric comparisons across AUC-ROC, AUC-PR, Recall, Precision, F1, and Recall@TopDecile.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Actions CI/CD                     │
│  Push → Test → Train → Gate Check → Docker Build → Deploy   │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │     MLflow Registry     │  ← DagsHub hosted
          │   cme_churn_model v2    │    (publicly viewable)
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │     FastAPI Service     │  ← Docker container
          │   /health  /predict     │    AKS / Minikube
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │     Streamlit UI        │  ← Relationship manager
          │  Risk Assessment Tool   │    interface
          └─────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML Models | LightGBM, RandomForest, Logistic Regression |
| Explainability | SHAP (TreeExplainer) |
| Experiment Tracking | MLflow + DagsHub (hosted tracking server) |
| API Framework | FastAPI + Pydantic |
| Containerisation | Docker (linux/amd64) |
| Orchestration | Kubernetes (Minikube / AKS-ready) |
| CI/CD | GitHub Actions |
| Monitoring | PSI drift detection + rolling recall |
| UI | Streamlit |
| Registry | GitHub Container Registry (ghcr.io) |

---

## Project Structure

```
cme_churn_mlops/
├── src/
│   ├── data/
│   │   └── ingest.py              # Data loading and schema validation
│   ├── features/
│   │   └── preprocess.py          # Feature engineering pipeline
│   ├── models/
│   │   └── train.py               # Multi-model training with MLflow
│   ├── api/
│   │   └── main.py                # FastAPI prediction service
│   └── monitoring/
│       └── monitor.py             # PSI drift + quality monitoring
├── k8s/
│   ├── deployment.yaml            # Kubernetes deployment (2 replicas)
│   ├── service.yaml               # LoadBalancer service
│   └── hpa.yaml                   # Horizontal pod autoscaler
├── .github/
│   └── workflows/
│       └── ci_cd.yml              # 4-stage CI/CD pipeline
├── app.py                         # Streamlit UI
├── Dockerfile                     # Container definition
├── best_model_info.json           # Training output pointer
└── requirements.txt
```

---

## CI/CD Pipeline

Every push to `main` triggers a 4-stage automated pipeline:

```
Stage 1 — Run Unit Tests (53s)
Stage 2 — Train Models + Threshold Gates (1m 11s)
           ↳ Trains LogisticRegression, RandomForest, LightGBM
           ↳ Gates: AUC-ROC > 0.75 AND Recall > 0.55
           ↳ Registers best model to MLflow registry
           ↳ Pipeline FAILS here if no model passes gates
Stage 3 — Build Docker Image (2m 15s)
           ↳ Builds linux/amd64 image
           ↳ Pushes to GitHub Container Registry
           ↳ Runs smoke test — health check must return 200
Stage 4 — Deploy to Production (4s)
           ↳ kubectl rolling update (zero downtime)
```

---

## Model Performance

Three models trained and evaluated. Best model selected automatically by AUC-ROC after threshold gate validation.

| Model | AUC-ROC | AUC-PR | Recall | Recall@TopDecile |
|---|---|---|---|---|
| LightGBM | 0.8499 | 0.6646 | 0.7795 | 0.4000 |
| **RandomForest** ✓ | **0.8548** | **0.6522** | **0.8282** | **0.3872** |
| LogisticRegression | 0.7469 | 0.4263 | 0.8231 | 0.2615 |

**RandomForest registered as production model** — highest AUC-ROC, passed all threshold gates.

### Key SHAP Feature Importances

Features monitored for drift, ranked by mean absolute SHAP value computed on this dataset:

```
Age                  0.106  ← strongest predictor
Engagement_Score     0.067
Age_Group            0.055
NumOfProducts        0.050
IsActiveMember       0.027
Gender               0.026
Balance              0.022
```

---

## Production Monitoring

Two-layer monitoring — catches problems before and after they affect predictions.

**Layer 1 — Data Drift (Leading Indicator)**
PSI calculated on top 7 SHAP features. Catches distributional shifts before prediction quality degrades.
- PSI < 0.10 → Stable
- PSI 0.10–0.20 → Monitor
- PSI > 0.20 → Alert, retraining review

**Layer 2 — Prediction Quality (Lagged Indicator)**
Rolling Recall@TopDecile compared against training baseline (0.3872). Needs ground truth to arrive — score today, check outcomes in 8 weeks.
If degradation > 5 percentage points → retraining triggered.

Sample monitoring output with moderate synthetic drift applied:
```
Age           PSI=0.0828  [STABLE]
Age_Group     PSI=0.1422  [MONITOR]
Balance       PSI=0.2145  [ALERT]
Recall@TopDecile: 0.3935  (baseline: 0.3872)  [STABLE]
Overall: MONITOR — increase monitoring frequency
```

---

## Running Locally

**Prerequisites:** Python 3.11, Docker Desktop, Git

```bash
# Clone and setup
git clone https://github.com/rajvamsi18/cme_churn_mlops.git
cd cme_churn_mlops
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download dataset (requires Kaggle API key)
kaggle datasets download -d shantanudhakadd/bank-customer-churn-prediction \
  -p data/raw --unzip

# Run pipeline
python src/features/preprocess.py
python src/models/train.py

# Start API
uvicorn src.api.main:app --port 8000

# Start UI (new terminal)
streamlit run app.py

# Run monitoring
python src/monitoring/monitor.py
```

**Docker:**
```bash
docker build --platform linux/amd64 -t cme-churn-api:v1 .
docker run --platform linux/amd64 -p 8000:8000 cme-churn-api:v1
```

**Kubernetes (Minikube):**
```bash
minikube start --driver=docker --cpus=2 --memory=3500
eval $(minikube docker-env)
docker build -t cme-churn-api:v1 .
eval $(minikube docker-env --unset)
kubectl apply -f k8s/ --validate=false
kubectl get pods
minikube service cme-churn-api-service --url
```

---

## API Reference

**Health check:**
```bash
GET /health
→ {"status": "healthy", "model_type": "RandomForestClassifier"}
```

**Prediction — HIGH risk example:**
```bash
POST /predict
Content-Type: application/json

{
  "CreditScore": 400, "Age": 45, "Tenure": 2,
  "Balance": 120000.0, "NumOfProducts": 1,
  "HasCrCard": 1, "IsActiveMember": 0,
  "EstimatedSalary": 80000.0,
  "Geography_France": 0, "Geography_Germany": 1,
  "Geography_Spain": 0, "Gender": 0
}

→ {
    "churn_probability": 0.72,
    "risk_tier": "HIGH",
    "top_reasons": [
      "Customer is not an active member",
      "Age group shows elevated churn risk",
      "Low number of products held"
    ],
    "model_version": "cme_churn_model_v1",
    "model_type": "RandomForestClassifier"
  }
```

---

## Design Decisions Worth Noting

**Why RandomForest over LightGBM?**
RandomForest had higher AUC-ROC (0.8548 vs 0.8499). LightGBM had higher AUC-PR and Recall@TopDecile. In a real production deployment the business metric — overall ranking quality vs top-decile targeting precision — would drive the final choice. Both passed threshold gates.

**Why two-strategy model loading?**
SQLite MLflow stores absolute filesystem paths in its database. These paths break when the container runs on a different machine. Strategy 1 tries the MLflow registry (works with managed registries like Azure ML). Strategy 2 uses a relative path pointer written by training (works everywhere). In production, Azure ML managed registry eliminates this entirely.

**Why PSI over simpler drift metrics?**
PSI captures full distributional shape changes, not just mean shifts. A feature can have the same mean but completely different tails — PSI catches this, mean comparison doesn't. Industry standard in financial services model risk management.

**Why Pydantic for input validation?**
In a regulated financial environment, silent bad inputs producing bad outputs is unacceptable. Every field has type and range constraints — malformed requests are rejected with clear error messages before reaching the model. Fail fast, fail loudly.

**Why DagsHub for experiment tracking?**
DagsHub provides a free hosted MLflow tracking server with public experiment visibility. This means training runs are logged to a persistent remote server rather than a local SQLite file, enabling team collaboration and providing a shareable audit trail of all model experiments. In a CME production environment this would be Azure ML's managed MLflow registry.

---

## What's Next

- [ ] Deploy to Railway/Render for public API URL
- [ ] Add unit tests for feature engineering functions
- [ ] Add model signature to MLflow logging
- [ ] Connect to Azure ML managed registry for enterprise-grade path resolution
- [ ] Implement A/B testing framework for model comparison in production

---

## Acknowledgements

This project was built with assistance from **Claude (Anthropic)** as an AI pair programmer. Claude helped with code generation, debugging, and documentation throughout the development process.

All technical decisions,  architecture decisions, project direction, domain framing, and validation were driven by the author. Claude was used as a tool, the same way a senior engineer might use Stack Overflow, documentation, or a knowledgeable colleague, to accelerate development and work through problems faster.

The use of AI assistance is disclosed here in the interest of transparency.
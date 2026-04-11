# Dockerfile

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Copy MLflow database and model artifacts
# mlflow.db is in root, this is where MLflow logged all experiments
# mlruns/ contains the actual model files saved during training
COPY mlflow.db .
COPY mlruns/ ./mlruns/

# Environment variables
# Tells MLflow exactly where to find its database inside the container
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
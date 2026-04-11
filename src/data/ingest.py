# src/data/ingest.py

import pandas as pd
import os
import logging
from pathlib import Path

# Set up logging — every production script logs what it's doing
# so you can debug failures without re-running everything
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from CSV.
    
    In production this function would instead connect to a database,
    an API, or a cloud storage bucket — but the interface stays identical.
    The rest of the pipeline never changes regardless of source.
    """
    logger.info(f"Loading raw data from {filepath}")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found at {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Check that the expected columns are present.
    
    This is the first data quality gate. In a real pipeline,
    if schema validation fails, the entire pipeline halts and 
    fires an alert — you never process corrupted data silently.
    """
    expected_columns = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary', 'Exited'  # Exited = our churn label
    ]
    
    missing = [col for col in expected_columns if col not in df.columns]
    
    if missing:
        logger.error(f"Schema validation failed. Missing columns: {missing}")
        return False
    
    logger.info("Schema validation passed")
    return True


def basic_data_report(df: pd.DataFrame) -> dict:
    """
    Generate a quick data quality report.
 
    This runs before any preprocessing so I have a baseline.
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'churn_rate': df['Exited'].mean().round(4),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    logger.info(f"Churn rate in dataset: {report['churn_rate']*100:.1f}%")
    logger.info(f"Missing values: {sum(report['missing_values'].values())}")
    
    return report


if __name__ == "__main__":
    
    RAW_PATH = "data/raw/Churn_Modelling.csv"
    
    df = load_raw_data(RAW_PATH)
    
    if not validate_schema(df):
        raise ValueError("Schema validation failed — check your data file")
    
    report = basic_data_report(df)
    
    print("\n--- Data Quality Report ---")
    print(f"Total rows: {report['total_rows']}")
    print(f"Churn rate: {report['churn_rate']*100:.1f}%")
    print(f"\nMissing values per column:")
    for col, count in report['missing_values'].items():
        if count > 0:
            print(f"  {col}: {count}")
    print(f"\nColumn types:")
    for col, dtype in report['dtypes'].items():
        print(f"  {col}: {dtype}")
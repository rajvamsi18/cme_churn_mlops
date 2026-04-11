# src/features/preprocess.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dropping columns that are identifiers, not features.
    
    RowNumber, CustomerId, Surname carry zero predictive signal.
    Including them would cause the model to memorise IDs rather
    than learn genuine behavioural patterns — a form of data leakage.
    """
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped identifier columns: {cols_to_drop}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical string columns into numbers.
    
    one-hot for Geography but label encode for Gender:
    Geography has 3 values (France, Spain, Germany), one-hot creates
    3 binary columns with no implied order,as
    there is no natural ordering between countries.
    
    Gender has 2 values (Male/Female), a single binary column is sufficient. 

    """
    # One-hot encode Geography — creates Geography_France, 
    # Geography_Germany, Geography_Spain
    # drop_first=True drops one column to avoid multicollinearity
    df = pd.get_dummies(df, columns=['Geography'], drop_first=False)
    logger.info("One-hot encoded Geography")
    
    # Binary encode Gender: Female=0, Male=1
    df['Gender'] = (df['Gender'] == 'Male').astype(int)
    logger.info("Binary encoded Gender")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creating new features that capture behavioural signals
    not present in the raw columns.
    
    Raw features tells what a customer looks like. 
    Engineered features tells how
    they behave and where they are in their lifecycle.
    """
    
    # 1. Balance to Salary ratio
    # A customer with £50k balance and £150k salary is very different
    # from one with £50k balance and £40k salary.
    # Raw balance alone loses this context.
    # Adding small epsilon to avoid division by zero
    df['Balance_to_Salary_ratio'] = (
        df['Balance'] / (df['EstimatedSalary'] + 1e-9)
    )
    logger.info("Created Balance_to_Salary_ratio")
    
    # 2. Products per year of tenure
    # Captures how quickly a customer has adopted products.
    # A customer with 3 products in 2 years is more engaged
    # than one with 3 products in 8 years.
    # Adding 1 to tenure to avoid division by zero for new customers
    df['Products_per_Year'] = (
        df['NumOfProducts'] / (df['Tenure'] + 1)
    )
    logger.info("Created Products_per_Year")
    
    # 3. Zero balance flag
    # Customers with exactly zero balance are often dormant.
    # Dormant accounts have very different churn patterns.
    # A binary flag captures this cliff edge better than
    # the continuous balance value alone.
    df['Is_Zero_Balance'] = (df['Balance'] == 0).astype(int)
    logger.info("Created Is_Zero_Balance flag")
    
    # 4. Age group bins
    # Age has a non-linear relationship with churn.
    # Binning captures this shape better than raw age.
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=[0, 1, 2, 3, 4]  # 0=young, 4=elderly
    ).astype(int)
    logger.info("Created Age_Group bins")
    
    # 5. Composite engagement score
    # IsActiveMember and NumOfProducts both signal engagement.
    # Combining them into one score captures overall engagement
    # level in a single feature the model can weight easily.
    df['Engagement_Score'] = (
        df['IsActiveMember'] + df['NumOfProducts']
    )
    logger.info("Created Engagement_Score")
    
    return df


def split_features_target(df: pd.DataFrame):
    """
    Separate features (X) from target label (y).
    
    'Exited' is our churn label — 1 means churned, 0 means stayed.
    Everything else becomes input features.
    """
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def run_preprocessing_pipeline(input_path: str, output_path: str):
    """
    Full preprocessing pipeline — runs all steps in order.
    
    This is the function the training script will call.
    Every step is logged so I can trace exactly what happened
    to the data when I debug issues later.
    """
    logger.info("=== Starting preprocessing pipeline ===")
    
    # Load
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Step 1 — Drop identifiers
    df = drop_identifier_columns(df)
    
    # Step 2 — Encode categoricals
    df = encode_categoricals(df)
    
    # Step 3 — Engineer features
    df = engineer_features(df)
    
    # Step 4 — Save processed data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
    
    # Step 5 — Quick summary
    X, y = split_features_target(df)
    logger.info(f"Final feature count: {X.shape[1]}")
    logger.info(f"Churn rate preserved: {y.mean():.3f}")
    logger.info("=== Preprocessing pipeline complete ===")
    
    return df


if __name__ == "__main__":
    run_preprocessing_pipeline(
        input_path="data/raw/Churn_Modelling.csv",
        output_path="data/processed/churn_processed.csv"
    )
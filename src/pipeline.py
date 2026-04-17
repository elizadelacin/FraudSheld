import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import run_preprocessing
from src.features import run_feature_engineering
from src.train import run_training


def run_pipeline(
    transaction_path: str = "data/raw/transaction.csv",
    identity_path: str = "data/raw/identity.csv",
    clean_path: str = "data/processed/clean_data.parquet",
    features_path: str = "data/processed/features.parquet",
    model_path: str = "models/xgb_fraud.joblib"
):
    print("=" * 40)
    print("STEP 1: Preprocessing")
    print("=" * 40)
    run_preprocessing(transaction_path, identity_path, clean_path)

    print("\n" + "=" * 40)
    print("STEP 2: Feature Engineering")
    print("=" * 40)
    run_feature_engineering(clean_path, features_path)

    print("\n" + "=" * 40)
    print("STEP 3: Training")
    print("=" * 40)
    run_training(features_path, model_path)

    print("\nPipeline complete!")


if __name__ == "__main__":
    run_pipeline()
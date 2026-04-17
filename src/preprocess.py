import pandas as pd
import numpy as np


def load_data(transaction_path: str, identity_path: str) -> pd.DataFrame:
    transaction = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)

    print("Transaction shape:", transaction.shape)
    print("Identity shape:", identity.shape)

    df = pd.merge(transaction, identity, on='TransactionID', how='left')
    print("Merged shape:", df.shape)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    nan_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    cols_to_drop = nan_percent[nan_percent > 80].index
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} columns with >80% NaN")

    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    categorical_cols = df.select_dtypes(include="object").columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    print("Remaining NaN values:", df.isna().sum().sum())
    print("Clean data shape:", df.shape)

    return df


def run_preprocessing(
    transaction_path: str = "data/raw/transaction.csv",
    identity_path: str = "data/raw/identity.csv",
    output_path: str = "data/processed/clean_data.parquet"
) -> pd.DataFrame:

    df = load_data(transaction_path, identity_path)
    df = clean_data(df)

    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    run_preprocessing()
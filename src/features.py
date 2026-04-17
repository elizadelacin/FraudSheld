import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['day'] = (df['TransactionDT'] // 86400) % 7
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if 6 <= x <= 9 else 0)
    print("Time features created!")
    return df


def create_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    df['amt_log'] = np.log1p(df['TransactionAmt'])

    anonymous_emails = ['protonmail.com', 'mail.com', 'outlook.es', 'aim.com']
    df['is_anonymous_email'] = df['P_emaildomain'].apply(
        lambda x: 1 if x in anonymous_emails else 0)

    df['is_mobile'] = df['DeviceType'].apply(
        lambda x: 1 if x == 'mobile' else 0)

    df['is_discover'] = df['card4'].apply(
        lambda x: 1 if x == 'discover' else 0)

    df['is_high_risk_product'] = df['ProductCD'].apply(
        lambda x: 1 if x == 'C' else 0)

    print("Binary features created!")
    return df


def create_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    # Card-level transaction statistics
    card_agg = df.groupby("card1")["TransactionAmt"].agg(["mean", "std", "count"])
    df['card1_amt_mean'] = df['card1'].map(card_agg['mean'])
    df['card1_amt_std'] = df['card1'].map(card_agg['std'])
    df['card1_count'] = df['card1'].map(card_agg['count'])

    # Email domain-level transaction statistics
    email_agg = df.groupby("P_emaildomain")["TransactionAmt"].agg(["mean", "std", "count"])
    df['email_amt_mean'] = df['P_emaildomain'].map(email_agg['mean'])
    df['email_amt_std'] = df['P_emaildomain'].map(email_agg['std'])
    df['email_count'] = df['P_emaildomain'].map(email_agg['count'])

    # Cards with a single transaction have no std — fill with 0
    df['card1_amt_std'] = df['card1_amt_std'].fillna(0)
    df['email_amt_std'] = df['email_amt_std'].fillna(0)

    print("Aggregation features created!")
    return df


def encode_categorical(df: pd.DataFrame):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Encoding {len(categorical_columns)} categorical columns...")

    label_encoder = LabelEncoder()
    for column in categorical_columns:
        df[column + '_encoded'] = label_encoder.fit_transform(df[column].astype(str))

    print("Encoding done!")
    return df, categorical_columns


def apply_pca(df: pd.DataFrame):
    v_columns = [col for col in df.columns if col.startswith("V")]
    print(f"Number of V columns: {len(v_columns)}")

    df[v_columns] = df[v_columns].astype(np.float32)

    scaler = StandardScaler()
    v_scaled = scaler.fit_transform(df[v_columns])

    pca_check = PCA()
    pca_check.fit(v_scaled)
    cumulative_explained_variance = pca_check.explained_variance_ratio_.cumsum()
    n_95 = next(i for i, v in enumerate(cumulative_explained_variance) if v >= 0.95) + 1
    print(f"Components needed for 95% variance: {n_95}")

    pca = PCA(n_components=n_95)
    v_pca = pca.fit_transform(v_scaled)
    print(f"Variance explained: {pca.explained_variance_ratio_.cumsum()[-1]*100:.2f}%")

    pca_df = pd.DataFrame(v_pca, columns=[f'V_pca_{i}' for i in range(n_95)])
    df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)

    df = df.drop(columns=v_columns)

    print("PCA done!")
    return df


def drop_unnecessary_columns(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    # Drop original columns that were replaced by engineered features
    cols_to_drop = [
        'TransactionID', 
        'TransactionDT',   
        'TransactionAmt',  
    ] + categorical_columns  

    df = df.drop(columns=cols_to_drop)
    print(f"Final shape: {df.shape}")
    return df


def run_feature_engineering(
    input_path: str = "data/processed/clean_data.parquet",
    output_path: str = "data/processed/features.parquet"
) -> pd.DataFrame:

    df = pd.read_parquet(input_path)
    print("Loaded shape:", df.shape)

    df = create_time_features(df)
    df = create_binary_features(df)
    df = create_aggregation_features(df)
    df, categorical_columns = encode_categorical(df)
    df = apply_pca(df)
    df = drop_unnecessary_columns(df, categorical_columns)

    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    run_feature_engineering()
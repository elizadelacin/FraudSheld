import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE


def load_features(input_path: str = "../data/processed/features.parquet"):
    df = pd.read_parquet(input_path)
    print("Shape:", df.shape)
    print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — Normal: {(y_balanced == 0).sum()}, Fraud: {(y_balanced == 1).sum()}")
    return X_balanced, y_balanced


def train_model(X_train_balanced, y_train_balanced, X_test, y_test):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=20,
        n_jobs=-1
    )

    model.fit(
        X_train_balanced, y_train_balanced,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    return model


def evaluate_model(model, X_test, y_test, threshold: float = 0.3):
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("=== Threshold: 0.5 ===")
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print(f"\n=== Threshold: {threshold} ===")
    y_pred_custom = (y_pred_proba >= threshold).astype(int)
    print(classification_report(y_test, y_pred_custom))
    print(confusion_matrix(y_test, y_pred_custom))

    return y_pred_proba


def save_model(model, output_path: str = "../models/xgb_fraud.joblib"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")


def run_shap_analysis(model, X_test, y_test, sample_size: int = 2000):
    normal_sample = X_test[y_test == 0].sample(n=int(sample_size * 0.965), random_state=42)
    fraud_sample = X_test[y_test == 1].sample(n=int(sample_size * 0.035), random_state=42)

    X_shap = pd.concat([normal_sample, fraud_sample])
    y_shap = pd.concat([y_test[normal_sample.index], y_test[fraud_sample.index]])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    print("SHAP values shape:", shap_values.shape)

    # Feature importance bar chart
    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=20, show=True)

    # Detailed summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_shap, max_display=20, show=True)

    # Waterfall plot for first fraud transaction
    fraud_indices = np.where(y_shap == 1)[0]
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[fraud_indices[0]],
            base_values=explainer.expected_value,
            data=X_shap.iloc[fraud_indices[0]],
            feature_names=X_shap.columns.tolist()
        )
    )


def run_training(
    input_path: str = "data/processed/features.parquet",
    model_path: str = "models/xgb_fraud.joblib"
):
    X, y = load_features(input_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    model = train_model(X_train_balanced, y_train_balanced, X_test, y_test)
    evaluate_model(model, X_test, y_test)
    save_model(model, model_path)
    run_shap_analysis(model, X_test, y_test)
    print("Training pipeline complete!")
    return model


if __name__ == "__main__":
    run_training()
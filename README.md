Real-time credit card fraud detection system built on 590K transactions from the IEEE-CIS Kaggle competition. The project covers the full lifecycle — data cleaning, feature engineering, model training, a REST API, and an interactive dashboard with per-prediction SHAP explanations.

Model Performance
ROC-AUC0.922AlgorithmXGBoostClass imbalanceSMOTEDecision threshold0.30 (optimized for recall)Training data590,540 transactionsFeatures163

Project Structure
FraudShield/
├── notebooks/
│   ├── 01_cleaning.ipynb       # Merge, null handling, type fixes
│   ├── 02_eda.ipynb            # Fraud patterns, distributions
│   ├── 03_features.ipynb       # Feature engineering + PCA
│   └── 04_modeling.ipynb       # Training, evaluation, SHAP
│
├── src/
│   ├── preprocess.py           # Cleaning functions
│   ├── features.py             # Feature engineering functions
│   ├── train.py                # Training + SHAP analysis
│   └── pipeline.py             # End-to-end pipeline runner
│
├── api/
│   └── main.py                 # FastAPI prediction endpoint
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard
│
├── data/
│   ├── raw/                    # Original CSVs
│   └── processed/              # Cleaned + engineered parquets
│
├── models/
│   └── xgb_fraud.joblib
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

Feature Engineering
The raw dataset has 400+ columns including anonymized Vesta features (V1–V339), device/email metadata, and transaction details. Key engineering steps:

Time — extracted hour, day_of_week, is_peak_hour from transaction timestamp
Amount — log1p transform to handle heavy right skew
Aggregations — card-level and email-level mean/std/count of transaction amounts
Binary flags — is_mobile, is_anonymous_email, is_high_risk_product, is_discover
PCA — reduced V1–V339 to ~30 components (95% variance retained)
Label encoding — categorical columns: card type, device type, product category, email domain


Quickstart
1. Clone and install
bashgit clone https://github.com/yourusername/FraudShield.git
cd FraudShield

python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
2. Download data
Download train_transaction.csv and train_identity.csv from Kaggle and place them in data/raw/.
3. Run the full pipeline
bashpython src/pipeline.py
This runs preprocessing → feature engineering → model training in one command and saves the model to models/xgb_fraud.joblib.
4. Start the API
bashuvicorn api.main:app --reload
# http://localhost:8000/docs
5. Start the dashboard
bashstreamlit run dashboard/app.py
# http://localhost:8501

Docker
bashdocker-compose up --build
ServiceURLREST APIhttp://localhost:8000API Docs (Swagger)http://localhost:8000/docsDashboardhttp://localhost:8501

API
POST /predict
Send any subset of features — missing values default to 0.
bashcurl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amt_log": 4.5, "hour": 7, "is_mobile": 1, "is_anonymous_email": 1}'
json{
  "is_fraud": true,
  "fraud_probability": 0.9717,
  "risk_level": "HIGH"
}

Dashboard
Three tabs:

Predict — enter transaction details, get an instant fraud score with a SHAP bar chart explaining which features drove the prediction
EDA — explore fraud patterns across time, product category, card network, device type, and email domain
Model — performance metrics, feature engineering summary, and pipeline overview


Tech Stack
LayerToolsMLXGBoost, scikit-learn, imbalanced-learnExplainabilitySHAPAPIFastAPI, UvicornDashboardStreamlit, PlotlyDataPandas, NumPy, PyArrowDevOpsDocker, Docker Compose
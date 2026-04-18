import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import os

# ─── Config ───────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"
DATA_PATH = "data/processed/clean_data.parquet"
MODEL_PATH = "models/xgb_fraud.joblib"
FEATURES_PATH = "data/processed/features.parquet"

st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f1628;
    border-right: 1px solid #1e2d4a;
}

/* Title */
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
}
.main-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    margin-top: -8px;
    margin-bottom: 24px;
    font-family: 'Space Mono', monospace;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2d4a;
}

/* Metric cards */
.metric-card {
    background: #0f1628;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #3b82f6;
}
.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Result cards */
.result-fraud {
    background: linear-gradient(135deg, #2d0a0a, #1a0505);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}
.result-legit {
    background: linear-gradient(135deg, #0a2d1a, #051a0d);
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
}
.result-prob {
    font-size: 2.5rem;
    font-weight: 700;
    margin-top: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0f1628;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
    color: #64748b;
    border-radius: 6px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background-color: #1e3a5f !important;
    color: #3b82f6 !important;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #1e2d4a;
    margin: 24px 0;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

/* Plotly bg override */
.js-plotly-plot .plotly .bg {
    fill: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Plotly theme ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,22,40,0.8)",
    font=dict(family="DM Sans", color="#94a3b8", size=12),
    title_font=dict(family="Space Mono", color="#e2e8f0", size=13),
    xaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a", tickcolor="#1e2d4a"),
    yaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a", tickcolor="#1e2d4a"),
    margin=dict(l=40, r=20, t=50, b=40),
)

# ─── Data & Model loading ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        cols = [
            'isFraud', 'TransactionAmt', 'TransactionDT',
            'ProductCD', 'card4', 'P_emaildomain', 'DeviceType'
        ]
        return pd.read_parquet(DATA_PATH, columns=cols)
    return None

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_features_data():
    if os.path.exists(FEATURES_PATH):
        return pd.read_parquet(FEATURES_PATH)
    return None

df = load_data()
model = load_model()

# ─── Header ───────────────────────────────────────────────────────────────────
col_logo, col_status = st.columns([5, 1])
with col_logo:
    st.markdown('<div class="main-title">🛡️ FraudShield</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Real-Time Transaction Fraud Detection · XGBoost + SHAP</div>', unsafe_allow_html=True)
with col_status:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("✅ API Online")
    except:
        st.error("❌ API Offline")

st.markdown("<hr>", unsafe_allow_html=True)

# ─── Top KPIs ─────────────────────────────────────────────────────────────────
if df is not None:
    total = len(df)
    fraud_count = df['isFraud'].sum()
    fraud_rate = df['isFraud'].mean() * 100
    avg_amt = df['TransactionAmt'].mean()
    avg_fraud_amt = df[df['isFraud'] == 1]['TransactionAmt'].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    metrics = [
        (k1, f"{total:,}", "Total Transactions"),
        (k2, f"{fraud_count:,}", "Fraud Cases"),
        (k3, f"{fraud_rate:.2f}%", "Fraud Rate"),
        (k4, f"${avg_amt:.0f}", "Avg Transaction"),
        (k5, f"${avg_fraud_amt:.0f}", "Avg Fraud Amount"),
    ]
    for col, val, label in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  PREDICT", "📊  EDA", "📈  MODEL"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Transaction Input</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0, step=10.0)
        hour = st.slider("Hour of Day", 0, 23, 14)
        is_peak = 1 if 6 <= hour <= 9 else 0
        st.caption(f"⏰ {'Peak hour (6–9 AM)' if is_peak else 'Off-peak hour'}")

    with c2:
        device = st.selectbox("Device Type", ["Desktop", "Mobile"])
        email_type = st.selectbox("Email Domain Type", ["Normal", "Anonymous (ProtonMail, etc.)"])
        product = st.selectbox("Product Category", ["W", "H", "C", "S", "R"])

    with c3:
        card1_count = st.number_input("Card Transaction Count (historical)", min_value=1, value=25)
        card1_amt_mean = st.number_input("Card Average Amount ($)", min_value=0.0, value=120.0)
        card1_amt_std = st.number_input("Card Std Dev Amount ($)", min_value=0.0, value=45.0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Run Fraud Detection", use_container_width=True, type="primary")

    if predict_btn:
        payload = {
            "amt_log": float(np.log1p(amt)),
            "hour": int(hour),
            "is_peak_hour": int(is_peak),
            "is_mobile": 1 if device == "Mobile" else 0,
            "is_anonymous_email": 1 if "Anonymous" in email_type else 0,
            "is_high_risk_product": 1 if product == "C" else 0,
            "card1_count": int(card1_count),
            "card1_amt_mean": float(card1_amt_mean),
            "card1_amt_std": float(card1_amt_std),
        }

        with st.spinner("Analyzing transaction..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                result = response.json()

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

                res_col, gauge_col, shap_col = st.columns([1, 1.2, 1.8])

                with res_col:
                    is_fraud = result["is_fraud"]
                    prob = result["fraud_probability"]
                    risk = result["risk_level"]

                    card_class = "result-fraud" if is_fraud else "result-legit"
                    label = "🚨 FRAUD" if is_fraud else "✅ LEGITIMATE"
                    prob_color = "#ef4444" if is_fraud else "#22c55e"

                    st.markdown(f"""
                    <div class="{card_class}">
                        <div class="result-label">{label}</div>
                        <div class="result-prob" style="color:{prob_color}">{prob*100:.1f}%</div>
                        <div style="color:#64748b; font-size:0.85rem; margin-top:8px">Risk: <b style="color:{prob_color}">{risk}</b></div>
                    </div>""", unsafe_allow_html=True)

                with gauge_col:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={"suffix": "%", "font": {"size": 28, "family": "Space Mono", "color": "#e2e8f0"}},
                        title={"text": "Fraud Probability", "font": {"size": 12, "family": "Space Mono", "color": "#64748b"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#1e2d4a", "tickfont": {"color": "#64748b", "size": 10}},
                            "bar": {"color": "#ef4444" if prob >= 0.7 else "#f59e0b" if prob >= 0.3 else "#22c55e", "thickness": 0.3},
                            "bgcolor": "#0f1628",
                            "bordercolor": "#1e2d4a",
                            "steps": [
                                {"range": [0, 30], "color": "#0a2d1a"},
                                {"range": [30, 70], "color": "#2d1f0a"},
                                {"range": [70, 100], "color": "#2d0a0a"},
                            ],
                            "threshold": {
                                "line": {"color": "#94a3b8", "width": 2},
                                "thickness": 0.75,
                                "value": 30
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=220,
                        margin=dict(l=20, r=20, t=30, b=10)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # SHAP waterfall
                with shap_col:
                    if model is not None:
                        st.markdown("**Why this prediction?** *(SHAP explanation)*")
                        features_df = load_features_data()
                        if features_df is not None:
                            X_sample = features_df.drop(columns=["isFraud"]).head(200)
                            input_df = pd.DataFrame([payload])
                            for col in X_sample.columns:
                                if col not in input_df.columns:
                                    input_df[col] = 0
                            input_df = input_df[X_sample.columns]

                            try:
                                explainer = shap.TreeExplainer(model)
                                shap_vals = explainer.shap_values(input_df)
                                sv = shap_vals[0]
                                feat_names = input_df.columns.tolist()

                                # Top 10 features by abs impact
                                indices = np.argsort(np.abs(sv))[-10:][::-1]
                                top_feats = [feat_names[i] for i in indices]
                                top_vals = [sv[i] for i in indices]

                                colors = ["#ef4444" if v > 0 else "#3b82f6" for v in top_vals]

                                fig_shap = go.Figure(go.Bar(
                                    x=top_vals[::-1],
                                    y=top_feats[::-1],
                                    orientation='h',
                                    marker_color=colors[::-1],
                                    marker_line_width=0,
                                ))
                                shap_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"}
                                fig_shap.update_layout(
                                    **shap_layout,
                                    height=260,
                                    title="Top Feature Contributions",
                                    xaxis_title="SHAP Value",
                                    margin=dict(l=10, r=10, t=40, b=30),
                                )
                                st.plotly_chart(fig_shap, use_container_width=True)
                                st.caption("🔴 Increases fraud probability  |  🔵 Decreases fraud probability")
                            except Exception as e:
                                st.info(f"SHAP not available: {e}")
                        else:
                            st.info("Load features.parquet for SHAP explanation.")
                    else:
                        st.info("Model not loaded for SHAP.")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Run: `uvicorn api.main:app --reload`")
            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if df is None:
        st.warning("Data not found at data/processed/clean_data.parquet")
    else:
        df_eda = df.copy()
        df_eda['hour'] = (df_eda['TransactionDT'] // 3600) % 24
        df_eda['label'] = df_eda['isFraud'].map({0: 'Legitimate', 1: 'Fraud'})

        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        col_pie, col_box = st.columns(2)

        with col_pie:
            fraud_counts = df_eda['isFraud'].value_counts().reset_index()
            fraud_counts.columns = ['isFraud', 'count']
            fraud_counts['label'] = fraud_counts['isFraud'].map({0: 'Legitimate', 1: 'Fraud'})
            fig = px.pie(fraud_counts, values='count', names='label',
                         title='Transaction Distribution',
                         color='label',
                         color_discrete_map={'Legitimate': '#3b82f6', 'Fraud': '#ef4444'},
                         hole=0.55)
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_traces(textfont_size=13, marker=dict(line=dict(color='#0a0e1a', width=2)))
            st.plotly_chart(fig, use_container_width=True)

        with col_box:
            fig = px.box(df_eda, x='label', y='TransactionAmt',
                         title='Transaction Amount by Class',
                         color='label',
                         color_discrete_map={'Legitimate': '#3b82f6', 'Fraud': '#ef4444'},
                         labels={'label': '', 'TransactionAmt': 'Amount ($)'},
                         points=False)
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Amount Analysis</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(df_eda, x='TransactionAmt', nbins=100,
                                title='Transaction Amount Distribution',
                                color_discrete_sequence=['#3b82f6'])
        fig_hist.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown('<div class="section-header">Fraud Patterns</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            # Hourly fraud rate
            hourly = df_eda.groupby('hour')['isFraud'].mean().reset_index()
            hourly.columns = ['Hour', 'FraudRate']
            hourly['FraudRate'] *= 100
            fig = px.line(hourly, x='Hour', y='FraudRate',
                          title='Fraud Rate by Hour of Day (%)',
                          markers=True,
                          labels={'FraudRate': 'Fraud Rate (%)', 'Hour': 'Hour'},
                          color_discrete_sequence=['#3b82f6'])
            fig.add_vrect(x0=6, x1=9, fillcolor="#ef4444", opacity=0.1,
                          annotation_text="Peak Risk", annotation_position="top left")
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Product fraud rate
            product_fraud = df_eda.groupby('ProductCD')['isFraud'].mean().reset_index()
            product_fraud.columns = ['ProductCD', 'FraudRate']
            product_fraud['FraudRate'] *= 100
            product_fraud = product_fraud.sort_values('FraudRate', ascending=True)
            fig = px.bar(product_fraud, x='FraudRate', y='ProductCD',
                         orientation='h',
                         title='Fraud Rate by Product Category (%)',
                         color='FraudRate', color_continuous_scale='Reds',
                         labels={'FraudRate': 'Fraud Rate (%)'})
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            # Card type
            card4_fraud = df_eda.groupby('card4')['isFraud'].mean().reset_index()
            card4_fraud.columns = ['card4', 'FraudRate']
            card4_fraud['FraudRate'] *= 100
            card4_fraud = card4_fraud.sort_values('FraudRate', ascending=True).dropna()
            fig = px.bar(card4_fraud, x='FraudRate', y='card4',
                         orientation='h',
                         title='Fraud Rate by Card Network (%)',
                         color='FraudRate', color_continuous_scale='Reds',
                         labels={'FraudRate': 'Fraud Rate (%)', 'card4': 'Card Network'})
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            # Device type
            device_fraud = df_eda.groupby('DeviceType')['isFraud'].mean().reset_index()
            device_fraud.columns = ['DeviceType', 'FraudRate']
            device_fraud['FraudRate'] *= 100
            device_fraud = device_fraud.sort_values('FraudRate', ascending=True).dropna()
            fig = px.bar(device_fraud, x='FraudRate', y='DeviceType',
                         orientation='h',
                         title='Fraud Rate by Device Type (%)',
                         color='FraudRate', color_continuous_scale='Reds',
                         labels={'FraudRate': 'Fraud Rate (%)', 'DeviceType': 'Device'})
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # Email domain
        st.markdown('<div class="section-header">Email Domain Risk</div>', unsafe_allow_html=True)
        email_fraud = df_eda.groupby('P_emaildomain')['isFraud'].mean().reset_index()
        email_fraud.columns = ['EmailDomain', 'FraudRate']
        email_fraud['FraudRate'] *= 100
        email_fraud = email_fraud.sort_values('FraudRate', ascending=False).head(15).dropna()
        fig = px.bar(email_fraud, x='EmailDomain', y='FraudRate',
                     title='Top 15 Email Domains by Fraud Rate (%)',
                     color='FraudRate', color_continuous_scale='Reds',
                     labels={'FraudRate': 'Fraud Rate (%)', 'EmailDomain': 'Email Domain'})
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    perf_metrics = [
        (m1, "0.922", "ROC-AUC Score"),
        (m2, "XGBoost", "Algorithm"),
        (m3, "163", "Features Used"),
        (m4, "0.30", "Decision Threshold"),
    ]
    for col, val, label in perf_metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feature Engineering Summary</div>', unsafe_allow_html=True)

    fe_data = {
        "Feature Group": [
            "Time Features", "Amount Features", "Binary Flags",
            "Aggregation Features", "PCA (V Columns)", "Encoded Categoricals"
        ],
        "Examples": [
            "hour, day, is_peak_hour",
            "amt_log (log1p transform)",
            "is_mobile, is_anonymous_email, is_discover, is_high_risk_product",
            "card1_count, card1_amt_mean, email_amt_mean",
            "V_pca_0 … V_pca_N (95% variance retained)",
            "card4_encoded, DeviceType_encoded, ProductCD_encoded"
        ],
        "Count": ["3", "1", "4", "6", "~30", "~15"]
    }
    fe_df = pd.DataFrame(fe_data)
    st.dataframe(fe_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Pipeline Architecture</div>', unsafe_allow_html=True)

    pipeline_html = """
    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin: 16px 0;">
        <div style="background:#0f1628; border:1px solid #1e3a5f; border-radius:8px; padding:12px 18px; font-family:'Space Mono',monospace; font-size:0.8rem; color:#3b82f6;">📥 Raw CSV</div>
        <div style="color:#1e2d4a; font-size:1.5rem;">→</div>
        <div style="background:#0f1628; border:1px solid #1e3a5f; border-radius:8px; padding:12px 18px; font-family:'Space Mono',monospace; font-size:0.8rem; color:#3b82f6;">🧹 Preprocessing</div>
        <div style="color:#1e2d4a; font-size:1.5rem;">→</div>
        <div style="background:#0f1628; border:1px solid #1e3a5f; border-radius:8px; padding:12px 18px; font-family:'Space Mono',monospace; font-size:0.8rem; color:#3b82f6;">⚙️ Feature Engineering</div>
        <div style="color:#1e2d4a; font-size:1.5rem;">→</div>
        <div style="background:#0f1628; border:1px solid #1e3a5f; border-radius:8px; padding:12px 18px; font-family:'Space Mono',monospace; font-size:0.8rem; color:#3b82f6;">⚖️ SMOTE</div>
        <div style="color:#1e2d4a; font-size:1.5rem;">→</div>
        <div style="background:#0f1628; border:1px solid #1e3a5f; border-radius:8px; padding:12px 18px; font-family:'Space Mono',monospace; font-size:0.8rem; color:#3b82f6;">🤖 XGBoost</div>
        <div style="color:#1e2d4a; font-size:1.5rem;">→</div>
        <div style="background:#1a2d1a; border:1px solid #22c55e; border-radius:8px; padding:12px 18px; font-family:'Space Mono',monospace; font-size:0.8rem; color:#22c55e;">🛡️ Prediction</div>
    </div>
    """
    st.markdown(pipeline_html, unsafe_allow_html=True)

    st.markdown('<div class="section-header">SHAP Global Analysis</div>', unsafe_allow_html=True)
    st.info("💡 Run SHAP from notebooks/04_modeling.ipynb to view global feature importance charts. "
            "Transaction-level SHAP explanations are available in the **Predict** tab.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Technology Stack</div>', unsafe_allow_html=True)

    tech_cols = st.columns(4)
    techs = [
        ("🐍 Python", "Core language"),
        ("📊 XGBoost", "ML model (AUC 0.922)"),
        ("⚡ FastAPI", "REST API backend"),
        ("🎨 Streamlit", "Interactive dashboard"),
        ("🔍 SHAP", "Model explainability"),
        ("🐼 Pandas", "Data processing"),
        ("📈 Plotly", "Visualizations"),
        ("🐳 Docker", "Containerization"),
    ]
    for i, (name, desc) in enumerate(techs):
        with tech_cols[i % 4]:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:8px; text-align:left; padding:14px 16px;">
                <div style="font-size:1rem; color:#e2e8f0; font-weight:600;">{name}</div>
                <div class="metric-label" style="text-transform:none; letter-spacing:0;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#1e2d4a; font-family:'Space Mono',monospace; font-size:0.7rem; padding:8px 0;">
    FraudShield · XGBoost Fraud Detection System · Built with FastAPI + Streamlit
</div>
""", unsafe_allow_html=True)
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
    border-radius: 20px; padding: 2.5rem 3rem; margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(102,126,234,0.4);
}
.hero-title  { font-size: 2.8rem; font-weight: 900; color: #ffffff; margin: 0; letter-spacing: -1px; }
.hero-subtitle { font-size: 1.1rem; color: rgba(255,255,255,0.85); margin-top: 0.5rem; font-weight: 300; }
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3); border-radius: 50px;
    padding: 0.3rem 1rem; font-size: 0.8rem; color: white; margin-top: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.12); border-radius: 16px;
    padding: 1.4rem 1.6rem; text-align: center; backdrop-filter: blur(10px);
    transition: transform 0.2s ease; margin-bottom: 1rem;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(102,126,234,0.3); }
.metric-icon  { font-size: 2.2rem; margin-bottom: 0.4rem; }
.metric-value {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-label { font-size: 0.8rem; color: #9ca3af; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }
.section-header { display: flex; align-items: center; gap: 0.8rem; margin: 2rem 0 1.2rem 0; }
.section-header .icon-circle { width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; }
.section-header h2 { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin: 0; }
.section-header p  { margin: 0; color: #9ca3af; font-size: 0.85rem; }
.info-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
    border-left: 4px solid #667eea;
}
.info-card h4 { color: #a78bfa; margin: 0 0 0.4rem 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px; }
.info-card p  { color: #d1d5db; margin: 0; font-size: 0.95rem; }
.result-fraud {
    background: linear-gradient(135deg, #ef4444, #dc2626); border-radius: 12px;
    padding: 1.5rem; text-align: center; color: white; font-size: 1.4rem;
    font-weight: 800; box-shadow: 0 8px 30px rgba(239,68,68,0.4); animation: pulse 2s infinite;
}
.result-legit {
    background: linear-gradient(135deg, #10b981, #059669); border-radius: 12px;
    padding: 1.5rem; text-align: center; color: white; font-size: 1.4rem;
    font-weight: 800; box-shadow: 0 8px 30px rgba(16,185,129,0.4);
}
@keyframes pulse {
    0%,100% { box-shadow: 0 8px 30px rgba(239,68,68,0.4); }
    50%      { box-shadow: 0 8px 50px rgba(239,68,68,0.8); }
}
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #9ca3af; padding: 0.5rem 1.2rem; font-weight: 600; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea, #764ba2) !important; color: white !important; }
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2); color: white;
    border: none; border-radius: 10px; padding: 0.7rem 2rem;
    font-weight: 700; font-size: 1rem; width: 100%; transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102,126,234,0.5); }
.success-alert { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); border-radius: 10px; padding: 1rem 1.2rem; color: #6ee7b7; margin-bottom: 0.8rem; }
.warning-alert { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4); border-radius: 10px; padding: 1rem 1.2rem; color: #fcd34d; margin-bottom: 0.8rem; }
.error-alert   { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); border-radius: 10px; padding: 1rem 1.2rem; color: #fca5a5; margin-bottom: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# DARK PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────
DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e0e0e0', family='Inter'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.1)'),
    margin=dict(l=20, r=20, t=50, b=20)
)

import os
import pandas as pd
import streamlit as st
import joblib


CSV_PATH = "creditcard.csv"
PKL_PATH = "xgboost.pkl"

FEATURE_COLS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# ─────────────────────────────────────────────────────────────────────
# AUTO-LOAD DATASET
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataset():
    try:
        if not os.path.isfile(CSV_PATH):
            return None, f"File not found: {CSV_PATH}"

        df = pd.read_csv(CSV_PATH)
        df.drop_duplicates(inplace=True)

        return df, ""
    
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────────────
# AUTO-LOAD MODEL
# ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        if not os.path.isfile(PKL_PATH):
            return None, f"File not found: {PKL_PATH}"

        model = joblib.load(PKL_PATH)

        return model, ""
    
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

df, data_error = load_dataset()
model, model_error = load_model()

if data_error:
    st.error(f"Dataset Error: {data_error}")
else:
    st.success("Dataset loaded successfully")
    st.write(df.head())

if model_error:
    st.error(f"Model Error: {model_error}")
else:
    st.success("Model loaded successfully")

# ─────────────────────────────────────────────────────────────────────
# AUTO-LOAD MODEL
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(PKL_PATH):
        return None, False, f"xgboost.pkl not found at: {PKL_PATH}"
    try:
        with open(PKL_PATH, "rb") as f:
            model = pickle.load(f)
        return model, True, ""
    except Exception as e:
        return None, False, str(e)

# Load both on startup
df, csv_err           = load_dataset()
xgb_model, xgb_ok, xgb_err = load_model()

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
for key in ['eval_results', 'X_test', 'y_test', 'fitted_scaler']:
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
      <div style="font-size:3rem">🛡️</div>
      <div style="font-size:1.1rem; font-weight:800; color:#a78bfa; margin-top:0.3rem">FraudGuard AI</div>
      <div style="font-size:0.75rem; color:#6b7280; margin-top:0.2rem">Credit Card Fraud Detection</div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.08); margin: 0.5rem 0 1.2rem 0">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Home & Dataset",
         "📊 EDA & Visualisations",
         "📋 Model Status & Evaluate",
         "🔮 Predict Transaction",
         "📈 Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08)'>", unsafe_allow_html=True)

    # File status
    st.markdown("<div style='font-size:0.78rem;color:#a78bfa;font-weight:700;margin-bottom:0.4rem'>📂 File Status</div>", unsafe_allow_html=True)
    csv_status = "✅ creditcard.csv" if df is not None else "❌ creditcard.csv missing"
    pkl_status = "✅ xgboost.pkl"   if xgb_ok        else "❌ xgboost.pkl missing"
    csv_color  = "#6ee7b7" if df is not None else "#fca5a5"
    pkl_color  = "#6ee7b7" if xgb_ok        else "#fca5a5"
    st.markdown(f"<div style='font-size:0.75rem;color:{csv_color}'>{csv_status}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.75rem;color:{pkl_color}'>{pkl_status}</div>", unsafe_allow_html=True)

    st.markdown("""
    <hr style='border-color:rgba(255,255,255,0.08);margin:0.8rem 0'>
    <div style="font-size:0.75rem; color:#6b7280; text-align:center">
      <b style="color:#a78bfa">Dataset:</b> Kaggle ULB Credit Card Fraud<br>
      284,807 transactions · 492 fraud cases<br><br>
      <b style="color:#a78bfa">Model:</b> XGBoost (xgboost.pkl)
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">💳 Credit Card Fraud Detection</div>
  <div class="hero-subtitle">AI-powered financial security · XGBoost pre-trained model</div>
  <span class="hero-badge">🎓 Machine Learning Project</span>
  <span class="hero-badge" style="margin-left:0.5rem">⚡ XGBoost Model</span>
  <span class="hero-badge" style="margin-left:0.5rem">📊 284K+ Transactions Analysed</span>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME & DATASET
# ═════════════════════════════════════════════════════════════════════
if page == "🏠 Home & Dataset":

    st.markdown("""
    <div class="section-header">
      <div class="icon-circle" style="background:linear-gradient(135deg,#667eea,#764ba2)">📁</div>
      <div><h2>Dataset Status</h2>
      <p>creditcard.csv is loaded </p></div>
    </div>
    """, unsafe_allow_html=True)

    if df is not None:
        st.markdown(f'<div class="success-alert">✅ <b>creditcard.csv</b> loaded automatically from:<br><code>{CSV_PATH}</code></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="error-alert">❌ <b>creditcard.csv not found!</b><br>Expected location: <code>{CSV_PATH}</code><br>Place creditcard.csv in the same folder as app.py and restart.</div>', unsafe_allow_html=True)

    # Model status
    st.markdown("""
    <div class="section-header">
      <div class="icon-circle" style="background:linear-gradient(135deg,#10b981,#059669)">📦</div>
      <div><h2>Model Status</h2><p>xgboost.pkl loaded automatically from same folder</p></div>
    </div>
    """, unsafe_allow_html=True)

    if xgb_ok:
        st.markdown(f'<div class="success-alert">✅ <b>xgboost.pkl</b> loaded successfully from:<br><code>{PKL_PATH}</code></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="error-alert">❌ <b>xgboost.pkl not found!</b><br>Expected: <code>{PKL_PATH}</code><br>Place xgboost.pkl in the same folder as app.py and restart.</div>', unsafe_allow_html=True)

    if df is not None:
        fraud_count = int(df['Class'].sum())
        legit_count = int((df['Class'] == 0).sum())
        fraud_pct   = fraud_count / len(df) * 100

        st.markdown("""
        <div class="section-header">
          <div class="icon-circle" style="background:linear-gradient(135deg,#f59e0b,#d97706)">📊</div>
          <div><h2>Dataset Overview</h2></div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, icon, val, label in [
            (c1, "📦", f"{len(df):,}",     "Total Transactions"),
            (c2, "✅", f"{legit_count:,}", "Legitimate"),
            (c3, "🚨", f"{fraud_count:,}", "Fraudulent"),
            (c4, "📉", f"{fraud_pct:.3f}%","Fraud Rate"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-icon">{icon}</div>
                  <div class="metric-value">{val}</div>
                  <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.dataframe(df.head(10), use_container_width=True, height=320)
        with col_b:
            for h, v in [
                ("Rows × Columns", f"{df.shape[0]:,} × {df.shape[1]}"),
                ("Null Values",    str(df.isnull().sum().sum())),
                ("Duplicates",     "0 (removed on load)"),
                ("Features",       "V1–V28 (PCA), Time, Amount"),
                ("Target",         "Class (0=Legit, 1=Fraud)"),
                ("Class Balance",  "99.8% Legit / 0.2% Fraud"),
            ]:
                st.markdown(f'<div class="info-card"><h4>{h}</h4><p>{v}</p></div>', unsafe_allow_html=True)

        with st.expander("📊 Statistical Summary"):
            st.dataframe(df.describe().T.style.background_gradient(cmap='Blues'), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE 2 – EDA & VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Visualisations":

    if df is None:
        st.markdown(f'<div class="error-alert">❌ creditcard.csv not found at <code>{CSV_PATH}</code>. Place it in the same folder as app.py and restart.</div>', unsafe_allow_html=True)
        st.stop()

    st.markdown("""
    <div class="section-header">
      <div class="icon-circle" style="background:linear-gradient(135deg,#f64f59,#c0392b)">📊</div>
      <div><h2>Exploratory Data Analysis</h2><p>Visual insights into the credit card fraud dataset</p></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🍩 Class Distribution", "⏰ Time Analysis",
        "💰 Amount Analysis", "🔥 Correlation", "📦 Outliers"
    ])

    with tab1:
        counts = df['Class'].value_counts()
        fig = make_subplots(rows=1, cols=2,
            specs=[[{'type': 'domain'}, {'type': 'xy'}]],
            subplot_titles=("Class Distribution (Donut)", "Transaction Counts (Bar)"))
        fig.add_trace(go.Pie(
            labels=['Non-Fraud', 'Fraud'], values=counts.values, hole=0.55, pull=[0, 0.12],
            marker=dict(colors=['#667eea', '#f64f59'])), row=1, col=1)
        fig.add_trace(go.Bar(
            x=['Non-Fraud', 'Fraud'], y=counts.values,
            marker=dict(color=['#667eea', '#f64f59']),
            text=[f"{v:,}" for v in counts.values], textposition='auto'), row=1, col=2)
        fig.update_layout(title="Fraud vs Legitimate Transactions", **DARK, height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="info-card" style="border-left-color:#10b981"><h4>✅ Legitimate</h4>'
                        f'<p><b style="font-size:1.4rem;color:#10b981">{counts[0]:,}</b> ({counts[0]/len(df)*100:.2f}%)</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="info-card" style="border-left-color:#ef4444"><h4>🚨 Fraudulent</h4>'
                        f'<p><b style="font-size:1.4rem;color:#ef4444">{counts[1]:,}</b> ({counts[1]/len(df)*100:.4f}%)</p></div>', unsafe_allow_html=True)

    with tab2:
        fig1 = px.histogram(df, x='Time', color='Class', nbins=60, barmode='overlay',
            title='Transaction Time Distribution by Class',
            color_discrete_map={0: '#667eea', 1: '#f64f59'}, opacity=0.75)
        fig1.update_layout(**DARK, height=380)
        st.plotly_chart(fig1, use_container_width=True)

        grouped = df.groupby('Time')['Amount'].mean().reset_index()
        fig2 = px.line(grouped, x='Time', y='Amount',
            title='Average Transaction Amount Over Time', color_discrete_sequence=['#a78bfa'])
        fig2.update_layout(**DARK, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig = px.box(df, x='Class', y='Amount', color='Class',
            title='Transaction Amount by Class',
            color_discrete_map={0: '#667eea', 1: '#f64f59'})
        fig.update_traces(boxpoints=False)
        fig.update_layout(**DARK, height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        fig = px.violin(df.sample(min(5000, len(df))), x='Class', y='V1',
            color='Class', box=True, title='V1 Distribution by Class (sample)',
            color_discrete_map={0: '#667eea', 1: '#f64f59'})
        fig.update_layout(**DARK, height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        corr_cols = [c for c in df.columns if c != 'Time']
        corrmat   = df[corr_cols].sample(min(10000, len(df))).corr()
        fig, ax   = plt.subplots(figsize=(18, 10))
        fig.patch.set_facecolor('#0f0c29')
        ax.set_facecolor('#0f0c29')
        sns.heatmap(corrmat, vmax=0.8, square=True, cmap='coolwarm',
                    annot=True, fmt='.1f', linewidths=0.3, annot_kws={'size': 7},
                    ax=ax, cbar_kws={"shrink": 0.7})
        ax.set_title('Feature Correlation Heatmap', color='white', fontsize=16, pad=15)
        plt.xticks(color='#9ca3af', rotation=45, ha='right', fontsize=7)
        plt.yticks(color='#9ca3af', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)

    with tab5:
        num_df = df.select_dtypes(include=['number']).drop(columns=['Class'])
        Q1, Q3 = num_df.quantile(0.25), num_df.quantile(0.75)
        IQR    = Q3 - Q1
        outliers = ((num_df < (Q1 - 1.5*IQR)) | (num_df > (Q3 + 1.5*IQR))).sum()
        fig = px.bar(x=outliers.index, y=outliers.values,
            title='Outlier Count per Feature (IQR Method)',
            labels={'x': 'Feature', 'y': 'Outlier Count'},
            color=outliers.values, color_continuous_scale='RdPu')
        fig.update_layout(**DARK, height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Total outliers: **{outliers.sum():,}** across all features")


# ═════════════════════════════════════════════════════════════════════
# PAGE 3 – MODEL STATUS & EVALUATE
# ═════════════════════════════════════════════════════════════════════
elif page == "📋 Model Status & Evaluate":

    st.markdown("""
    <div class="section-header">
      <div class="icon-circle" style="background:linear-gradient(135deg,#f093fb,#f5576c)">📋</div>
      <div><h2>XGBoost Model Status & Evaluation</h2>
      <p>Model loaded automatically from xgboost.pkl — no training needed!</p></div>
    </div>
    """, unsafe_allow_html=True)

    if xgb_ok:
        st.markdown(f'<div class="success-alert">✅ <b>XGBoost</b> loaded from <code>{PKL_PATH}</code></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="error-alert">❌ <b>XGBoost not loaded</b> — {xgb_err}<br>Place <code>xgboost.pkl</code> in the same folder as app.py and restart.</div>', unsafe_allow_html=True)
        st.stop()

    if df is None:
        st.markdown(f'<div class="error-alert">❌ <b>creditcard.csv not found</b> at <code>{CSV_PATH}</code><br>Place it in the same folder as app.py and restart.</div>', unsafe_allow_html=True)
        st.stop()

    st.markdown("---")
    st.markdown("""
    <div class="section-header">
      <div class="icon-circle" style="background:linear-gradient(135deg,#4facfe,#00f2fe)">🧪</div>
      <div><h2>Evaluate Model</h2>
      <p>Run XGBoost on a test split of creditcard.csv</p></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Evaluation Configuration", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            test_size   = st.slider("Test Split (%)", 10, 40, 20) / 100
        with c2:
            random_seed = st.number_input("Random Seed", value=42, min_value=0)

    if st.button("🔍 Evaluate XGBoost Model"):
        with st.spinner("⏳ Running evaluation…"):
            df2 = df.copy()
            df2['Class'] = df2['Class'].astype(int)
            X = df2[FEATURE_COLS]
            Y = df2['Class']

            _, X_test_raw, _, y_test = train_test_split(
                X, Y, test_size=test_size, random_state=int(random_seed))

            sc = StandardScaler()
            X_test_sc = pd.DataFrame(sc.fit_transform(X_test_raw), columns=X_test_raw.columns)
            st.session_state.fitted_scaler = sc

            y_pred = xgb_model.predict(X_test_sc)
            y_prob = xgb_model.predict_proba(X_test_sc)[:, 1]

            st.session_state.eval_results = [{
                "Model":     "XGBoost",
                "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
                "ROC AUC":   round(roc_auc_score(y_test, y_prob), 4),
                "_y_pred":   y_pred,
                "_y_prob":   y_prob,
            }]
            st.session_state.X_test = X_test_sc
            st.session_state.y_test = y_test

        st.markdown('<div class="success-alert">✅ Evaluation complete!</div>', unsafe_allow_html=True)

    if st.session_state.eval_results:
        r = st.session_state.eval_results[0]
        keys = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
        st.markdown("#### 📊 Results")
        st.dataframe(pd.DataFrame([{k: r[k] for k in keys}]), use_container_width=True)

        metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
        colors  = ['#667eea', '#f64f59', '#10b981', '#f59e0b', '#a78bfa']
        fig = go.Figure()
        for m, c in zip(metrics, colors):
            fig.add_trace(go.Bar(name=m, x=["XGBoost"], y=[r[m]],
                marker_color=c, text=[f"{r[m]:.4f}"], textposition='auto'))
        fig.update_layout(title="XGBoost Metrics", barmode='group',
                          **DARK, height=420, legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(102,126,234,0.2),rgba(118,75,162,0.2));
                    border:1px solid rgba(167,139,250,0.4);border-radius:14px;padding:1.4rem 1.8rem;margin-top:1rem">
          <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;color:#a78bfa;font-weight:700">⚡ XGBoost Results</div>
          <div style="color:#d1d5db;margin-top:0.3rem">
            F1 Score: <b style="color:#34d399">{r['F1 Score']:.4f}</b> &nbsp;|&nbsp;
            ROC AUC:  <b style="color:#60a5fa">{r['ROC AUC']:.4f}</b> &nbsp;|&nbsp;
            Accuracy: <b style="color:#fbbf24">{r['Accuracy']:.4f}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════
# PAGE 4 – PREDICT TRANSACTION
# ═════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Transaction":

    # ─────────────────────────────────────────────────────────────
    # LOAD SCALER 
    # ─────────────────────────────────────────────────────────────
    SCALER_PATH = "scaler.pkl"

    @st.cache_resource
    def load_scaler():
        if os.path.isfile(SCALER_PATH):
            return joblib.load(SCALER_PATH)
        return None

    scaler = load_scaler()

    # ─────────────────────────────────────────────────────────────
    # CHECK MODEL
    # ─────────────────────────────────────────────────────────────
    if not xgb_ok:
        st.markdown(f'<div class="error-alert">❌ xgboost.pkl not found at <code>{PKL_PATH}</code></div>', unsafe_allow_html=True)
        st.stop()

    # ─────────────────────────────────────────────────────────────
    # UI HEADER
    # ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="icon-circle" style="background:linear-gradient(135deg,#43e97b,#38f9d7)">🔮</div>
      <div><h2>Real-Time Fraud Prediction</h2>
      <p>Enter transaction details to get instant fraud probability</p></div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # INPUT FIELDS
    # ─────────────────────────────────────────────────────────────
    st.markdown("#### 📝 Enter Transaction Details")

    c1, c2 = st.columns(2)
    with c1:
        time_val = st.number_input("⏰ Time (seconds)", value=0.0)
    with c2:
        amount_val = st.number_input("💵 Amount ($)", value=100.0, min_value=0.0)

    with st.expander("🔢 PCA Features V1–V28 (Advanced)", expanded=False):
        v_vals = {}
        cols_v = st.columns(4)

        for i in range(1, 29):
            with cols_v[(i - 1) % 4]:
                v_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v{i}")

    # ─────────────────────────────────────────────────────────────
    # PREPARE INPUT
    # ─────────────────────────────────────────────────────────────
    input_data = {}

    for col in FEATURE_COLS:
        if col == 'Time':
            input_data[col] = time_val
        elif col == 'Amount':
            input_data[col] = amount_val
        else:
            input_data[col] = v_vals.get(col, 0.0)

    # ─────────────────────────────────────────────────────────────
    # PREDICTION BUTTON
    # ─────────────────────────────────────────────────────────────
    if st.button("🔍 Analyse Transaction"):

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # 🔥 IMPORTANT: Maintain same feature order
        input_df = input_df[FEATURE_COLS]

        # ─────────────────────────────────────────
        # APPLY SCALING (FIXED)
        # ─────────────────────────────────────────
        if scaler is not None:
            input_sc = scaler.transform(input_df)
        else:
            input_sc = input_df.values
            st.warning("⚠️ Scaler not found. Using raw values (accuracy may drop).")

        # ─────────────────────────────────────────
        # PREDICTION
        # ─────────────────────────────────────────
        prediction = xgb_model.predict(input_sc)[0]
        probability = xgb_model.predict_proba(input_sc)[0]

        fraud_prob = probability[1] * 100
        legit_prob = probability[0] * 100

        # ─────────────────────────────────────────
        # OUTPUT
        # ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        if prediction == 1:
            st.markdown(f'''
            <div class="result-fraud">
            🚨 FRAUDULENT TRANSACTION DETECTED!<br>
            <span style="font-size:1rem;">Fraud Probability: {fraud_prob:.2f}%</span>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="result-legit">
            ✅ LEGITIMATE TRANSACTION<br>
            <span style="font-size:1rem;">Legitimate Probability: {legit_prob:.2f}%</span>
            </div>
            ''', unsafe_allow_html=True)

        # ─────────────────────────────────────────
        # GAUGE CHART
        # ─────────────────────────────────────────
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob,
            title={'text': "Fraud Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#f64f59' if prediction == 1 else '#10b981'},
                'steps': [
                    {'range': [0, 30], 'color': 'lightgreen'},
                    {'range': [30, 70], 'color': 'orange'},
                    {'range': [70, 100], 'color': 'red'}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ─────────────────────────────────────────
        # METRICS
        # ─────────────────────────────────────────
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Model", "XGBoost")
        with c2:
            st.metric("Legit Probability", f"{legit_prob:.2f}%")
        with c3:
            st.metric("Fraud Probability", f"{fraud_prob:.2f}%")


# ═════════════════════════════════════════════════════════════════════
# PAGE 5 – MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":

    results = st.session_state.eval_results
    y_test  = st.session_state.y_test

    if not results:
        st.warning("⚠️ No evaluation results yet. Go to **Model Status & Evaluate** and run evaluation first.")
        st.stop()

    r = results[0]

    st.markdown("""
    <div class="section-header">
      <div class="icon-circle" style="background:linear-gradient(135deg,#fa709a,#fee140)">📈</div>
      <div><h2>XGBoost Detailed Performance</h2>
      <p>Confusion matrix · ROC curve · Feature importance</p></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔲 Confusion Matrix", "📉 ROC Curve", "⭐ Feature Importance"])

    with tab1:
        cm = confusion_matrix(y_test, r['_y_pred'])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                    linewidths=1.5, linecolor='#1a1a2e', cbar=False)
        cm_max = cm.max()
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                val = cm[row, col]
                brightness = val / cm_max if cm_max > 0 else 0
                ax.text(col + 0.5, row + 0.5, str(val), ha='center', va='center',
                        fontsize=18, fontweight='bold',
                        color='white' if brightness > 0.5 else 'black')
        ax.set_title("XGBoost Confusion Matrix", color='white', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted', color='#9ca3af', fontsize=11)
        ax.set_ylabel('Actual', color='#9ca3af', fontsize=11)
        ax.set_xticklabels(['Legit (0)', 'Fraud (1)'], color='#cccccc')
        ax.set_yticklabels(['Legit (0)', 'Fraud (1)'], color='#cccccc', rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        tn, fp, fn, tp = cm.ravel()
        c1, c2, c3, c4 = st.columns(4)
        for col, icon, val, label in [
            (c1, "✅", f"{tn:,}", "True Negatives"),
            (c2, "⚠️", f"{fp:,}", "False Positives"),
            (c3, "❌", f"{fn:,}", "False Negatives"),
            (c4, "🚨", f"{tp:,}", "True Positives"),
        ]:
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-icon">{icon}</div>'
                            f'<div class="metric-value">{val}</div>'
                            f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    with tab2:
        fpr, tpr, _ = roc_curve(y_test, r['_y_prob'])
        auc_val = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
            name=f"XGBoost (AUC={auc_val:.4f})",
            line=dict(color='#667eea', width=3), mode='lines',
            fill='tozeroy', fillcolor='rgba(102,126,234,0.1)'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random (AUC=0.5)',
            line=dict(color='rgba(255,255,255,0.3)', dash='dash'), mode='lines'))
        fig.update_layout(
            title=f"ROC Curve — XGBoost (AUC = {auc_val:.4f})",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            **DARK, height=500,
            legend=dict(x=0.55, y=0.1, bgcolor='rgba(0,0,0,0.4)',
                        bordercolor='rgba(255,255,255,0.2)', borderwidth=1))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if hasattr(xgb_model, 'feature_importances_'):
            importances = xgb_model.feature_importances_
            feat_df = pd.DataFrame({
                'Feature':    FEATURE_COLS[:len(importances)],
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(20)

            fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                         title='Top 20 Feature Importances — XGBoost',
                         color='Importance', color_continuous_scale='Purples')
            fig.update_layout(**DARK, height=560)
            fig.update_yaxes(autorange='reversed')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importances not available for this model.")

        st.markdown("#### 🕸️ XGBoost Metrics Radar")
        metrics_list = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
        values = [r[m] for m in metrics_list] + [r[metrics_list[0]]]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values, theta=metrics_list + [metrics_list[0]], fill='toself',
            name='XGBoost', line_color='#667eea', fillcolor='rgba(102,126,234,0.2)'))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1],
                    gridcolor='rgba(255,255,255,0.1)', linecolor='rgba(255,255,255,0.1)',
                    tickcolor='white', tickfont=dict(color='white')),
                angularaxis=dict(linecolor='rgba(255,255,255,0.2)',
                    gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='rgba(255,255,255,0.2)', borderwidth=1),
            height=500, margin=dict(l=50, r=50, t=30, b=50),
            title=dict(text="XGBoost Performance Radar", font=dict(color='white', size=16)))
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(255,255,255,0.08); margin-top: 3rem">
<div style="text-align:center; padding: 1.5rem; color:#6b7280; font-size:0.82rem">
  <b style="color:#a78bfa">💳 Credit Card Fraud Detection</b> · Machine Learning Project<br>
  <span style="color:#a78bfa">⚡ Powered by XGBoost (xgboost.pkl)</span>
</div>
""", unsafe_allow_html=True)

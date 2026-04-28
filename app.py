import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import plotly.express as px
from dateutil.relativedelta import relativedelta
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# ════════════════════════════════════════════════════════════
# 1. KONFIGURASI GLOBAL
# ════════════════════════════════════════════════════════════
st.set_page_config(page_title="Retail Churn Analytics Dashboard", layout="wide")

MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

REGIONAL_MAP = {
    'REGIONAL 1': 'SP',
    'REGIONAL 2': 'SMBR',
    'REGIONAL 6': 'ST'
}

# ════════════════════════════════════════════════════════════
# 2. CORE FUNCTIONS (PREPROCESSING & ML LOGIC)
# ════════════════════════════════════════════════════════════

def cleansing_data(df):
    df = df.copy()
    # Datetime conversion
    for col in ['Tanggal Transaksi', 'Created_at_Dist']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Numeric conversion
    for col in ['Harga', 'TON Quantity', 'Harga_Per_KG']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # ID cleaning
    df['ID Toko'] = pd.to_numeric(df['ID Toko'], errors='coerce')
    
    # Text normalization & Regional Mapping
    text_cols = ['Nama Toko', 'Kabupaten Toko', 'Area AP Toko', 'Brands', 'Cluster Pareto']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            
    if 'Area AP Toko' in df.columns:
        df['Area AP Toko'] = df['Area AP Toko'].map(REGIONAL_MAP).fillna(df['Area AP Toko'])
            
    df['Est_Spend'] = df['TON Quantity'] * df['Harga_Per_KG'].fillna(0)
    return df

def buat_label_churn(df, cutoff, window_days=60):
    """Menentukan churn berdasarkan aktivitas masa depan (Performance Window)"""
    toko_aktif = df[df['Tanggal Transaksi'] <= cutoff]['ID Toko'].unique()
    future_limit = cutoff + pd.Timedelta(days=window_days)
    transaksi_masa_depan = df[(df['Tanggal Transaksi'] > cutoff) & 
                              (df['Tanggal Transaksi'] <= future_limit)]['ID Toko'].unique()
    
    label_list = [{'ID Toko': t, 'Label_Churn': (0 if t in transaksi_masa_depan else 1)} for t in toko_aktif]
    return pd.DataFrame(label_list).set_index('ID Toko')

def feature_engineering(df, df_loyalty, cutoff):
    df_tr = df[df['Tanggal Transaksi'] <= cutoff].copy()
    agg = df_tr.groupby('ID Toko').agg(
        last_trx=('Tanggal Transaksi', 'max'),
        Frequency=('No Transaksi', 'count'),
        Total_Ton=('TON Quantity', 'sum'),
        Monetary=('Est_Spend', 'sum'),
        Regional=('Area AP Toko', 'first'),
        Cluster_Pareto=('Cluster Pareto', 'first')
    )
    agg['Recency'] = (cutoff - agg['last_trx']).dt.days
    
    # Tonnage Drop logic
    recent_ton = df_tr[df_tr['Tanggal Transaksi'] >= (cutoff - pd.Timedelta(days=30))].groupby('ID Toko')['TON Quantity'].sum()
    agg['Tonnage_Drop'] = (recent_ton / (agg['Total_Ton'] / 6 + 0.001)).fillna(0).clip(upper=10)
    
    # Loyalty Status
    if df_loyalty is not None:
        lid = set(df_loyalty['ID Toko'].astype(str).str.strip())
        agg['Is_Loyalty'] = np.where(agg.index.astype(str).isin(lid), 1, 0)
    else:
        agg['Is_Loyalty'] = 0
        
    return agg

def encode_features(agg, features_list=None):
    enc = pd.get_dummies(agg, columns=['Regional'], prefix='Regional', dtype=int)
    enc.columns = [re.sub(r'[\[\]<>{}:",\s]', '_', str(c)) for c in enc.columns]
    
    drop_cols = ['last_trx', 'Label_Churn', 'Cluster_Pareto']
    if features_list:
        enc = enc.reindex(columns=features_list, fill_value=0)
        
    feats = [c for c in enc.columns if c not in drop_cols]
    return enc[feats], (enc['Label_Churn'] if 'Label_Churn' in enc.columns else None)

# ════════════════════════════════════════════════════════════
# 3. TRAINING & MODEL MANAGEMENT
# ════════════════════════════════════════════════════════════

def train_new_model(df, df_loyalty):
    st.info("🔄 Menjalankan Retraining dengan Time-Window Framing...")
    df = cleansing_data(df)
    
    # Tentukan Cutoff Training (Gunakan data 2 bulan terakhir sebagai jendela performa)
    max_date = df['Tanggal Transaksi'].max()
    train_cutoff = max_date - pd.Timedelta(days=60)
    
    # 1. Feature & Labeling
    agg = feature_engineering(df, df_loyalty, train_cutoff)
    labels = buat_label_churn(df, train_cutoff, window_days=60)
    agg_train = agg.join(labels, how='inner')
    
    X, y = encode_features(agg_train)
    
    # 2. SMOTE & Model Fit
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_res, y_res)
    
    # 3. Save Artifacts
    joblib.dump(model, os.path.join(MODEL_DIR, "model_best.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump({"features": list(X.columns), "cutoff": str(max_date.date())}, f)
    
    st.success(f"✅ Model Diperbarui! Pengetahuan Terakhir: {max_date.date()}")

# ════════════════════════════════════════════════════════════
# 4. DASHBOARD INTERFACE
# ════════════════════════════════════════════════════════════

st.sidebar.title("🛠️ Kontrol Panel")
uploaded_data = st.sidebar.file_uploader("Unggah Transaksi (Parquet/CSV)", type=['parquet', 'csv'])
uploaded_loyalty = st.sidebar.file_uploader("Unggah Daftar Loyalty (CSV)", type=['csv'])

if uploaded_data and st.sidebar.button("⚙️ Retrain Model Sekarang"):
    df_new = pd.read_parquet(uploaded_data) if ".parquet" in uploaded_data.name else pd.read_csv(uploaded_data)
    df_l = pd.read_csv(uploaded_loyalty) if uploaded_loyalty else None
    train_new_model(df_new, df_l)

# --- LOAD ASSETS ---
try:
    model = joblib.load(os.path.join(MODEL_DIR, "model_best.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
        meta = json.load(f)
    features_list = meta['features']
except:
    st.warning("⚠️ Belum ada model. Silakan unggah data dan klik 'Retrain Model'.")
    st.stop()

# --- MAIN PAGE ---
if uploaded_data:
    df_raw = pd.read_parquet(uploaded_data) if ".parquet" in uploaded_data.name else pd.read_csv(uploaded_data)
    df_l = pd.read_csv(uploaded_loyalty) if uploaded_loyalty else None
    
    df_clean = cleansing_data(df_raw)
    current_cutoff = df_clean['Tanggal Transaksi'].max()
    
    # Prediksi
    agg = feature_engineering(df_clean, df_l, current_cutoff)
    X_pred, _ = encode_features(agg, features_list)
    
    probs = model.predict_proba(X_pred)[:, 1]
    agg['Probabilitas_Churn'] = probs
    agg['Status_Risiko'] = np.where(probs > 0.26, "🚨 TINGGI", "✅ AMAN")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Toko", len(agg))
    c2.metric("Risiko Churn", len(agg[agg['Status_Risiko'] == "🚨 TINGGI"]))
    c3.metric("Toko Loyalty Berisiko", len(agg[(agg['Status_Risiko'] == "🚨 TINGGI") & (agg['Is_Loyalty'] == 1)]))
    c4.metric("Avg Probabilitas", f"{probs.mean()*100:.1f}%")

    # Tabs
    t1, t2 = st.tabs(["📋 Watchlist Prioritas", "📈 Analisis Trend"])
    
    with t1:
        st.subheader("Daftar Watchlist Strategis")
        f_loyalty = st.checkbox("Hanya Peserta Loyalty Program")
        view_df = agg.copy()
        if f_loyalty: view_df = view_df[view_df['Is_Loyalty'] == 1]
        
        st.dataframe(view_df.sort_values('Probabilitas_Churn', ascending=False), use_container_width=True)
        st.download_button("📥 Ekspor Watchlist", view_df.to_csv().encode('utf-8'), "watchlist.csv")

    with t2:
        st.plotly_chart(px.scatter(agg, x="Recency", y="Tonnage_Drop", color="Status_Risiko", 
                                   size="Monetary", hover_name=agg.index, title="Peta Risiko: Recency vs Tonnage Drop"))

st.markdown("---")
st.caption(f"Dashboard v3.1 | Pengetahuan Model Terakhir: {meta['cutoff']}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import plotly.express as px
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# ════════════════════════════════════════════════════════════
# 1. KONFIGURASI GLOBAL & TAMPILAN
# ════════════════════════════════════════════════════════════
st.set_page_config(page_title="Retail Churn Intelligence", page_icon="🎯", layout="wide")

MODEL_DIR = "models/"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

REGIONAL_MAP = {
    'REGIONAL 1': 'SP',
    'REGIONAL 2': 'SMBR',
    'REGIONAL 6': 'ST'
}

# CSS untuk mempercantik metrik
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stMetric {background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #0052cc;}
    </style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 2. CORE FUNCTIONS (PIPELINE DATA)
# ════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cleansing_data(df):
    df = df.copy()
    for col in ['Tanggal Transaksi', 'Created_at_Dist']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    for col in ['Harga', 'TON Quantity', 'Harga_Per_KG']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    text_cols = ['Nama Toko', 'Area AP Toko', 'Brands', 'Cluster Pareto']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            
    if 'Area AP Toko' in df.columns:
        df['Area AP Toko'] = df['Area AP Toko'].map(REGIONAL_MAP).fillna(df['Area AP Toko'])
            
    df['Est_Spend'] = df['TON Quantity'] * df['Harga_Per_KG'].fillna(0)
    return df

def buat_label_churn(df, cutoff, window_days=60):
    toko_aktif = df[df['Tanggal Transaksi'] <= cutoff]['ID Toko'].unique()
    future_limit = cutoff + pd.Timedelta(days=window_days)
    transaksi_masa_depan = df[(df['Tanggal Transaksi'] > cutoff) & 
                              (df['Tanggal Transaksi'] <= future_limit)]['ID Toko'].unique()
    
    label_list = [{'ID Toko': t, 'Label_Churn': (0 if t in transaksi_masa_depan else 1)} for t in toko_aktif]
    return pd.DataFrame(label_list).set_index('ID Toko')

@st.cache_data(show_spinner=False)
def feature_engineering(df, df_loyalty, cutoff):
    df_tr = df[df['Tanggal Transaksi'] <= cutoff].copy()
    if df_tr.empty: return pd.DataFrame()

    agg = df_tr.groupby('ID Toko').agg(
        last_trx=('Tanggal Transaksi', 'max'),
        Frequency=('No Transaksi', 'count'),
        Total_Ton=('TON Quantity', 'sum'),
        Monetary=('Est_Spend', 'sum'),
        Regional=('Area AP Toko', 'first'),
        Cluster_Pareto=('Cluster Pareto', 'first')
    )
    agg['Recency'] = (cutoff - agg['last_trx']).dt.days
    
    recent_ton = df_tr[df_tr['Tanggal Transaksi'] >= (cutoff - pd.Timedelta(days=30))].groupby('ID Toko')['TON Quantity'].sum()
    agg['Tonnage_Drop'] = (recent_ton / (agg['Total_Ton'] / 6 + 0.001)).fillna(0).clip(upper=10)
    
    if df_loyalty is not None and not df_loyalty.empty:
        col_id = df_loyalty.columns[0]
        lid = set(df_loyalty[col_id].astype(str).str.strip())
        agg['Is_Loyalty'] = np.where(agg.index.astype(str).isin(lid), 1, 0)
    else:
        agg['Is_Loyalty'] = 0
        
    return agg

def encode_features(agg, features_list=None):
    if agg.empty: return pd.DataFrame(), None
    
    enc = pd.get_dummies(agg, columns=['Regional'], prefix='Regional', dtype=int)
    enc.columns = [re.sub(r'[\[\]<>{}:",\s]', '_', str(c)) for c in enc.columns]
    
    # Hapus kolom yang bukan fitur untuk prediksi
    drop_cols = ['last_trx', 'Label_Churn', 'Cluster_Pareto']
    for col in drop_cols:
        if col in enc.columns:
            enc = enc.drop(columns=[col])
            
    # SINKRONISASI PAKSA: Mencegah ValueError jumlah fitur beda
    if features_list:
        enc = enc.reindex(columns=features_list, fill_value=0)
        return enc[features_list], None
        
    return enc, (enc['Label_Churn'] if 'Label_Churn' in enc.columns else None)

# ════════════════════════════════════════════════════════════
# 3. FUNGSI RETRAIN OTOMATIS
# ════════════════════════════════════════════════════════════
def train_new_model(df_raw, df_loyalty):
    with st.spinner("🔄 Sedang melatih ulang model (Sinkronisasi Fitur)..."):
        df_clean = cleansing_data(df_raw)
        max_date = df_clean['Tanggal Transaksi'].max()
        train_cutoff = max_date - pd.Timedelta(days=60)
        
        agg = feature_engineering(df_clean, df_loyalty, train_cutoff)
        labels = buat_label_churn(df_clean, train_cutoff, window_days=60)
        agg_train = agg.join(labels, how='inner')
        
        X, y = encode_features(agg_train)
        
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        
        model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        model.fit(X_res, y_res)
        
        joblib.dump(model, os.path.join(MODEL_DIR, "model_best.pkl"), compress=3)
        with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
            json.dump({"features": list(X.columns), "cutoff": str(max_date.date())}, f)
        
        st.success("✅ Model berhasil diperbarui dan disinkronkan!")
        st.rerun()

# ════════════════════════════════════════════════════════════
# 4. KONTROL PANEL (SIDEBAR)
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2601/2601112.png", width=80)
    st.title("Data Input")
    uploaded_data = st.file_uploader("1. Transaksi (CSV/Parquet)",
